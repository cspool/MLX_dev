import copy
import json
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from mlxsim.model_streaming_pair_events import streaming_pair_events
from mlxsim.model_block_pipeline import compile_block_pipelines
from mlxsim.model_vector_program import vector_program
from mlxsim.model_matrix_program import matrix_program
from mlxsim.model_control_program import control_program
from test_event_schedule import binary
from test_event_loops import run
from test_block_pipeline import matrix_pair
from test_ready_graph import binary as native_binary,execute
from test_physical_model import compiled_nodes,outputs
from test_model_tensor_semantics import node,ref,literal
from scripts.verify_mlx_event_schedule import audit_trace


def compare(binary,native_binary,tmp_path,program,expected,tokens,timed,memory=None):
    memory=memory or dict(latency=8,accept_period=1,nack_every=0)
    native=execute(native_binary,tmp_path/"native",program,dict(tile_pipeline=True,template_load_timing=timed,memory=memory))
    values,actual_tokens=outputs(native)[0];np.testing.assert_allclose(values,expected,rtol=5e-6,atol=5e-6);assert actual_tokens==np.asarray(tokens).reshape(-1).tolist()
    p=streaming_pair_events(program,tmp_path/"patterns",timed_templates=timed,memory=memory)
    r=run(binary,tmp_path/"event",p)
    group=native["pipeline_groups"][0];eg=r["pipeline_groups"][0]
    assert r["cycles"]==group["end_cycle"]-group["begin_cycle"],(r["cycles"],group)
    for key in ("blocks","frontier","admitted","completed","event_slots","peak_event_slots","finished","active","pending_visibility","out_of_order_done"):
        assert eg[key]==group[key],key
    by_source={e["source_operator_id"]:e for e in native["events"]}
    for s in r["source_intervals"]:
        n=by_source[s["source_operator_id"]]
        assert s["begin_cycle"]==n["start_cycle"] and s["publish_cycle"]==n["publish_cycle"]
    assert eg["consumer_context_limit"]==program["block_pipeline_plan"]["pairs"][0]["consumer_context_limit"]
    for ours,theirs in (("contexts","peak_contexts"),("rf_vectors_per_pe","peak_rf_vectors_per_pe"),("spm_vectors","peak_spm_vectors"),("rom_words_per_pe","peak_rom_words_per_pe")):
        assert r["peak"][ours]==native["array"][theirs]
    matrix=native["windows"]["matrix"];vector=native["windows"]["vector"]
    assert r["integrated_usage"].get("compute_inflight_pe_cycles",0)==sum(w["compute_busy_pe_cycles"] for w in matrix)+sum(w["vector_busy_pe_cycles"] for w in vector)
    assert r["integrated_usage"].get("sfu_inflight_pe_cycles",0)==sum(w["trans_busy_pe_cycles"] for w in vector)
    for unit in ("spm","dma"):assert r["integrated_usage"].get(unit+"_inflight_cycles",0)==sum(w[unit+"_busy_cycles"] for w in matrix+vector)
    audit_trace(p,r)
    return p,r,native


@pytest.mark.parametrize("precision",["f16","f32"])
@pytest.mark.parametrize("slots",[1,2,32])
@pytest.mark.parametrize("timed",[False,True])
def test_matrix_pair_uses_actual_block_visibility_and_reserved_resources(binary,native_binary,tmp_path,precision,slots,timed):
    program,expected,tokens=matrix_pair(m=5,n=19,k=1,precision=precision,pes=2,slots=slots)
    _,r,_=compare(binary,native_binary,tmp_path,program,expected,tokens,timed)
    first_consumer=min(e["cycle"] for e in r["trace"] if e["source_operator_id"]==1 and e["event"]=="issue")
    assert first_consumer<r["source_intervals"][0]["publish_cycle"]


def extra_program(kind):
    if kind=="batched":
        a=torch.ones(2,3,1);a[1]*=4;b=torch.tensor([1,4,16]*7).float()[:19].reshape(1,1,19);y=a@b
        producer=node(0,"matmul",[ref("a"),ref("b")],y);producer["matrix_program"]=matrix_program("f32","f32");assets={"a":literal(a),"b":literal(b)}
    else:
        x=torch.ones(32,17) if kind=="mean" else torch.zeros(2,17) if kind=="softmax" else torch.arange(57).float().reshape(3,19)
        assets={"x":literal(x)};args=[ref("x")]
        if kind=="mean":args.extend([[-1],True]);y=x.mean(-1,keepdim=True)
        elif kind=="softmax":args.extend([-1,"torch.float32"]);y=x.softmax(-1)
        else:y=-x
        producer=node(0,kind,args,y);producer["vector_program"]=vector_program(kind,["f32"],"f32",width=17 if kind in {"mean","softmax"} else None)
    consumer_kind="neg" if kind in {"softmax","neg"} else "rsqrt"
    expected=-y if consumer_kind=="neg" else y.rsqrt();tokens=expected.argmax(-1)
    consumer=node(1,consumer_kind,[ref("v0")],expected);consumer["vector_program"]=vector_program(consumer_kind,["f32"],"f32")
    token=node(2,"argmax",[ref("v1"),-1],tokens);token["control_program"]=control_program("argmax","f32")
    p=compiled_nodes(assets,[producer,consumer,token],"v1","v2")
    for family in ("matrix","vector"):p[family+"_schedule_options"]=dict(rows=1,columns=2,trace=True,trace_limit=200000)
    return compile_block_pipelines(p,event_slots=2),expected.numpy(),tokens.tolist()


@pytest.mark.parametrize("kind",["batched","neg","mean","softmax"])
@pytest.mark.parametrize("timed",[False,True])
def test_other_pair_mappings_and_matrix_batch_offsets(binary,native_binary,tmp_path,kind,timed):
    p,expected,tokens=extra_program(kind);assert len(p["block_pipeline_plan"]["pairs"])==1
    compare(binary,native_binary,tmp_path,p,expected,tokens,timed)


@pytest.mark.parametrize("damage",["credit","shape","mapping","consumer","resources","duplicate","closure","dtype","short_write","nonwrite_end"])
def test_invalid_streaming_pair_contract_rejected(binary,tmp_path,damage):
    p,_,_=matrix_pair(m=5,n=19,k=1,pes=2);c=streaming_pair_events(p,tmp_path/"patterns");pair=c["streaming_pairs"][0]
    if damage=="credit":pair["event_slots"]=33;error="pair event slots"
    elif damage=="shape":c["source_streams"][1]["output_shape"]=[1];error="output shapes"
    elif damage=="mapping":pair["mapping"]["n"]+=1;error="extent mismatch"
    elif damage=="consumer":c["source_streams"][1]["operator_kind"]="mean";error="pointwise consumer"
    elif damage=="resources":c["hardware"]["contexts"]=1;error="reserve producer"
    elif damage=="duplicate":c["streaming_pairs"].append(copy.deepcopy(pair));error="disjoint"
    elif damage=="closure":
        source=copy.deepcopy(c["source_streams"][1]);source["source_operator_id"]=2;c["source_streams"].append(source);error="single-consumer"
    elif damage=="dtype":c["source_streams"][0].pop("output_dtype");error="output dtype"
    else:
        key=c["source_streams"][0]["windows"][0]["runs"][0]["event_pattern"];spec=c["event_patterns"][key];file=Path(spec["path"]);body=json.loads(file.read_text())
        if damage=="short_write":body["events"][-1]["repeat"]-=1;spec["dynamic_events"]-=1;error="output volume"
        else:body["events"].append(dict(id="tail",op="mul",active_lanes=1,dependencies=[]));spec["dynamic_events"]+=1;error="final write response"
        raw=json.dumps(body).encode();file.write_bytes(raw);spec["sha256"]=hashlib.sha256(raw).hexdigest()
    run(binary,tmp_path/"bad",c,error)


@pytest.mark.parametrize("latency",[1,3])
@pytest.mark.parametrize("accept_period",[2,4])
def test_registered_acceptance_replaces_frontend_endpoint_latency(binary,native_binary,tmp_path,latency,accept_period):
    p,expected,tokens=matrix_pair(m=5,n=19,k=1,pes=2)
    for family in ("matrix","vector"):p[family+"_schedule_options"].update(dma_latency=101,dma_request_period=2,dma_response_period=3)
    compare(binary,native_binary,tmp_path,p,expected,tokens,True,dict(latency=latency,accept_period=accept_period,nack_every=0))


def test_full_array_consumer_reservation_keeps_space_for_f32_producer(binary,native_binary,tmp_path):
    p,expected,tokens=matrix_pair(m=9,n=33,k=2,precision="f32",pes=16,slots=2)
    _,r,_=compare(binary,native_binary,tmp_path,p,expected,tokens,True)
    assert r["pipeline_groups"][0]["consumer_context_limit"]==11


def test_pair_epochs_array_exclusivity_and_private_control_overlap(binary,tmp_path):
    p,_,_=matrix_pair(m=5,n=19,k=1,pes=2);c=streaming_pair_events(p,tmp_path/"patterns",timed_templates=True)
    a,b=copy.deepcopy(c["source_streams"]);a.update(source_operator_id=2,parents=[1]);b.update(source_operator_id=3,parents=[2]);c["source_streams"]+=[a,b]
    pair=copy.deepcopy(c["streaming_pairs"][0]);pair.update(producer_source=2,consumer_source=3);c["streaming_pairs"].append(pair)
    extra=copy.deepcopy(a);extra.update(source_operator_id=4,parents=[]);c["source_streams"].append(extra)
    value=dict(schema="mlx_block_event_pattern_v1",events=[dict(id="e0",op="control_literal",dependencies=[])]);raw=json.dumps(value).encode();file=tmp_path/"control.json";file.write_bytes(raw)
    c["event_patterns"]["control"]=dict(path=str(file),sha256=hashlib.sha256(raw).hexdigest(),dynamic_events=1,zero_work=False)
    c["source_streams"].append(dict(source_operator_id=5,family="control",parents=[],windows=[dict(runs=[dict(count=1,event_pattern="control")])]))
    r=run(binary,tmp_path/"epochs",c);groups=r["pipeline_groups"];sources={s["source_operator_id"]:s for s in r["source_intervals"]}
    assert [g["epoch"] for g in groups]==[1,2]
    assert groups[1]["begin_cycle"]==groups[0]["end_cycle"] and sources[4]["begin_cycle"]==groups[1]["end_cycle"]
    assert sources[5]["begin_cycle"]==0 and sources[5]["publish_cycle"]==1
    audit_trace(c,r)


def test_unimplemented_physical_retry_policy_is_rejected(binary,tmp_path):
    p,_,_=matrix_pair();c=streaming_pair_events(p,tmp_path/"patterns");c["physical_memory"]["nack_every"]=3
    run(binary,tmp_path/"bad",c,"retry policy not implemented")
