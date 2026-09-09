import copy
import json

import numpy as np
import pytest
import torch

from mlxsim.model_memory_events import memory_events
from test_event_schedule import binary,event
from test_event_loops import run
from test_memory_window_scheduler import binary as memory_binary,job,run as native_run
from test_model_tensor_semantics import literal,node,ref
from test_split_value_outputs import split_node


def compare(binary,memory_binary,tmp_path,spec):
    spec=copy.deepcopy(spec);spec["external"]=False
    native,actual=native_run(memory_binary,tmp_path/"native",spec)
    n=spec["nodes"][-1];observed=native["windows"][-1]
    request=dict(schema="mlx_memory_event_window_job_v1",node=n,options=spec.get("options",{}))
    if n["kind"]=="where":
        # Only actual predicate data is passed on. No cycles, durations or
        # source timing are input to the independent event simulation.
        request["predicate_choices"]=[bool(e["data"]) for e in observed["trace"] if e["event"]=="response" and (n["memory_program"]["words"][e["pc"]]&255)==5]
    (tmp_path/"memory-event-job.json").write_text(json.dumps(request))
    p=memory_events(request);result=run(binary,tmp_path/"event",p)
    assert result["block_intervals"][0]["window_cycles"]==observed["cycles"],(result["cycles"],observed["cycles"])
    work=result["declared_work"];reads=sum(work.get(op+"_bytes",0) for op in ("dma_read","memory_index_read","memory_predicate_read"))
    assert reads==observed["dma_read_bytes"] and work.get("dma_write_bytes",0)==observed["dma_write_bytes"]
    assert work.get("memory_index_read_events",0)==observed["numeric_instructions"]["index_reads"]
    assert work.get("memory_predicate_read_events",0)==observed["numeric_instructions"]["predicate_reads"]
    instructions=sum(value for key,value in work.items() if key.endswith("_events") and key!="memory_complete_events")
    assert instructions==observed["numeric_instructions"]["instructions"]
    usage=result["integrated_usage"]
    assert usage.get("dma_inflight_cycles",0)-observed["dma_requests"]==observed["response_stalls"]
    assert usage.get("memory_conversion_inflight_cycles",0)-work.get("memory_convert_events",0)==observed["conversion_stalls"]
    assert usage.get("memory_controller_resource_wait_context_cycles",0)+usage.get("memory_controller_dma_request_wait_context_cycles",0)==observed["request_stalls"]
    assert result["peak"]["contexts"]==result["peak"]["spm_vectors"]==result["peak"]["rf_vectors_per_pe"]==result["peak"]["rom_words_per_pe"]==0
    return result,actual,request


@pytest.mark.parametrize("dtype",[torch.float16,torch.float32,torch.int64,torch.bool])
@pytest.mark.parametrize("periods",[False,True])
def test_strided_cast_memory_cycles_and_real_values(binary,memory_binary,tmp_path,dtype,periods):
    x=torch.arange(15).float().reshape(3,5);t=x.T;expected=t.to(dtype,copy=True)
    spec=job({"x":literal(x)},[node(0,"transpose",[ref("x"),0,1],t),node(1,"cast",[ref("v0"),str(dtype),False,True],expected)])
    if periods:spec["options"]=dict(dma_latency=5,request_period=3,response_period=7,convert_latency=4)
    _,actual,_=compare(binary,memory_binary,tmp_path,spec);np.testing.assert_array_equal(actual,expected.numpy())


@pytest.mark.parametrize("kind",["embedding","advanced_index","cat","new_ones","where","where_literal"])
def test_all_memory_selectors_consume_the_declared_work(binary,memory_binary,tmp_path,kind):
    x=torch.arange(12).float().reshape(3,4);assets={"x":literal(x)}
    if kind=="embedding":
        ids=torch.tensor([2,0]);assets["ids"]=literal(ids);expected=x[ids];args=[ref("x"),ref("ids")]
    elif kind=="advanced_index":
        a=torch.tensor([[0],[-1]]);b=torch.tensor([[3,0]]);assets.update(a=literal(a),b=literal(b));expected=x[a,b];args=[ref("x"),[ref("a"),ref("b")]]
    elif kind=="cat":expected=torch.cat([x,x],-1);args=[[ref("x"),ref("x")],-1]
    elif kind=="new_ones":expected=torch.ones(2,3);args=[ref("x"),[2,3]]
    else:
        predicate=torch.tensor([[True],[False],[True]]);assets["p"]=literal(~predicate)
        other=-x if kind=="where" else -3.0
        if kind=="where":assets["y"]=literal(other)
        expected=torch.where(predicate,x,other);args=[ref("p"),ref("x"),ref("y") if kind=="where" else other]
    actual_kind="where" if kind.startswith("where") else kind
    spec=job(assets,[node(0,actual_kind,args,expected)],options=dict(dma_latency=3,request_period=2,response_period=5,convert_latency=3))
    if kind.startswith("where"):spec["physical_values"]={"p":literal(predicate)}
    _,actual,_=compare(binary,memory_binary,tmp_path,spec);np.testing.assert_array_equal(actual,expected.numpy())


@pytest.mark.parametrize("kind",["view","split","empty_transfer","empty_embedding"])
def test_zero_output_and_view_do_not_invent_pe_or_dma_work(binary,memory_binary,tmp_path,kind):
    x=torch.arange(6).float().reshape(2,3);assets={"x":literal(x)}
    if kind=="view":n=node(0,"transpose",[ref("x"),0,1],x.T)
    elif kind=="split":n,_=split_node(0,"x",x,1,1)
    elif kind=="empty_transfer":n=node(0,"new_ones",[ref("x"),[0]],torch.empty(0))
    else:
        assets={"w":literal(torch.empty(4,0)),"i":literal(torch.tensor([1,3]))};n=node(0,"embedding",[ref("w"),ref("i")],torch.empty(2,0))
    result,_,_=compare(binary,memory_binary,tmp_path,job(assets,[n]))
    if kind=="empty_embedding":assert result["declared_work"]["memory_index_read_events"]==2
    else:
        assert result["block_intervals"][0]["window_cycles"]==0
        assert result["memory_controller_resources"]["peak_active"]==0


def test_mixed_tensor_literal_where_requires_actual_choices():
    x=torch.ones(3);p=torch.tensor([True,False,True])
    spec=job({"x":literal(x),"p":literal(p)},[node(0,"where",[ref("p"),ref("x"),0.0],x)])
    request=dict(schema="mlx_memory_event_window_job_v1",node=spec["nodes"][0])
    with pytest.raises(ValueError,match="actual predicate"):memory_events(request)
    request["predicate_choices"]=[True,False]
    with pytest.raises(ValueError,match="witness"):memory_events(request)


def test_memory_conversion_overlaps_array_compute_without_spm_claim(binary,tmp_path):
    x=torch.ones(1);spec=job({"x":literal(x)},[node(0,"new_ones",[ref("x"),[1]],x)])
    p=memory_events(dict(schema="mlx_memory_event_window_job_v1",node=spec["nodes"][0],options=dict(convert_latency=10)))
    p["templates"]=[dict(id="t",words=[1],rf_vectors=8,spm_vectors=5)]
    p["blocks"]=[dict(id="array",source_operator_id=1,pe=0,template="t",admission_dependencies=[],events=[event("compute")])]
    r=run(binary,tmp_path/"mixed",p)
    assert r["peak"]["contexts"]==1 and r["memory_controller_resources"]["peak_active"]==1
    assert r["counts"].get("spm_port_claims",0)==0
    assert r["integrated_usage"]["memory_conversion_inflight_cycles"]==10
    assert r["integrated_usage"]["compute_inflight_pe_cycles"]==4


def test_controller_and_array_share_one_bus_but_not_spm(binary,tmp_path):
    x=torch.ones(1);spec=job({"x":literal(x)},[node(0,"cast",[ref("x"),"torch.float16",False,True],x.half())])
    p=memory_events(dict(schema="mlx_memory_event_window_job_v1",node=spec["nodes"][0]))
    p["templates"]=[dict(id="t",words=[1],rf_vectors=8,spm_vectors=5)]
    p["blocks"]=[dict(id="array",source_operator_id=1,pe=0,template="t",admission_dependencies=[],events=[event("array-read","dma_read")])]
    r=run(binary,tmp_path/"bus",p)
    starts={row["operation"]:row["cycle"] for row in r["trace"] if row["event"]=="issue"}
    assert starts=={"memory:e0":0,"memory:e1":9,"array-read":9,"memory:e2":18}
    assert r["cycles"]==27 and r["counts"]["spm_port_claims"]==1
    assert r["integrated_usage"]["memory_controller_resource_wait_context_cycles"]==6


def test_memory_controller_periods_restart_at_each_source_origin(binary,tmp_path):
    x=torch.ones(1);spec=job({"x":literal(x)},[node(0,"new_ones",[ref("x"),[1]],x)])
    p=memory_events(dict(schema="mlx_memory_event_window_job_v1",node=spec["nodes"][0],options=dict(request_period=3,response_period=5)))
    second=copy.deepcopy(p["controllers"][0]);second.update(id="second",source_operator_id=1,admission_dependencies=["memory"])
    for e in second["events"][0]["body"]:e["id"]="second:"+e["id"]
    p["controllers"].append(second)
    r=run(binary,tmp_path/"origins",p)
    assert [row["window_cycles"] for row in r["block_intervals"]]==[16,16]
    assert r["block_intervals"][1]["admit_cycle"]==16


@pytest.mark.parametrize("field",["data_register_bytes","staging_bytes","max_active"])
def test_memory_controller_capacity_cannot_expand(binary,tmp_path,field):
    x=torch.ones(1);spec=job({"x":literal(x)},[node(0,"new_ones",[ref("x"),[1]],x)])
    p=memory_events(dict(schema="mlx_memory_event_window_job_v1",node=spec["nodes"][0]))
    p["hardware"]["memory_controller"][field]*=2
    run(binary,tmp_path/field,p,"resources differ")


@pytest.mark.parametrize("kind",["dropout_inference","cast","new_ones"])
def test_lowering_does_not_drop_source_flags_or_dtype(kind):
    x=torch.ones(1)
    args=[ref("x"),0.5,False] if kind=="dropout_inference" else [ref("x"),"torch.float32",False,True] if kind=="cast" else [ref("x"),[1]]
    n=job({"x":literal(x)},[node(0,kind,args,x)])["nodes"][0]
    if kind=="dropout_inference":n["args"][2]=True
    elif kind=="cast":n["args"][1]="torch.float16"
    else:n["kwargs"]["pin_memory"]=True
    with pytest.raises(ValueError):memory_events(dict(schema="mlx_memory_event_window_job_v1",node=n))
