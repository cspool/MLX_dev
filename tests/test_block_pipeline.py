"""Compiler-selected producer/consumer pairs driven by actual write completion."""
import copy
import subprocess

import numpy as np
import pytest
import torch

from mlxsim.model_block_pipeline import compile_block_pipelines
from mlxsim.model_control_program import control_program
from mlxsim.model_matrix_program import matrix_program
from mlxsim.model_vector_program import vector_program
from test_model_tensor_semantics import literal, node, ref, native
from test_physical_model import compiled_nodes, outputs, graph
from test_ready_graph import binary, execute, BUILD


def matrix_pair(m=8,n=16,k=4,precision="f16",pes=1,slots=32):
    dtype=torch.float16 if precision=="f16" else torch.float32
    a=torch.ones(m,k,dtype=dtype);levels=torch.tensor(([1,4,16]*((n+2)//3))[:n],dtype=dtype)
    weight=(levels[:,None]/k).expand(n,k).contiguous();matrix=torch.nn.functional.linear(a,weight);expected=matrix.rsqrt();token=expected.argmax(-1)
    items=[node(0,"linear",[ref("a"),ref("w")],matrix),node(1,"rsqrt",[ref("v0")],expected),node(2,"argmax",[ref("v1"),-1],token)]
    items[0]["matrix_program"]=matrix_program(precision,precision);items[1]["vector_program"]=vector_program("rsqrt",[precision],precision);items[2]["control_program"]=control_program("argmax",precision)
    program=compiled_nodes({"a":literal(a),"w":literal(weight)},items,"v1","v2")
    for kind in ("matrix","vector"):program[kind+"_schedule_options"]={"rows":1 if pes<=4 else 4,"columns":min(pes,4),"trace":True,"trace_limit":200000}
    return compile_block_pipelines(program,event_slots=slots),expected.numpy(),token.tolist()


def inspect_pair(result,producer=0,consumer=1,expect_early=True):
    assert result["classification"]=="ready_graph_bounded_pair_events_not_general_cdc_or_system_acceptance"
    assert not result["complete_cdc_verified"] and not result["inference_performance_eligible"]
    event={r["source_operator_id"]:r for r in result["events"]}
    assert event[consumer]["start_cycle"]<event[producer]["publish_cycle"]
    windows=[w for ws in result["windows"].values() for w in ws]
    p=[w for w in windows if w["source_operator_id"]==producer];c=[w for w in windows if w["source_operator_id"]==consumer]
    assert len(c)==1
    trace=c[0]["trace"]
    assert any(e["event"]=="wait_event" for e in trace) and any(e["event"]=="wake" and e["pipeline"]=="dependency" for e in trace)
    first_issue=min(e["array_cycle"] for e in trace if e["event"]=="issue")
    assert (first_issue<event[producer]["publish_cycle"])==expect_early
    groups=result["pipeline_groups"];assert len(groups)==1
    group=groups[0];assert group["finished"] and group["frontier"]==group["blocks"]==group["admitted"]==group["completed"]
    assert group["peak_event_slots"]<=group["event_slots"]<=32 and group["active"]==group["pending_visibility"]==group["out_of_order_done"]==0
    assert result["array"]["idle"] and result["array"]["peak_rf_vectors_per_pe"]<=16 and result["array"]["peak_spm_vectors"]<=128
    assert result["memory"]["idle"] and result["physical_mux"]["idle"] and result["arena_drained"]["reserved_bytes"]==0
    return p,c


@pytest.mark.parametrize("precision",["f16","f32"])
@pytest.mark.parametrize("slots",[1,2,32])
def test_consumer_resides_then_runs_before_whole_producer_finishes(binary,tmp_path,precision,slots):
    program,expected,tokens=matrix_pair(precision=precision,slots=slots)
    assert len(program["block_pipeline_plan"]["pairs"])==1
    result=execute(binary,tmp_path/"tiles",program,{"tile_pipeline":True,"memory":{"latency":3,"nack_every":5}})
    baseline=execute(binary,tmp_path/"barrier",program,{"tile_pipeline":True,"pipeline_whole_source_barrier":True,"memory":{"latency":3,"nack_every":5}})
    p,c=inspect_pair(result);bp,bc=inspect_pair(baseline,expect_early=False)
    actual,ids=outputs(result)[0];whole,whole_ids=outputs(baseline)[0]
    np.testing.assert_array_equal(actual,expected);np.testing.assert_array_equal(actual.view(np.uint8),whole.view(np.uint8));assert ids==whole_ids==tokens
    assert p[0]["numeric_instructions"]==bp[0]["numeric_instructions"] and c[0]["numeric_instructions"]==bc[0]["numeric_instructions"]


@pytest.mark.parametrize("pes,precision",[(1,"f16"),(4,"f16"),(16,"f32")])
def test_tail_dependencies_and_producer_reservation_do_not_deadlock(binary,tmp_path,pes,precision):
    program,expected,tokens=matrix_pair(m=9,n=19,precision=precision,pes=pes,slots=2)
    result=execute(binary,tmp_path/"tails",program,{"tile_pipeline":True,"memory":{"latency":3,"accept_period":2,"nack_every":7}})
    inspect_pair(result)
    actual,ids=outputs(result)[0];np.testing.assert_array_equal(actual,expected);assert ids==tokens


def test_event_records_are_reused_across_more_than_256_consumer_blocks(binary,tmp_path):
    program,expected,tokens=matrix_pair(m=9,n=513,k=1,slots=2)
    result=execute(binary,tmp_path/"reuse",program,{"tile_pipeline":True,"memory":{"latency":1}})
    inspect_pair(result);assert result["pipeline_groups"][0]["blocks"]>32
    assert result["windows"]["vector"][0]["admitted"]>256
    actual,ids=outputs(result)[0];np.testing.assert_array_equal(actual,expected);assert ids==tokens


def test_completed_prefix_epoch_credit_and_next_edge_contract(binary):
    build=subprocess.run(["cmake","--build",str(BUILD),"--target","completion-window-contract","-j4"],capture_output=True,text=True,timeout=180)
    assert build.returncode==0,build.stdout+build.stderr
    result=subprocess.run([str(BUILD/"completion-window-contract")],capture_output=True,text=True,timeout=30)
    assert result.returncode==0 and "COMPLETION_WINDOW_CONTRACT_PASS" in result.stdout,result.stdout+result.stderr


@pytest.mark.parametrize("damage",["mapping","counts","resources","epochs","missing"])
def test_unbound_or_corrupted_pipeline_plans_are_rejected(binary,tmp_path,damage):
    program,_,_=matrix_pair();pair=program["block_pipeline_plan"]["pairs"][0]
    if damage=="mapping":pair["mapping"]["m"]+=1;error="extent mismatch"
    elif damage=="counts":pair["producer_blocks"]+=1;error="block count differs"
    elif damage=="resources":pair["consumer_context_limit"]+=1;error="reservation differs"
    elif damage=="epochs":program["block_pipeline_plan"]["event_slots"]=33;error="event capacity"
    else:del program["block_pipeline_plan"];error="unsupported block pipeline plan"
    execute(binary,tmp_path/damage,program,{"tile_pipeline":True},error)


def test_pipeline_lowering_never_changes_the_numerical_graph():
    program,_,_=matrix_pair();plan=program.pop("block_pipeline_plan");rebuilt=compile_block_pipelines(program)
    assert rebuilt.pop("block_pipeline_plan")==plan and rebuilt==program


@pytest.mark.parametrize("producer",["vector","mean"])
def test_vector_and_reduction_producers_publish_real_block_results(binary,tmp_path,producer):
    if producer=="mean":
        x=torch.tensor([1,4,16]*11,dtype=torch.float16).reshape(33,1).expand(33,16).contiguous();p=x.mean(-1,keepdim=True)
        items=[node(0,"mean",[ref("x"),[-1],True],p)];items[0]["vector_program"]=vector_program("mean",["f16"],"f16",width=16)
        expected=p.rsqrt();items.append(node(1,"rsqrt",[ref("v0")],expected));items[1]["vector_program"]=vector_program("rsqrt",["f16"],"f16")
    else:
        x=torch.arange(513,dtype=torch.float16).reshape(1,513);p=-x
        items=[node(0,"neg",[ref("x")],p)];items[0]["vector_program"]=vector_program("neg",["f16"],"f16")
        expected=-p;items.append(node(1,"neg",[ref("v0")],expected));items[1]["vector_program"]=vector_program("neg",["f16"],"f16")
    token=expected.argmax(-1);items.append(node(2,"argmax",[ref("v1"),-1],token));items[2]["control_program"]=control_program("argmax","f16")
    program=compiled_nodes({"x":literal(x)},items,"v1","v2")
    for family in ("matrix","vector"):program[family+"_schedule_options"]={"rows":1,"columns":1,"trace":True,"trace_limit":200000}
    program=compile_block_pipelines(program,event_slots=2);assert len(program["block_pipeline_plan"]["pairs"])==1
    result=execute(binary,tmp_path/producer,program,{"tile_pipeline":True,"memory":{"nack_every":5}})
    inspect_pair(result);actual,ids=outputs(result)[0];np.testing.assert_array_equal(actual,expected.numpy());assert ids==token.flatten().tolist()


def test_matrix_batch_offsets_keep_completion_events_unique(binary,tmp_path):
    a=torch.arange(30,dtype=torch.float32).reshape(2,1,3,5)/16;w=torch.arange(140,dtype=torch.float32).reshape(1,4,7,5)/32;b=w.transpose(-2,-1)
    p=a@b;expected=-p;token=expected.argmax(-1)
    items=[node(0,"transpose",[ref("w"),-2,-1],b),node(1,"matmul",[ref("a"),ref("v0")],p),node(2,"neg",[ref("v1")],expected),node(3,"argmax",[ref("v2"),-1],token)]
    items[1]["matrix_program"]=matrix_program("f32","f32");items[2]["vector_program"]=vector_program("neg",["f32"],"f32");items[3]["control_program"]=control_program("argmax","f32")
    program=compiled_nodes({"a":literal(a),"w":literal(w)},items,"v2","v3")
    for family in ("matrix","vector"):program[family+"_schedule_options"]={"rows":1,"columns":1,"trace":True,"trace_limit":200000}
    program=compile_block_pipelines(program,event_slots=1)
    result=execute(binary,tmp_path/"batches",program,{"tile_pipeline":True,"memory":{"nack_every":7}})
    p_windows,_=inspect_pair(result,producer=1,consumer=2);assert len(p_windows)==8
    assert [w["block_flow"]["block_offset"] for w in p_windows]==list(range(0,16,2))
    actual,ids=outputs(result)[0];np.testing.assert_array_equal(actual,expected.numpy());assert ids==token.flatten().tolist()


def test_generated_token_and_cache_graph_crosses_distinct_pipeline_epochs(binary,graph,tmp_path):
    program=compile_block_pipelines(graph[0],event_slots=2);assert len(program["block_pipeline_plan"]["pairs"])>=3
    result=execute(binary,tmp_path/"generation",program,{"tile_pipeline":True,"memory":{"nack_every":5}})
    for (actual,ids),(expected,expected_ids) in zip(outputs(result),outputs(graph[1]),strict=True):
        np.testing.assert_array_equal(actual.view(np.uint8),expected.view(np.uint8));assert ids==expected_ids
    groups=result["pipeline_groups"];assert len(groups)==len(program["block_pipeline_plan"]["pairs"])
    assert len({g["epoch"] for g in groups})==len(groups) and all(g["finished"] and g["peak_event_slots"]<=2 for g in groups)


def test_unsupported_pair_capacity_keeps_existing_operator_routes():
    program,_,_=matrix_pair();original=copy.deepcopy(program["nodes"])
    program["matrix_schedule_options"]["contexts"]=1;program["vector_schedule_options"]["contexts"]=1
    result=compile_block_pipelines(program)
    assert result["nodes"]==original and not result["block_pipeline_plan"]["pairs"]
    assert result["block_pipeline_plan"]["rejected_pairs"][0]["reason"]=="producer_consumer_need_two_contexts_per_pe"


@pytest.mark.parametrize("field,value",[("spm_bytes_used",0),("spm_bytes_used",2**32-1),("rf_vectors_used",2**32-1)])
def test_malformed_storage_requirements_fail_before_reservation_arithmetic(binary,tmp_path,field,value):
    program,_,_=matrix_pair()
    program["nodes"][1]["vector_program"][field]=value
    execute(binary,tmp_path/"invalid-storage",program,{"tile_pipeline":True},"invalid pipeline storage requirement")
