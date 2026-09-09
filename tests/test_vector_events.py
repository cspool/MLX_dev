import copy

import numpy as np
import pytest
import torch

from mlxsim.model_vector_events import vector_events
from mlxsim.model_vector_program import FLOAT_KINDS,vector_program
from mlxsim.model_dtype_lowering import lower_softmax_input_cast
from test_model_tensor_semantics import literal,node,ref,execute_nodes,native
from test_vector_window_scheduler import binary as vector_binary,run as run_vector
from test_event_schedule import binary
from test_event_loops import run


def job(kind,width=17,rows=2,precision="f16",scalar=False,**options):
    dtype=torch.float16 if precision=="f16" else torch.float32
    x=torch.linspace(-2,2,width*rows).reshape(rows,width).to(dtype)
    if kind=="rsqrt":x=x.abs()+0.5
    assets={"x":literal(x)};args=[ref("x")];types=[precision]
    if kind in {"add","sub","mul","div","maximum"}:
        if scalar:args.append(0.75);types.append("f32")
        else:
            assets["y"]=literal(torch.linspace(0.5,1.5,width).to(dtype));args.append(ref("y"));types.append(precision)
    expected=x.clone()
    if kind=="pow":args.append(2)
    if kind=="mean":args.extend([[-1],True]);expected=x.mean(-1,keepdim=True)
    if kind=="softmax":args.extend([-1,"torch.float32"]);expected=x.softmax(-1,dtype=torch.float32)
    n=node(0,kind,args,expected)
    n["vector_program"]=vector_program(kind,types,"f16" if expected.dtype==torch.float16 else "f32",width=width if kind in {"mean","softmax"} else None)
    return dict(schema="mlx_vector_window_job_v1",assets=assets,node=n,options=dict(rows=1,columns=2,**options))


def compare(binary,vector_binary,native,tmp_path,j):
    expected=execute_nodes((native[0],"none"),tmp_path/"microcode",j["assets"],[j["node"]])[0]
    observed,actual=run_vector(vector_binary,tmp_path/"native",j)
    np.testing.assert_array_equal(actual.view(np.uint8),expected.view(np.uint8))
    result=run(binary,tmp_path/"event",vector_events(j))
    assert result["cycles"]==observed["cycles"],(result["cycles"],observed["cycles"])
    usage=result["integrated_usage"];work=result["declared_work"]
    assert work.get("dma_read_bytes",0)==observed["numeric_instructions"]["global_read_bytes"]
    assert work["dma_write_bytes"]==observed["numeric_instructions"]["global_write_bytes"]
    instructions=sum(v for k,v in work.items() if k.endswith("_events") and k not in {"operand_prepare_events","spm_initialize_events","dma_read_events","dma_write_events"})
    assert instructions==observed["numeric_instructions"]["instructions"]
    sfu={"exp","div","sqrt","cos","sin"};arithmetic=sfu|{"mul","add","sub","maximum","neg"}
    assert sum(work.get(op+"_declared_active_lanes",0) for op in sfu)==observed["numeric_instructions"]["transcendental_lanes"]
    assert sum(work.get(op+"_declared_active_lanes",0) for op in arithmetic)==observed["numeric_instructions"]["arithmetic_lanes"]
    for ours,theirs in (("compute_inflight_pe_cycles","vector_busy_pe_cycles"),("sfu_inflight_pe_cycles","trans_busy_pe_cycles"),
                       ("spm_inflight_cycles","spm_busy_cycles"),("dma_inflight_cycles","dma_busy_cycles"),
                       ("compute_sfu_inflight_overlap_pe_cycles","vector_sfu_overlap_pe_cycles"),
                       ("same_pe_inflight_context_overlap_pe_cycles","same_pe_context_overlap_cycles")):
        assert usage.get(ours,0)==observed[theirs],ours
    return result,observed


@pytest.mark.parametrize("kind",sorted(FLOAT_KINDS))
@pytest.mark.parametrize("precision",["f16","f32"])
def test_complete_vector_windows_match_native_values_cycles_and_resource_use(binary,vector_binary,native,tmp_path,kind,precision):
    compare(binary,vector_binary,native,tmp_path,job(kind,precision=precision))


@pytest.mark.parametrize("kind",["add","sub","mul","div","maximum"])
def test_literal_operand_initialization_is_not_fake_dma(binary,vector_binary,native,tmp_path,kind):
    j=job(kind,width=3,rows=1,precision="f32",scalar=True)
    result,_=compare(binary,vector_binary,native,tmp_path,j)
    assert result["declared_work"]["dma_read_bytes"]==12
    assert result["declared_work"]["spm_initialize_events"]==2


@pytest.mark.parametrize("kind",["mean","softmax"])
@pytest.mark.parametrize("width",[1,3,65])
def test_reduction_padding_carry_stack_and_periods(binary,vector_binary,native,tmp_path,kind,width):
    compare(binary,vector_binary,native,tmp_path,job(kind,width=width,rows=2,precision="f32",spm_period=2,writeback_period=3,dma_request_period=3,dma_response_period=2,vector_ii=4,trans_ii=7))


@pytest.mark.parametrize("kind",["mean","softmax"])
@pytest.mark.parametrize("width",[129,4096])
def test_long_reduction_keeps_all_chunks_and_padding(binary,vector_binary,native,tmp_path,kind,width):
    compare(binary,vector_binary,native,tmp_path,job(kind,width=width,rows=1,precision="f32",trace=False))


def test_softmax_narrows_input_at_the_registered_microprogram_point(binary,vector_binary,native,tmp_path):
    j=job("softmax",width=19,rows=2,precision="f32")
    j["node"]["output"]["dtype"]="f16";j["node"]["args"][2]="torch.float16"
    j["node"]["vector_program"]=vector_program("softmax",["f32"],"f16",width=19)
    lower_softmax_input_cast(j["node"])
    compare(binary,vector_binary,native,tmp_path,j)


@pytest.mark.parametrize("kind",["add","sub"])
def test_nonunit_alpha_remains_in_microprogram_not_timing_shortcut(binary,vector_binary,native,tmp_path,kind):
    j=job(kind,width=9,rows=2,precision="f16")
    j["node"]["kwargs"]["alpha"]=0.25
    j["node"]["vector_program"]=vector_program(kind,["f16","f16"],"f16",alpha=0.25)
    compare(binary,vector_binary,native,tmp_path,j)


def test_full_array_residency_is_not_automatically_vector_sfu_overlap(binary,vector_binary,native,tmp_path):
    j=job("silu",width=512,rows=1,precision="f32",dma_latency=1,exp_latency=30,div_latency=40,trace=False)
    j["options"].update(rows=4,columns=4)
    event_result,observed=compare(binary,vector_binary,native,tmp_path,j)
    assert event_result["peak"]["contexts"]==observed["peak_contexts"]==25
    assert event_result["integrated_usage"].get("compute_sfu_inflight_overlap_pe_cycles",0)==observed["vector_sfu_overlap_pe_cycles"]==0


def test_same_pe_vector_and_sfu_overlap_is_preserved_when_work_is_ready(binary,vector_binary,native,tmp_path):
    j=job("silu",width=64,rows=2,precision="f16",dma_latency=1,exp_latency=30,div_latency=40)
    j["options"].update(rows=1,columns=1)
    event_result,observed=compare(binary,vector_binary,native,tmp_path,j)
    assert event_result["integrated_usage"]["compute_sfu_inflight_overlap_pe_cycles"]==observed["vector_sfu_overlap_pe_cycles"]>0


@pytest.mark.parametrize("damage",["serial","word","shape","input","period"])
def test_vector_events_reject_incomplete_or_changed_contract(damage):
    j=job("silu")
    if damage=="serial":j["options"]["overlap"]=False
    elif damage=="word":j["node"]["vector_program"]["phases"]["body"].pop()
    elif damage=="shape":j["node"]["output"]["shape"]=[1]
    elif damage=="input":j["assets"]["x"]["dtype"]="bool"
    else:j["options"]["trans_ii"]=0
    with pytest.raises(ValueError):vector_events(j)
