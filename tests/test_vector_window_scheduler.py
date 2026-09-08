import copy
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from mlxsim.model_vector_program import FLOAT_KINDS, vector_program
from mlxsim.model_dtype_lowering import lower_softmax_input_cast
from test_model_tensor_semantics import execute_nodes, literal, native, node, ref

ROOT=Path(__file__).resolve().parents[1]
BUILD=ROOT/"build/mlx-vector-window"


@pytest.fixture(scope="module")
def binary():
    for command in (["cmake","-S",str(ROOT/"simulator_ext/vector_schedule"),"-B",str(BUILD),"-DCMAKE_BUILD_TYPE=Release"], ["cmake","--build",str(BUILD),"--target","mlx-vector-window","-j4"]):
        result=subprocess.run(command,capture_output=True,text=True,timeout=180)
        assert result.returncode==0,result.stdout+result.stderr
    return BUILD/"mlx-vector-window"


def make_job(kind="silu",width=17,rows=3,precision="f16",**options):
    dtype=torch.float16 if precision=="f16" else torch.float32
    x=torch.linspace(-3,3,rows*width).reshape(rows,width).to(dtype)
    if kind=="rsqrt":x=x.abs()+0.25
    assets={"x":literal(x)};args=[ref("x")];dtypes=[precision];expected=x.clone()
    if kind in {"add","mul"}:
        y=torch.linspace(-1,1,width).to(dtype);assets["y"]=literal(y);args.append(ref("y"));dtypes.append(precision)
    if kind=="pow":args.append(2)
    if kind=="mean":args+=[[-1],True];expected=x.mean(-1,keepdim=True)
    if kind=="softmax":args +=[-1,"torch.float32"];expected=x.softmax(-1,dtype=torch.float32)
    item=node(0,kind,args,expected)
    item["vector_program"]=vector_program(kind,dtypes,"f16" if expected.dtype==torch.float16 else "f32",width=width if kind in {"mean","softmax"} else None)
    return {"schema":"mlx_vector_window_job_v1","assets":assets,"node":item,"options":{"rows":1,"columns":1,**options}}


def run(binary,directory,job,*,error=None):
    directory.mkdir();(directory/"job.json").write_text(json.dumps(job))
    result=subprocess.run([str(binary),str(directory/"job.json"),str(directory/"out")],capture_output=True,text=True,timeout=180)
    if error:
        assert result.returncode!=0 and error in result.stderr,result.stdout+result.stderr
        return
    assert result.returncode==0,result.stdout+result.stderr
    report=json.loads((directory/"out/result.json").read_text())
    dtype=np.float16 if job["node"]["output"]["dtype"]=="f16" else np.float32
    actual=np.fromfile(directory/"out/output.bin",dtype=dtype).reshape(job["node"]["output"]["shape"])
    assert report["done"] and report["admitted"]==report["retired"]
    assert not any(report[key] for key in ("pending_dma","pending_spm","pending_fu","active_contexts","allocated_spm_vectors"))
    assert report["dma_requests"]==report["dma_responses"]
    assert not report["mlx_system_verified"] and not report["inference_performance_eligible"]
    return report,actual


def audit(report):
    active={};last_retire={};ready={};issued={};dma=None
    issue_ports=Counter();write_ports=Counter()
    for event in report["trace"]:
        slot=(event["pe"],event["context_slot"]);identity=(event["block_id"],event["epoch"])
        key=identity+(event["phase"],event["pc"],event["base"],event["level"])
        if event["event"]=="admit":
            assert slot not in active and event["cycle"]>last_retire.get(slot,-1)
            active[slot]=identity;ready[identity]=event["cycle"]
            assert len(active)*5<=128
        else:assert active[slot]==identity
        if event["event"] in {"issue","predicated_issue"}:
            issue_ports[event["cycle"],event["pe"]]+=1
            assert event["cycle"]>ready.get(identity,-1)
            if event["event"]=="issue":
                assert key not in issued;issued[key]=event
                if event["pipeline"]=="trans":assert int(event["lane_mask"]).bit_count()<=4
            else:ready[identity]=event["cycle"]
        if event["event"]=="complete":
            start=issued.pop(key);assert event["cycle"]>start["cycle"]
            assert event["opcode"]==start["opcode"] and event["lane_mask"]==start["lane_mask"]
            ready[identity]=event["cycle"]
            if event["opcode"] not in (8,11,27):write_ports[event["cycle"],event["pe"]]+=1
        if event["event"]=="operand_ready":ready[identity]=event["cycle"]
        if event["event"]=="dma_request":assert dma is None;dma=event
        if event["event"]=="dma_response":
            assert dma is not None and event["request_id"]==dma["request_id"] and event["cycle"]>dma["cycle"]
            for field in ("block_id","epoch","phase","pc","base","level","move","region","byte_offset","bytes","write","spm_byte_offset"):assert event[field]==dma[field]
            dma=None
        if event["event"]=="retire":
            assert not any(k[:2]==identity for k in issued)
            del active[slot];last_retire[slot]=event["cycle"]
    assert not active and not issued and dma is None
    assert max(issue_ports.values(),default=0)<=1 and max(write_ports.values(),default=0)<=1


@pytest.mark.parametrize("kind",sorted(FLOAT_KINDS))
@pytest.mark.parametrize("precision",["f16","f32"])
def test_scheduled_vector_matches_functional_microinstructions(binary,native,tmp_path,kind,precision):
    job=make_job(kind,precision=precision)
    expected=execute_nodes((native[0],"none"),tmp_path/"functional",job["assets"],[job["node"]])[0]
    reference=json.loads((tmp_path/"functional/out/result.json").read_text())["vector_microcode"]
    report,actual=run(binary,tmp_path/"scheduled",job)
    np.testing.assert_array_equal(actual.view(np.uint16 if actual.dtype==np.float16 else np.uint32),expected.view(np.uint16 if expected.dtype==np.float16 else np.uint32))
    assert report["numeric_instructions"]==reference
    audit(report)


@pytest.mark.parametrize("kind",["mean","softmax"])
@pytest.mark.parametrize("width",[1,3,65,4096])
def test_reduction_state_continues_across_chunks_and_ready_stalls(binary,native,tmp_path,kind,width):
    job=make_job(kind,width=width,rows=2,precision="f32",dma_request_period=3,dma_response_period=5,spm_period=2,writeback_period=4,trace=width<100)
    expected=execute_nodes((native[0],"none"),tmp_path/"functional",job["assets"],[job["node"]])[0]
    report,actual=run(binary,tmp_path/"stalled",job)
    np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
    assert report["writeback_stall_response_cycles"]>0
    if width<100:audit(report)


def test_vector_sfu_can_overlap_without_extra_registers_or_memory(binary,tmp_path):
    job=make_job("silu",width=64,rows=2,dma_latency=1,exp_latency=30,div_latency=40)
    parallel,a=run(binary,tmp_path/"overlap",job)
    job["options"]["overlap"]=False
    serial,b=run(binary,tmp_path/"serial",job)
    np.testing.assert_array_equal(a.view(np.uint16),b.view(np.uint16))
    assert parallel["same_pe_context_overlap_cycles"]>0 and serial["same_pe_context_overlap_cycles"]==0
    assert parallel["vector_sfu_overlap_pe_cycles"]>0
    assert parallel["peak_contexts"]==serial["peak_contexts"]==2
    assert parallel["numeric_instructions"]==serial["numeric_instructions"]
    assert parallel["cycles"]<serial["cycles"]
    audit(parallel);audit(serial)


def test_shared_spm_limits_array_residency(binary,tmp_path):
    report,_=run(binary,tmp_path/"array",make_job("silu",width=1024,rows=1,precision="f32",columns=4))
    assert report["peak_contexts"]<=8 and report["peak_spm_vectors"]==report["peak_contexts"]*5
    audit(report)


def test_full_array_cannot_duplicate_shared_spm(binary,tmp_path):
    job=make_job("silu",width=1024,rows=1,precision="f32",columns=4)
    job["options"]["rows"]=4
    report,_=run(binary,tmp_path/"full-array",job)
    assert report["peak_contexts"]==25
    assert report["peak_spm_vectors"]==125
    assert report["spm_capacity_vectors"]==128 and report["admission_stall_cycles"]>0
    audit(report)


@pytest.mark.parametrize("options,error",[({"contexts":3},"capacity"),({"spm_bytes":65536},"unknown"),({"inject_stale_response":True},"response token"),({"max_cycles":2},"cycle bound")])
def test_bad_resource_response_and_deadline_are_rejected(binary,tmp_path,options,error):
    run(binary,tmp_path/"invalid",make_job(**options),error=error)


@pytest.fixture(scope="module")
def external_binary(binary):
    result=subprocess.run(["cmake","--build",str(BUILD),"--target","vector-external-memory","-j4"],capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stdout+result.stderr
    return BUILD/"vector-external-memory"


@pytest.mark.parametrize("fault,error",[("",None),("token","response token"),("error","reported error"),("unsolicited","unsolicited"),("unstable","backpressure"),("withdraw","backpressure")])
def test_external_memory_is_authoritative_not_local_tensor_data(external_binary,tmp_path,fault,error):
    job=make_job("rsqrt",width=4,rows=1,precision="f32")
    # Virtual tensor descriptors have no backing pointer in the C++ driver.
    # Only the external port can supply values or observe stores.
    job["assets"]["x"]["values"]=[[10000.0]*4]
    job["external_values"]={"x":literal(torch.tensor([[1.0,4.0,9.0,16.0]]))}
    job["fault"]=fault
    job["options"]["dma_response_period"]=13
    if error:
        run(external_binary,tmp_path/"external",job,error=error)
    else:
        report,actual=run(external_binary,tmp_path/"external",job)
        expected=np.array([[1.0,0.5,np.float32(1)/np.float32(3),0.25]],dtype=np.float32)
        np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
        assert report["virtual_tensor_backing_used"] and report["external_memory_port"]
        audit(report)
