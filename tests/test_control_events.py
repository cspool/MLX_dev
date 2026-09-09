import copy
import json

import numpy as np
import pytest
import torch

from mlxsim.model_control_events import control_events
from test_control_window_scheduler import binary as control_binary,make_job,run as native_run
from test_event_schedule import binary,event
from test_event_loops import run
from test_model_tensor_semantics import literal,ref


def request_from_execution(spec,native):
    layouts={name:dict(shape=spec.get("layouts",{}).get(name,{}).get("shape",a["shape"]),dtype=a["dtype"]) for name,a in spec["assets"].items()}
    request=dict(schema="mlx_control_event_window_job_v1",node=spec["node"],input_layouts=layouts,options=spec.get("options",{}))
    assert not native["trace_truncated"]
    if spec["node"]["kind"]=="argmax":
        # Only the actually executed PC path is retained; no timing data.
        selects=[e for e in native["trace"] if e["event"]=="instruction_issue" and e["phase"]=="select"]
        updated={(e["row"],e["column"]) for e in selects if e["pc"]==1}
        request["branch_taken"]=[(e["row"],e["column"]) not in updated for e in selects if e["pc"]==0]
        assert sum(request["branch_taken"])==native["branches_taken"]
    if spec["node"]["kind"]=="guard":
        request["guard_value"]=bool(next(e["data"]&255 for e in native["trace"] if e["event"]=="response" and not e["write"]))
    return request


def compare(binary,control_binary,path,spec,expected):
    spec=copy.deepcopy(spec);spec["external"]=False
    native,actual=native_run(control_binary,path/"native",spec)
    np.testing.assert_array_equal(actual,expected.numpy())
    request=request_from_execution(spec,native)
    (path/"control-event-job.json").write_text(json.dumps(request))
    program=control_events(request);result=run(binary,path/"event",program)
    assert result["block_intervals"][0]["window_cycles"]==native["cycles"]
    assert result["declared_work"].get("dma_read_bytes",0)==native["read_bytes"]
    assert result["declared_work"].get("dma_write_bytes",0)==native["write_bytes"]
    assert sum(result["declared_work"].get("control_"+op+"_events",0) for op in ("alu","multiply","float","branch"))==native["instructions"]
    assert result["integrated_usage"].get("control_instruction_inflight_cycles",0)-native["instructions"]==native["execution_stalls"]
    assert result["peak"]["contexts"]==result["peak"]["rf_vectors_per_pe"]==result["peak"]["spm_vectors"]==result["peak"]["rom_words_per_pe"]==0
    assert result["memory_controller_resources"]["peak_active"]==0
    assert result["control_controller_resources"]["register_bytes"]==512
    # Compare every issued operation's timestamp after independent execution.
    def leaves(items):
        for item in items:
            if "repeat" in item:yield from leaves(item["body"])
            else:yield item
    descriptors={e["id"]:e for e in leaves(program["controllers"][0]["events"])}
    event_sequence=[]
    for e in result["trace"]:
        if e["event"]!="issue":continue
        d=descriptors[e["operation"]]
        if d["op"]!="control_complete":event_sequence.append((e["cycle"],d["op"],d.get("bytes",0)))
    native_sequence=[]
    for e in native["trace"]:
        if e["event"]=="request":native_sequence.append((e["cycle"],"dma_write" if e["write"] else "dma_read",e["bytes"]))
        elif e["event"]=="literal":native_sequence.append((e["cycle"],"control_literal",0))
        elif e["event"]=="instruction_issue":
            w=e["word"];o=w&127
            op="branch" if o==0x63 else "float" if o==0x53 else "multiply" if o==0x33 and w>>25==1 else "alu"
            native_sequence.append((e["cycle"],"control_"+op,0))
    assert event_sequence==native_sequence
    return result,request


@pytest.mark.parametrize("slow",[False,True])
@pytest.mark.parametrize("kind",["arange","add","mul","le","ge","bitwise_and","all","guard","argmax"])
def test_complete_control_routes_match_actual_rv64_execution(binary,control_binary,tmp_path,kind,slow):
    a=torch.tensor([[2**60+1,2**60+3,-(2**63),2**63-1]]);b=torch.tensor([[3],[-1]])
    assets={"a":literal(a),"b":literal(b)}
    if kind=="arange":args,expected=[17],torch.arange(17)
    elif kind=="argmax":args,expected=[ref("a"),-1],a.argmax(-1)
    elif kind in {"bitwise_and","all","guard"}:
        a=torch.tensor([True,False,True]) if kind!="guard" else torch.tensor(True);b=~a
        assets={"a":literal(a),"b":literal(b)}
        if kind=="guard":args,expected=[ref("a"),True],a
        elif kind=="all":args,expected=[ref("a")],a.all()
        else:args,expected=[ref("a"),ref("b")],a&b
    else:
        args=[ref("a"),ref("b")]
        expected={"add":lambda:a+b,"mul":lambda:a*b,"le":lambda:a<=b,"ge":lambda:a>=b}[kind]()
    options=dict(dma_latency=3,alu_latency=2,multiply_latency=5,float_latency=4,branch_latency=3,request_period=4,response_period=7) if slow else {}
    compare(binary,control_binary,tmp_path,make_job(kind,args,expected,assets,options=options),expected)


@pytest.mark.parametrize("dtype",[torch.float16,torch.float32])
@pytest.mark.parametrize("kind",["argmax","le","ge"])
def test_float_control_nan_tie_and_literal_paths(binary,control_binary,tmp_path,dtype,kind):
    x=torch.tensor([[1,4,4,-1,3],[1,float("nan"),3,float("nan"),-2]],dtype=dtype)
    raw=x.T.contiguous();file=tmp_path/"raw.bin";file.write_bytes(raw.numpy().tobytes())
    name="f16" if dtype==torch.float16 else "f32"
    assets={"x":dict(kind="mapped_file",path=str(file),dtype=name,shape=list(raw.shape),bytes=raw.numel()*raw.element_size(),byte_offset=0)}
    expected=x.argmax(-1,keepdim=True) if kind=="argmax" else x<=0 if kind=="le" else x>=0
    args=[ref("x"),-1,True] if kind=="argmax" else [ref("x"),0.0]
    spec=make_job(kind,args,expected,assets,name,layouts={"x":dict(shape=[2,5],strides=[1,2])},options=dict(response_period=5,request_period=3))
    compare(binary,control_binary,tmp_path,spec,expected)


@pytest.mark.parametrize("kind",["arange","add","all","argmax"])
def test_empty_control_work_preserves_init_and_publication(binary,control_binary,tmp_path,kind):
    x=torch.empty((0,3) if kind=="argmax" else (0,),dtype=torch.bool if kind=="all" else torch.int64)
    if kind=="arange":args,expected=[0],x
    elif kind=="all":args,expected=[ref("x")],x.all()
    elif kind=="argmax":args,expected=[ref("x"),-1],x.argmax(-1)
    else:args,expected=[ref("x"),1],x
    compare(binary,control_binary,tmp_path,make_job(kind,args,expected,{"x":literal(x)}),expected)


def base_request():
    x=torch.tensor([1,3,2]);spec=make_job("argmax",[ref("x"),-1],x.argmax(-1),{"x":literal(x)})
    return dict(schema="mlx_control_event_window_job_v1",node=spec["node"],input_layouts={"x":dict(shape=[3],dtype="i64")},branch_taken=[False,True])


@pytest.mark.parametrize("damage",["missing_branch","branch_type","guard","code","dtype","shape","timing_hint"])
def test_invalid_control_lowering_rejected(damage):
    request=base_request()
    if damage=="missing_branch":request.pop("branch_taken")
    elif damage=="branch_type":request["branch_taken"][0]=1
    elif damage=="guard":request["guard_value"]=True
    elif damage=="code":request["node"]["control_program"]["phases"]["select"][0]=0
    elif damage=="dtype":request["input_layouts"]["x"]["dtype"]="bool"
    elif damage=="shape":request["node"]["output"]["shape"]=[1]
    else:request["native_cycles"]=100
    with pytest.raises(ValueError):control_events(request)


@pytest.mark.parametrize("witness",[None,False])
def test_guard_requires_actual_passing_value(witness):
    x=torch.tensor(True);spec=make_job("guard",[ref("x"),True],x,{"x":literal(x)})
    request=dict(schema="mlx_control_event_window_job_v1",node=spec["node"],input_layouts={"x":dict(shape=[],dtype="bool")})
    if witness is not None:request["guard_value"]=witness
    with pytest.raises(ValueError,match="witness|mismatch"):control_events(request)


def test_control_memory_and_pe_have_distinct_execution_resources(binary,tmp_path):
    p=control_events(base_request())
    p["controllers"][0]["events"]=[dict(id="control",op="control_float",dependencies=[])]
    p["controllers"].append(dict(id="memory",domain="memory",source_operator_id=1,admission_dependencies=[],events=[dict(id="convert",op="memory_convert",dependencies=[])]))
    p["templates"]=[dict(id="t",words=[1],rf_vectors=8,spm_vectors=5)]
    p["blocks"]=[dict(id="array",source_operator_id=2,pe=0,template="t",admission_dependencies=[],events=[event("compute")])]
    r=run(binary,tmp_path/"mixed",p)
    assert [e["cycle"] for e in r["trace"] if e["event"]=="issue"]==[0,0,1]
    assert r["peak"]["contexts"]==r["control_controller_resources"]["peak_active"]==r["memory_controller_resources"]["peak_active"]==1
    assert r["integrated_usage"]["control_instruction_inflight_cycles"]==3
    assert r["integrated_usage"]["memory_conversion_inflight_cycles"]==2
    assert r["counts"].get("spm_port_claims",0)==0


@pytest.mark.parametrize("field,value",[("register_bytes",1024),("rom_words",64),("max_active",2)])
def test_control_resource_expansion_rejected(binary,tmp_path,field,value):
    p=control_events(base_request());p["hardware"]["control_controller"][field]=value
    run(binary,tmp_path/"bad",p,"control resources")


def test_control_cannot_use_memory_private_conversion(binary,tmp_path):
    p=control_events(base_request());p["controllers"][0]["events"]=[dict(id="bad",op="memory_convert",dependencies=[])]
    run(binary,tmp_path/"bad",p,"different resource domain")


def test_control_and_memory_compete_for_one_physical_bus(binary,tmp_path):
    p=control_events(base_request())
    p["controllers"][0]["events"]=[dict(id="cr",op="dma_read",bytes=8,dependencies=[]),dict(id="alu",op="control_alu",dependencies=[]),dict(id="cw",op="dma_write",bytes=8,dependencies=[])]
    p["controllers"].append(dict(id="memory",domain="memory",source_operator_id=1,admission_dependencies=[],events=[
        dict(id="mr",op="dma_read",bytes=4,dependencies=[]),dict(id="convert",op="memory_convert",dependencies=[]),dict(id="mw",op="dma_write",bytes=4,dependencies=[])]))
    r=run(binary,tmp_path/"bus",p)
    issue={e["operation"]:e["cycle"] for e in r["trace"] if e["event"]=="issue"}
    assert issue==dict(cr=0,alu=9,mr=9,cw=18,convert=18,mw=27)
    assert r["cycles"]==36 and r["counts"].get("spm_port_claims",0)==0


def test_empty_control_holds_its_frontend_until_publication(binary,tmp_path):
    p=control_events(base_request());c=p["controllers"][0]
    c["events"]=[dict(id="empty",op="control_complete",dependencies=[])]
    p["controllers"].append(dict(id="second",domain="control",source_operator_id=1,admission_dependencies=[],events=[dict(id="alu",op="control_alu",dependencies=[])]))
    r=run(binary,tmp_path/"empty",p)
    assert r["block_intervals"][0]["window_cycles"]==0
    assert r["block_intervals"][1]["admit_cycle"]==1
    assert r["integrated_usage"]["control_controller_resident_cycles"]==r["cycles"]==3


def test_control_port_period_restarts_at_each_admission(binary,tmp_path):
    p=control_events(base_request());p["hardware"]["control_controller"].update(request_period=3,response_period=5)
    p["controllers"][0]["events"]=[dict(id="read0",op="dma_read",bytes=8,dependencies=[])]
    p["controllers"].append(dict(id="second",domain="control",source_operator_id=1,admission_dependencies=[],events=[dict(id="read1",op="dma_read",bytes=8,dependencies=[])]))
    r=run(binary,tmp_path/"period",p)
    assert [c["window_cycles"] for c in r["block_intervals"]]==[11,11]
    assert r["block_intervals"][1]["admit_cycle"]==11
