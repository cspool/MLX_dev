import copy
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from mlxsim.model_array_group_events import array_group_events,native_group_job
from scripts.verify_mlx_event_schedule import audit_trace
from test_control_events import base_request
from mlxsim.model_control_events import control_events
from test_event_schedule import binary,event
from test_event_loops import run
from test_event_ports import at
from test_matrix_window_scheduler import make_job,binary as matrix_binary,run as run_matrix
from test_vector_events import job as vector_job
from test_vector_window_scheduler import binary as vector_binary,run as run_vector

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def group_binary():
    build=ROOT/"build/event-native-group"
    for command in (["cmake","-S",str(ROOT/"simulator_ext/event_alignment"),"-B",str(build),"-DCMAKE_BUILD_TYPE=Release"],
                    ["cmake","--build",str(build),"--target","mlx-event-native-group","-j4"]):
        r=subprocess.run(command,capture_output=True,text=True,timeout=240);assert r.returncode==0,r.stdout+r.stderr
    return build/"mlx-event-native-group"


def two_sources():
    p=control_events(base_request());p["controllers"]=[];p["source_tick_order"]=True
    p["templates"]=[dict(id="t",words=[1],rf_vectors=4,spm_vectors=4)]
    p["blocks"]=[dict(id=str(i),source_operator_id=i,pe=0,template="t",admission_dependencies=[],events=[]) for i in range(2)]
    return p


def test_source_order_precedes_global_spm_completion_priority(binary,tmp_path):
    p=two_sources();p["hardware"]["latencies"].update(mul=3,spm_read=2)
    p["blocks"][0]["events"]=[event("compute")];p["blocks"][1]["events"]=[event("read","spm_read")]
    r=run(binary,tmp_path/"source",p);audit_trace(p,r)
    assert at(r,"compute","complete")==4 and at(r,"read","complete")==5
    p["source_tick_order"]=False;old=run(binary,tmp_path/"global",p)
    assert at(old,"read","complete")==4 and at(old,"compute","complete")==5


def test_lower_source_issue_can_delay_higher_source_dma_completion(binary,tmp_path):
    p=two_sources();p["hardware"]["latencies"].update(mul=2,dma_read=3)
    p["blocks"][0]["events"]=[event("compute"),event("store","spm_write")]
    p["blocks"][1]["events"]=[event("dma","dma_read")]
    r=run(binary,tmp_path/"source",p);audit_trace(p,r)
    assert at(r,"store","issue")==4 and at(r,"dma","complete")==5
    p["source_tick_order"]=False;old=run(binary,tmp_path/"global",p)
    assert at(old,"dma","complete")==4 and at(old,"store","issue")==5


@pytest.mark.parametrize("families",["mm","mv","vm","vv"])
@pytest.mark.parametrize("precision",["f16","f32"])
@pytest.mark.parametrize("slow",[False,True])
@pytest.mark.parametrize("columns",[1,2])
def test_full_numerical_windows_match_source_order(binary,group_binary,matrix_binary,vector_binary,tmp_path,families,precision,slow,columns):
    common=dict(dma_latency=3,spm_period=2,writeback_period=3,dma_request_period=2,dma_response_period=5) if slow else {}
    windows=[];expected=[]
    for index,family in enumerate(families):
        if family=="m":
            w,_=make_job(m=3,n=19,k=5,precision=precision,columns=columns,**common)
            _,values=run_matrix(matrix_binary,tmp_path/f"solo{index}",w)
        else:
            w=vector_job("silu" if index==0 else "softmax",width=17,rows=2,precision=precision,**common);w["options"]["columns"]=columns
            _,values=run_vector(vector_binary,tmp_path/f"solo{index}",w)
        windows.append(w);expected.append(values)
    spec=dict(schema="mlx_array_event_group_v1",windows=windows);(tmp_path/"group-job.json").write_text(json.dumps(spec))
    native=native_group_job(spec);(tmp_path/"native-job.json").write_text(json.dumps(native))
    proc=subprocess.run([str(group_binary),str(tmp_path/"native-job.json"),str(tmp_path/"native")],capture_output=True,text=True,timeout=120)
    (tmp_path/"native.log").write_text(proc.stdout+proc.stderr);assert proc.returncode==0,proc.stdout+proc.stderr
    report=json.loads((tmp_path/"native/result.json").read_text());p=array_group_events(spec);r=run(binary,tmp_path/"event",p);audit_trace(p,r)
    assert r["cycles"]==report["cycles"],(r["cycles"],report["cycles"])
    for ours,theirs in (("contexts","peak_contexts"),("rf_vectors_per_pe","peak_rf_vectors_per_pe"),("spm_vectors","peak_spm_vectors"),("rom_words_per_pe","peak_rom_words_per_pe")):
        assert r["peak"][ours]==report["array"][theirs]
    for index,values in enumerate(expected):
        raw=np.fromfile(tmp_path/f"native/output-{index}.bin",dtype=values.dtype).reshape(values.shape)
        np.testing.assert_array_equal(raw.view(np.uint8),values.view(np.uint8))
        retire=max(b["retire_cycle"] for b in r["block_intervals"] if b["source_operator_id"]==index)
        assert retire==report["windows"][index]["cycles"]
        trace=report["windows"][index]["trace"]
        for block in (b for b in r["block_intervals"] if b["source_operator_id"]==index):
            block_id=int(block["id"].split(":b")[1]);native_block=[e for e in trace if e["block_id"]==block_id]
            assert next(e["array_cycle"] for e in native_block if e["event"]=="admit")==block["admit_cycle"]
            assert next(e["array_cycle"]+1 for e in native_block if e["event"]=="retire")==block["retire_cycle"]
    def traffic(w,direction):return w["numeric_instructions"]["global_"+direction+"_bytes"] if "global_read_bytes" in w["numeric_instructions"] else w["dma_"+direction+"_bytes"]
    for direction in ("read","write"):assert r["declared_work"]["dma_"+direction+"_bytes"]==sum(traffic(w,direction) for w in report["windows"])
    p["source_tick_order"]=False;run(binary,tmp_path/"legacy",p)


@pytest.mark.parametrize("damage",["capacity","ports","latency","count"])
def test_incompatible_group_contract_rejected(damage):
    a,_=make_job();b=copy.deepcopy(a)
    if damage=="capacity":b["options"]["columns"]=2
    if damage=="ports":b["options"]["spm_period"]=2
    if damage=="latency":b["options"]["multiply_latency"]=8
    windows=[a] if damage=="count" else [a,b]
    with pytest.raises(ValueError):array_group_events(dict(schema="mlx_array_event_group_v1",windows=windows))


@pytest.mark.parametrize("damage",["policy","type"])
def test_invalid_source_tick_mode_rejected(binary,tmp_path,damage):
    p=two_sources()
    for i,b in enumerate(p["blocks"]):b["events"]=[event(f"e{i}")]
    if damage=="policy":p["policy"]="round_robin";error="requires source priority"
    else:p["source_tick_order"]="true";error="must be Boolean"
    run(binary,tmp_path/"bad",p,error)
