import copy
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from mlxsim.model_array_graph_events import array_graph_events,native_array_graph_job
from scripts.verify_mlx_event_schedule import audit_trace
from test_event_source_order import group_binary,two_sources
from test_event_schedule import binary,event
from test_event_loops import run
from test_matrix_window_scheduler import make_job,binary as matrix_binary,run as run_matrix
from test_vector_events import job as vector_job
from test_vector_window_scheduler import binary as vector_binary,run as run_vector
from test_model_tensor_semantics import literal
from mlxsim.model_control_events import control_events
from test_control_events import base_request


def graph_job(precision="f16",slow=False,limit=2):
    options=dict(columns=2)
    if slow:options.update(dma_latency=3,spm_period=2,writeback_period=3,dma_request_period=2,dma_response_period=5)
    a,_=make_job(m=2,n=16,k=3,precision=precision,**options);b=copy.deepcopy(a)
    b["a"]["values"]=(np.asarray(b["a"]["values"])*2).tolist()
    def vec(kind):
        p=vector_job(kind,width=16,rows=2,precision=precision,**{k:v for k,v in options.items() if k!="columns"});p["options"]["columns"]=2;return p
    def output(index):return dict(kind="group_output",window=index,dtype=precision,shape=[2,16])
    join=vec("add");join["assets"]={"x":output(0),"y":output(1)}
    independent=vec("neg")
    multiply=vec("mul");multiply["assets"]={"x":output(2),"y":output(3)}
    reduce=vec("mean");reduce["assets"]={"x":output(4)}
    fork=vec("silu");fork["assets"]={"x":output(0)}
    windows=[a,b,join,independent,multiply,reduce,fork]
    sources=[dict(source_operator_id=i,parents=parents,windows=indices) for i,(parents,indices) in enumerate([
        ([],[0,1]),([0],[2]),([],[3]),([1,2],[4]),([3],[5]),([0],[6])])]
    return dict(schema="mlx_array_event_graph_v1",windows=windows,sources=sources,max_active_sources=limit)


@pytest.mark.parametrize("precision",["f16","f32"])
@pytest.mark.parametrize("slow",[False,True])
@pytest.mark.parametrize("limit",[2,4])
def test_numeric_fork_join_batches_and_frontend_slots(binary,group_binary,matrix_binary,vector_binary,tmp_path,precision,slow,limit):
    spec=graph_job(precision,slow,limit);expected=[]
    for index,window in enumerate(spec["windows"]):
        solo=copy.deepcopy(window)
        if "assets" in solo:
            for name,asset in solo["assets"].items():
                if asset["kind"]=="group_output":solo["assets"][name]=literal(torch.from_numpy(expected[asset["window"]]))
        runner,executable=(run_matrix,matrix_binary) if "program" in solo else (run_vector,vector_binary)
        _,values=runner(executable,tmp_path/f"solo{index}",solo);expected.append(values)
    (tmp_path/"graph-job.json").write_text(json.dumps(spec));native=native_array_graph_job(spec);(tmp_path/"native-job.json").write_text(json.dumps(native))
    process=subprocess.run([str(group_binary),str(tmp_path/"native-job.json"),str(tmp_path/"native")],capture_output=True,text=True,timeout=120)
    (tmp_path/"native.log").write_text(process.stdout+process.stderr);assert process.returncode==0,process.stdout+process.stderr
    n=json.loads((tmp_path/"native/result.json").read_text());p=array_graph_events(spec);r=run(binary,tmp_path/"event",p);audit_trace(p,r)
    assert r["source_graph_completed"] and r["cycles"]==n["cycles"],(r["cycles"],n["cycles"])
    assert r["source_frontend_peak"]==n["source_frontend_peak"]<=limit
    assert [{k:v for k,v in row.items() if k!="family"} for row in r["source_intervals"]]==n["source_intervals"]
    for index,values in enumerate(expected):
        actual=np.fromfile(tmp_path/f"native/output-{index}.bin",dtype=values.dtype).reshape(values.shape)
        np.testing.assert_array_equal(actual.view(np.uint8),values.view(np.uint8))


def marker_graph():
    p=two_sources();p["blocks"][0]["events"]=[event("e0")];p["blocks"][1]["events"]=[event("e1")]
    p["source_graph"]=[dict(source_operator_id=i,family="matrix",parents=[0] if i else [],windows=[[str(i)]]) for i in range(2)]
    return p


def test_dependency_waiting_source_does_not_occupy_frontend(binary,tmp_path):
    p=marker_graph();r=run(binary,tmp_path/"chain",p)
    assert r["source_frontend_peak"]==1
    assert r["source_intervals"][1]["begin_cycle"]==r["source_intervals"][0]["publish_cycle"]==6


def test_waiting_for_pe_still_holds_a_frontend_slot(binary,tmp_path):
    p=marker_graph();p["hardware"]["source_window_limit"]=2;p["hardware"]["contexts"]=1
    p["source_graph"][1]["parents"]=[]
    third=copy.deepcopy(p["blocks"][1]);third.update(id="2",source_operator_id=2,events=[event("e2")]);p["blocks"].append(third)
    p["source_graph"].append(dict(source_operator_id=2,family="matrix",parents=[],windows=[["2"]]))
    r=run(binary,tmp_path/"slots",p)
    assert r["source_frontend_peak"]==2 and r["peak"]["contexts"]==1
    assert r["source_intervals"][1]["begin_cycle"]==0 and r["block_intervals"][1]["admit_cycle"]==6
    assert r["source_intervals"][2]["begin_cycle"]==6


@pytest.mark.parametrize("damage",["missing_source","duplicate_block","missing_block","window_order","parent_cycle","domain","mode"])
def test_malformed_cpp_source_graph_rejected(binary,tmp_path,damage):
    p=marker_graph();g=p["source_graph"];error="source graph"
    if damage=="missing_source":g.pop()
    elif damage=="duplicate_block":g[0]["windows"][0].append("0")
    elif damage=="missing_block":g[0]["windows"][0]=["missing"]
    elif damage=="window_order":g[0]["windows"]=[]
    elif damage=="parent_cycle":g[0]["parents"]=[1]
    elif damage=="domain":g[0]["family"]="control"
    else:p["source_tick_order"]=False
    run(binary,tmp_path/"bad",p,error)


@pytest.mark.parametrize("damage",["missing_dependency","missing_window","duplicate_source","shape","forward"])
def test_graph_compiler_rejects_missing_real_data_dependencies(damage):
    p=graph_job()
    if damage=="missing_dependency":p["sources"][1]["parents"]=[]
    elif damage=="missing_window":p["sources"][-1]["windows"]=[]
    elif damage=="duplicate_source":p["sources"][1]["source_operator_id"]=0
    elif damage=="shape":p["windows"][2]["assets"]["x"]["dtype"]="f32"
    else:p["windows"][2]["assets"]["x"]["window"]=6
    with pytest.raises(ValueError):array_graph_events(p)


def test_new_source_waits_for_shared_dma_quiescence(binary,tmp_path):
    p=control_events(base_request());p["source_tick_order"]=True
    p["controllers"][0]["events"]=[dict(id="control",op="control_branch",dependencies=[])]
    p["controllers"].append(dict(id="memory",domain="memory",source_operator_id=1,admission_dependencies=[],events=[dict(id="dma",op="dma_read",bytes=4,dependencies=[])]))
    p["templates"]=[dict(id="t",words=[1],rf_vectors=4,spm_vectors=4)]
    p["blocks"]=[dict(id="array",source_operator_id=2,pe=0,template="t",admission_dependencies=[],events=[event("mul")])]
    p["source_graph"]=[dict(source_operator_id=i,family=f,parents=parents,windows=[[name]]) for i,f,parents,name in [(0,"control",[],"control"),(1,"memory",[],"memory"),(2,"matrix",[0],"array")]]
    r=run(binary,tmp_path/"quiescent",p);audit_trace(p,r)
    assert r["source_intervals"][0]["publish_cycle"]==2
    assert r["source_intervals"][2]["begin_cycle"]==9


def test_control_frontend_remains_exclusive_while_other_sources_run(binary,tmp_path):
    p=control_events(base_request());p["source_tick_order"]=True;p["hardware"]["latencies"]["control_alu"]=10
    p["controllers"][0]["events"]=[dict(id="control0",op="control_alu",dependencies=[])]
    p["controllers"]+=[dict(id="memory",domain="memory",source_operator_id=1,admission_dependencies=[],events=[dict(id="convert",op="memory_convert",dependencies=[])]),
                       dict(id="control1",domain="control",source_operator_id=2,admission_dependencies=[],events=[dict(id="branch",op="control_branch",dependencies=[])])]
    p["source_graph"]=[dict(source_operator_id=i,family=f,parents=parents,windows=[[name]]) for i,f,parents,name in [(0,"control",[],"control"),(1,"memory",[],"memory"),(2,"control",[1],"control1")]]
    r=run(binary,tmp_path/"exclusive",p);audit_trace(p,r)
    assert r["source_intervals"][1]["publish_cycle"]==3
    assert r["source_intervals"][2]["begin_cycle"]==11 and r["cycles"]==13
