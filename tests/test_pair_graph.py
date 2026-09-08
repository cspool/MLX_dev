"""Whole-graph pair routing preserves source work and overlapping lifetimes."""
import copy
import json
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from mlxsim.model_control_program import control_program
from mlxsim.model_matrix_program import matrix_program
from mlxsim.model_vector_program import vector_program
from mlxsim.model_system_evidence import task_coverage,check_pair_kernel
from system_sim.physical_host.graph_lowering import compile_graph,literal_bytes
from system_sim.physical_host.lowering import collect_layouts
from scripts.mlx_system_attempt import launch_metadata
from scripts.run_mlx_clocked_chipyard import prepare
from test_model_tensor_semantics import literal,node,ref,native
from test_physical_model import compiled_nodes
from test_spike_graph_runtime import graph
from test_block_pipeline import matrix_pair
from test_clocked_device import binary,run

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def storage_binary():
    build=ROOT/"build/mlx-model-storage"
    for command in (["cmake","-S",str(ROOT/"simulator_ext/model_storage"),"-B",str(build),"-DCMAKE_BUILD_TYPE=Release"],
                    ["cmake","--build",str(build),"--target","model-storage-contract","-j4"]):
        p=subprocess.run(command,capture_output=True,text=True,timeout=180);assert p.returncode==0,p.stdout+p.stderr
    return build/"model-storage-contract"


def lifetime(binary,path,program):
    path.mkdir();(path/"program.json").write_text(json.dumps(program))
    p=subprocess.run([str(binary),str(path/"program.json"),str(path/"life.json")],capture_output=True,text=True,timeout=120)
    assert p.returncode==0,p.stdout+p.stderr
    return json.loads((path/"life.json").read_text())


def separated_pair():
    x=torch.tensor([[1.,2.,3.,4.]],dtype=torch.float16);w=torch.eye(4,dtype=torch.float16);a=-x;p=a@w.T;b=-x;c=p+b
    items=[node(0,"neg",[ref("x")],a),node(1,"linear",[ref("v0"),ref("w")],p),node(2,"neg",[ref("x")],b),
           node(3,"add",[ref("v1"),ref("v2")],c),node(4,"argmax",[ref("v3"),-1],c.argmax(-1))]
    for i in (0,2):items[i]["vector_program"]=vector_program("neg",["f16"],"f16")
    items[1]["matrix_program"]=matrix_program("f16","f16");items[3]["vector_program"]=vector_program("add",["f16","f16"],"f16")
    items[4]["control_program"]=control_program("argmax","f16")
    items[1]["release"]=["v0"];items[3]["release"]=["v1","v2"]
    return compiled_nodes({"x":literal(x),"w":literal(w)},items,"v3","v4"),c.numpy()


def batch_pair():
    a=torch.arange(30,dtype=torch.float32).reshape(2,1,3,5)/16;w=torch.arange(140,dtype=torch.float32).reshape(1,4,7,5)/32;b=w.transpose(-2,-1);p=a@b;c=-p
    items=[node(0,"transpose",[ref("w"),-2,-1],b),node(1,"matmul",[ref("a"),ref("v0")],p),node(2,"neg",[ref("v1")],c),node(3,"argmax",[ref("v2"),-1],c.argmax(-1))]
    items[1]["matrix_program"]=matrix_program("f32","f32");items[2]["vector_program"]=vector_program("neg",["f32"],"f32");items[3]["control_program"]=control_program("argmax","f32")
    return compiled_nodes({"a":literal(a),"w":literal(w)},items,"v2","v3"),c.numpy()


def no_live_overlap(plan):
    intervals=plan["storage_intervals"]
    for i,a in enumerate(intervals):
        for b in intervals[i+1:]:
            if a["last"]<b["first"] or b["last"]<a["first"]:continue
            assert a["base"]+a["reserved_bytes"]<=b["base"] or b["base"]+b["reserved_bytes"]<=a["base"]


def test_nonadjacent_pair_replans_real_serial_reuse(storage_binary,tmp_path):
    program,_=separated_pair();life=lifetime(storage_binary,tmp_path/"input",program);before=copy.deepcopy(program)
    assert life["events"][0]["allocation"]["base"]==life["events"][2]["allocation"]["base"]
    blob,plan=compile_graph(program,life,block_pairs=True,event_slots=2)
    assert program==before and plan["execution_groups"]==[[0],[2],[1,3],[4]]
    assert plan["source_calls"]==5 and plan["task_count"]==4 and plan["pair_count"]==1
    no_live_overlap(plan);task_coverage(program,plan)
    addresses={i["root"]:i["base"] for i in plan["storage_intervals"]}
    assert len({addresses[k] for k in ("v0","v1","v2","v3")})==4
    task=next(t for t in plan["tasks"] if t["kind"]==3)
    assert struct.unpack_from('<8Q',blob,task["command_offset"])[4:]==(1,3,2,1)
    launch=launch_metadata(program,plan)[-1]
    assert launch["source_operator_id"]==1 and launch["consumer"]["source_operator_id"]==3


@pytest.mark.parametrize("batched",[False,True])
def test_replanned_commands_execute_real_values_after_intervening_source(binary,storage_binary,tmp_path,batched):
    program,expected=batch_pair() if batched else separated_pair();life=lifetime(storage_binary,tmp_path/"input",program)
    blob,plan=compile_graph(program,life,block_pairs=True,event_slots=2)
    job={"base":plan["device_base"],"bytes":plan["device_bytes"],"initial":[{"address":plan["assets"][name]["base"],"data":list(literal_bytes(asset))} for name,asset in program["assets"].items()],
         "matrix_options":{"rows":1,"columns":1},"vector_options":{"rows":1,"columns":1},"memory_options":{},
         "commands":[{"address":plan["device_base"]+4096,"data":list(blob[t["command_offset"]:t["command_offset"]+t["bytes"]])} for t in plan["tasks"] if t["kind"] in (2,3)],
         "outputs":[{"address":plan["outputs"][0]["binding"]["base"],"bytes":expected.nbytes}]}
    result=run(binary,tmp_path/"device",job);assert result["outputs"]==[list(expected.tobytes())]
    kernel=result["launches"][-1]["kernel"];pair=next(t for t in plan["tasks"] if t["kind"]==3)
    profile={"matrix_options":{"rows":1,"columns":1,"contexts":2},"vector_options":{"rows":1,"columns":1,"contexts":2}}
    values={**program["assets"],**{n["id"]:n["output"] for n in program["nodes"]}}
    work=check_pair_kernel(program,pair,kernel,profile,values,2,len(result["launches"]))
    assert work["matrix_mac_lanes"]==(840 if batched else 16) and work["write_bytes"]==2*expected.nbytes
    for damage in ("batch","macs","source","events","capacity","template","mux"):
        bad=copy.deepcopy(kernel)
        if damage=="batch":bad["producer_windows"].pop()
        elif damage=="macs":bad["producer_windows"][0]["numeric_instructions"]["mul_active_lanes"]-=1
        elif damage=="source":bad["consumer_source"]+=1
        elif damage=="events":bad["block_events"]["epoch"]+=1
        elif damage=="capacity":bad["array"]["peak_rf_vectors_per_pe"]=17
        elif damage=="template":bad["array"]["template_words_loaded"]-=1
        else:bad["physical_mux"]["channels"][0]["consumed"]-=1
        with pytest.raises(RuntimeError):check_pair_kernel(program,pair,bad,profile,values,2,len(result["launches"]))


@pytest.mark.parametrize("damage",["groups","deps","consumer","missing","capacity","address","bytes","version"])
def test_pair_evidence_rejects_partial_or_changed_routes(storage_binary,tmp_path,damage):
    program,_,_=matrix_pair();life=lifetime(storage_binary,tmp_path/"input",program);_,plan=compile_graph(program,life,block_pairs=True)
    if damage=="groups":plan["execution_groups"][0]=[0]
    elif damage=="deps":plan["source_dependencies"][1]=[]
    elif damage=="consumer":plan["tasks"][0]["consumer_source_id"]=99
    elif damage=="missing":plan["tasks"].pop()
    elif damage=="capacity":plan["required_mapped_bytes"]-=64
    elif damage=="address":plan["storage_intervals"][-1]["base"]+=64
    elif damage=="bytes":plan["tasks"][0]["bytes"]=1088
    else:plan["host_abi_version"]=3
    with pytest.raises((RuntimeError,ValueError)):task_coverage(program,plan)


def test_compiler_preserves_legacy_mode_and_rejects_capacity(storage_binary,tmp_path):
    program,_,_=matrix_pair();life=lifetime(storage_binary,tmp_path/"input",program)
    assert compile_graph(program,life)==compile_graph(program,life,block_pairs=False)
    with pytest.raises(ValueError,match="mapped limit"):compile_graph(program,life,block_pairs=True,device_bytes=65536)
    with pytest.raises(ValueError,match="event slots"):compile_graph(program,life,block_pairs=True,event_slots=0)


def test_complete_generation_image_has_dual_source_checks(graph,tmp_path):
    original,program,life=graph;_,plan=compile_graph(program,life,block_pairs=True)
    no_live_overlap(plan);task_coverage(program,plan)
    assert plan["source_calls"]==45 and plan["pair_count"]>0
    out=tmp_path/"image";image,_=prepare({"program":str(original/"program.json"),"lifetimes":str(original/"life.json"),"reference":str(original/"out/result.json")},out,block_pairs=True,event_slots=2)
    text=(out/"test.c").read_text()
    assert '.version=2' in text and 'source_table' in text and 'completion[i]!=source_ids[i]' in text
    assert image["source_calls"]==45 and not image["model_data_executed"]
