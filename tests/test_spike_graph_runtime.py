import copy
import json
import struct
import subprocess
from pathlib import Path

import pytest
import torch

from system_sim.physical_host.graph_lowering import compile_graph,literal_bytes
from test_model_tensor_semantics import native
from test_model_memory_program import test_all_lowering_families_execute_one_generation_and_cache_graph as prepare_generation

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def graph(native,tmp_path_factory):
    path=tmp_path_factory.mktemp("dispatch-graph");prepare_generation(native,path,True)
    program=json.loads((path/"program.json").read_text())
    binary=ROOT/"build/mlx-model-storage/model-storage-contract"
    result=subprocess.run([str(binary),str(path/"program.json"),str(path/"life.json")],capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stdout+result.stderr
    return path,program,json.loads((path/"life.json").read_text())


def test_generic_task_table_covers_every_source_and_elides_views(graph):
    _,program,life=graph;blob,plan=compile_graph(program,life)
    assert plan["source_calls"]==len(program["nodes"])==45 and plan["task_count"]==45
    assert plan["family_source_calls"]["control"]==15 and plan["family_source_calls"]["view"]==9
    assert sum(t["kind"]==2 for t in plan["tasks"])==21
    assert all(t["bytes"]==0 and t["command_offset"]==0 for t in plan["tasks"] if t["kind"]==0)
    assert all(t["command_offset"]+t["bytes"]<=len(blob) for t in plan["tasks"])
    assert all(binding["base"]>=2**32+65536 for binding in plan["assets"].values())


@pytest.mark.parametrize("damage",["capacity","ambiguous","lifetime","source"])
def test_generic_graph_compiler_rejects_inconsistent_inputs(graph,damage):
    _,program,life=graph;program,life=copy.deepcopy((program,life));options={}
    if damage=="capacity":options["device_bytes"]=65536
    elif damage=="ambiguous":program["nodes"][0]["control_program"]={}
    elif damage=="lifetime":life["events"][0]["allocation"]["bytes"]+=1
    else:program["nodes"][0]["args"][0]["value"]="not-live"
    with pytest.raises((ValueError,KeyError)):compile_graph(program,life,**options)


def test_literal_packer_preserves_declared_cpp_initialization_rounding():
    assert literal_bytes({"dtype":"i64","shape":[2],"values":[2**60+1,-2**63]})==struct.pack("<qq",2**60+1,-2**63)
    assert literal_bytes({"dtype":"bool","shape":[1],"values":[1e-50]})==b"\0"
    value=1.00048828125+2**-30
    single=struct.unpack("<f",struct.pack("<f",value))[0]
    assert literal_bytes({"dtype":"f16","shape":[1],"values":[value]})==struct.pack("<e",single)
    with pytest.raises(ValueError):literal_bytes({"dtype":"i64","shape":[1],"values":[1.5]})


@pytest.mark.parametrize("perturb",[False,True])
def test_generated_rv64_dispatch_runs_complete_generation_graph(graph,tmp_path,perturb):
    original,program,_=graph;program=copy.deepcopy(program)
    expected=json.loads((original/"out/result.json").read_text())
    if perturb:
        key=next(n for n in program["nodes"] if n["kind"]=="embedding")["args"][0]["value"]
        program["assets"][key]["values"][1]=[30.0,0.0,0.0,0.0]
        weight=torch.tensor(program["assets"][key]["values"],dtype=torch.float16);current=torch.tensor([[1]]);cache=torch.empty(1,0,4,dtype=torch.float16);outputs=[]
        for step in range(3):
            cache=torch.cat([cache,torch.nn.functional.embedding(current,weight)],dim=1)
            logits=cache.transpose(1,2).mean(-1)*2+1;logits=torch.where(torch.arange(4)<=step+1,logits,-65504.0)
            token=logits.argmax(-1);current=token.unsqueeze(1);file=tmp_path/f"reference-{step}.bin";file.write_bytes(logits.numpy().tobytes())
            outputs.append({"forward_id":step,"dtype":"f16","shape":list(logits.shape),"tokens":token.tolist(),"logits_file":str(file)})
        expected={"outputs":outputs}
        assert [o["tokens"] for o in outputs]==[[0],[0],[0]]
    file=tmp_path/"program.json";file.write_text(json.dumps(program));ref=tmp_path/"reference.json";ref.write_text(json.dumps(expected));out=tmp_path/"run"
    result=subprocess.run([str(ROOT/".venv/bin/python"),"-m","scripts.run_mlx_spike_graph","--program",str(file),"--lifetimes",str(original/"life.json"),"--reference",str(ref),"--output",str(out)],cwd=ROOT,capture_output=True,text=True,timeout=300)
    assert result.returncode==0,result.stdout+result.stderr
    report=json.loads((out/"report.json").read_text())
    assert report["actual_cpu_dispatch"] and report["output_bytes_checked_in_elf"] and report["source_calls"]==45 and report["device_windows"]==21
    assert not report["full_model_execution_verified"] and not report["mlx_system_verified"]
