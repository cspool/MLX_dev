"""One source, multiple independent view values and real C++ storage ownership."""
import copy
import json
import subprocess

import numpy as np
import pytest
import torch

from mlxsim.model_execution_inventory import ModelExecutionInventory
from mlxsim.model_tensor_compiler import compile_inventory
from mlxsim.model_block_pipeline import compile_block_pipelines
from mlxsim.model_ready_evidence import verify_ready_execution
from mlxsim.model_physical_evidence import verify_physical_execution
from mlxsim.model_value_outputs import require_value_contract
from system_sim.physical_host.graph_lowering import compile_graph
from test_model_tensor_semantics import literal, node, ref, native
from test_memory_window_scheduler import binary as memory_binary, job, run as run_memory
from test_physical_model import binary as physical_binary, run as run_physical
from test_ready_graph import binary as graph_binary, execute as run_graph


def split_node(identifier, source, tensor, size, axis):
    parts = tensor.split(size, axis)
    item = node(identifier, "split", [ref(source), size, axis], tensor)
    item["split_outputs"] = [{"id":f"v{identifier}:{i}","output":literal(part)|{}} for i,part in enumerate(parts)]
    for output in item["split_outputs"]:
        output["output"] = {k:output["output"][k] for k in ("dtype","shape")}
    return item, parts


@pytest.mark.parametrize("dtype",[torch.float16,torch.float32,torch.int64,torch.bool])
def test_all_split_parts_are_independent_views_and_reordered_correctly(memory_binary,tmp_path,dtype):
    x=torch.arange(14).reshape(2,7)
    if dtype==torch.int64:x=x+2**60
    if dtype==torch.bool:x=x.remainder(3).eq(0)
    x=x.to(dtype);item,parts=split_node(0,"x",x,3,1)
    expected=torch.cat([parts[2],parts[0],parts[1]],1)
    items=[item,node(1,"cat",[[ref("v0:2"),ref("v0:0"),ref("v0:1")],1],expected)]
    spec=job({"x":literal(x)},items,buffered_bridge=True)
    spec["nodes"][0]["release"]=["x","v0"]
    report,actual=run_memory(memory_binary,tmp_path/"parts",spec)
    np.testing.assert_array_equal(actual,expected.numpy())
    first=report["windows"][0]
    assert first["view_elided"] and first["dma_requests"]==first["cycles"]==0
    assert first["numeric_instructions"]["profile"]=="mlx-memory-plan-v3"
    assert first["numeric_instructions"]["split_view_outputs"]==3


def test_noncontiguous_split_children_survive_owner_release(memory_binary,tmp_path):
    x=torch.arange(24).reshape(3,8);t=x.T
    split,parts=split_node(1,"v0",t,2,0)
    expected=torch.cat(list(reversed(parts)),0)
    items=[node(0,"transpose",[ref("x"),0,1],t),split,
           node(2,"cat",[[ref(f"v1:{i}") for i in range(len(parts)-1,-1,-1)],0],expected)]
    spec=job({"x":literal(x)},items)
    spec["nodes"][0]["release"]=["x"];spec["nodes"][1]["release"]=["v0","v1"]
    report,actual=run_memory(memory_binary,tmp_path/"strided",spec)
    np.testing.assert_array_equal(actual,expected.numpy())
    assert report["windows"][1]["numeric_instructions"]["split_view_outputs"]==4


@pytest.mark.parametrize("shape,size,axis",[([2,0],0,1),([0,3],4,0),([1,5],9,-1)])
def test_empty_or_single_split_result_is_still_a_returned_value(memory_binary,tmp_path,shape,size,axis):
    x=torch.empty(shape) if not np.prod(shape) else torch.arange(int(np.prod(shape))).reshape(shape).float()
    split,parts=split_node(0,"x",x,size,axis)
    spec=job({"x":literal(x)},[split,node(1,"cast",[ref("v0:0"),"torch.float32",False,True],parts[0])])
    spec["nodes"][0]["release"]=["x","v0"]
    _,actual=run_memory(memory_binary,tmp_path/"empty",spec)
    np.testing.assert_array_equal(actual,parts[0].numpy())


@pytest.mark.parametrize("damage",["missing","offset","dtype","root","identity","profile"])
def test_split_canonical_layout_and_complete_outputs_are_checked(memory_binary,tmp_path,damage):
    x=torch.arange(12).reshape(1,6,2).float();split,parts=split_node(0,"x",x,1,-1)
    spec=job({"x":literal(x)},[split,node(1,"contiguous",[ref("v0:0")],parts[0].contiguous())])
    n=spec["nodes"][0];p=n["memory_program"]
    if damage=="missing":n["split_outputs"].pop();error="omits or adds"
    if damage=="offset":p["split_layouts"]["v0:1"]["offset"]=0;error="layout/ownership"
    if damage=="dtype":n["split_outputs"][1]["output"]["dtype"]="i64";error="shape/dtype"
    if damage=="root":p["split_layouts"]["v0:1"]["root"]="foreign";error="layout/ownership"
    if damage=="identity":n["split_outputs"][1]["id"]="v0:0";error="not canonical"
    if damage=="profile":p["profile"]="mlx-memory-plan-v2";error="mismatched memory"
    run_memory(memory_binary,tmp_path/"bad",spec,error)


def capture(tmp_path,length=28,unused=False):
    (tmp_path/"model.safetensors.index.json").write_text(json.dumps({"weight_map":{}}))
    x=torch.arange(length*(3 if unused else 2)).reshape(1,length,3 if unused else 2).float()
    trace=ModelExecutionInventory(torch.nn.Identity(),capture_bindings=True)
    trace.bind_input("x",x)
    with torch.inference_mode(),trace.step(0,"prefill"):
        computed=x*2+1
        parts=computed.split(1,-1)
        left=parts[0].squeeze(-1).contiguous()
        right=parts[-1].squeeze(-1).contiguous()
        logits=left*3+right
        trace.phase="token_selection";token=logits.argmax(-1)
    report=trace.report();report.update(model_identity={"path":str(tmp_path),"family":"tuple-component","variant":"not-model-validation","parameters":0,"files":{}},
                                        reference_checks=[{"forward_id":0}],input={"length":length,"unused":unused})
    program,coverage=compile_inventory(report,matrix_backend="scheduled",vector_backend="scheduled",memory_backend="scheduled",control_backend="scheduled")
    return program,report,coverage,logits.numpy(),token.tolist()


@pytest.mark.parametrize("length,unused",[(28,False),(64,False),(5,True)])
def test_complete_source_compilation_and_tuple_lifetimes_in_both_native_graphs(physical_binary,graph_binary,tmp_path,length,unused):
    program,inventory,coverage,expected,tokens=capture(tmp_path,length,unused)
    split=next(n for n in program["nodes"] if n["kind"]=="split")
    assert program["schema"]=="mlx_tensor_semantics_v2" and coverage["source_calls"]==len(inventory["operations"])==len(program["nodes"])
    assert split["id"] in split["release"]
    if unused:assert split["split_outputs"][1]["id"] in split["release"]
    for runner,binary,name in ((run_physical,physical_binary,"serial"),(run_graph,graph_binary,"dag")):
        actual=runner(binary,tmp_path/name,program)
        np.testing.assert_array_equal(np.fromfile(actual["outputs"][0]["logits_file"],dtype=np.float32).reshape(expected.shape),expected)
        assert actual["outputs"][0]["tokens"]==tokens and actual["arena_drained"]["reserved_bytes"]==0
        event=next(e for e in actual["events"] if e["source_operator_id"]==split["source_operator_id"])
        assert event["produced_values"]==[o["id"] for o in split["split_outputs"]]
        if name=="serial":
            bad=copy.deepcopy(actual);next(e for e in bad["events"] if e["source_operator_id"]==split["source_operator_id"])["produced_values"].pop()
            with pytest.raises(RuntimeError,match="every output"):verify_physical_execution(program,bad)
    paired=compile_block_pipelines(program,event_slots=4)
    options={"base":2**32,"bytes":65536,"max_cycles":10000000,"tile_pipeline":True,"template_load_timing":True}
    result=run_graph(graph_binary,tmp_path/"paired",paired,options)
    verify_ready_execution(paired,result,options)
    bad=copy.deepcopy(result);next(e for e in bad["events"] if e["source_operator_id"]==split["source_operator_id"])["produced_values"].pop()
    with pytest.raises(RuntimeError,match="every output"):verify_ready_execution(paired,bad,options)


@pytest.mark.parametrize("damage",["schema","contract","owner_operand","owner_export","duplicate"])
def test_tuple_program_cannot_be_relabelled_as_old_or_use_internal_owner(graph_binary,tmp_path,damage):
    program,_,_,_,_=capture(tmp_path)
    split=next(n for n in program["nodes"] if n["kind"]=="split")
    if damage=="schema":program["schema"]="mlx_tensor_semantics_v1"
    if damage=="contract":del program["value_contract"]
    if damage=="owner_operand":program["nodes"][-1]["args"][0]=ref(split["id"])
    if damage=="owner_export":program["outputs"][0]["logits"]=split["id"]
    if damage=="duplicate":split["split_outputs"][1]["id"]=split["split_outputs"][0]["id"]
    with pytest.raises(ValueError):require_value_contract(program)
    run_graph(graph_binary,tmp_path/"invalid",program,error="split" if damage.startswith("owner") or damage=="duplicate" else "tuple-view")


def test_plain_native_consumes_all_split_values_but_observer_requires_selection(native,tmp_path):
    program,_,_,expected,tokens=capture(tmp_path)
    p=tmp_path/"program.json";p.write_text(json.dumps(program))
    result=subprocess.run([str(native[0]),str(p),str(tmp_path/"plain"),"none","1"],capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
    report=json.loads((tmp_path/"plain/result.json").read_text())
    np.testing.assert_array_equal(np.fromfile(report["outputs"][0]["logits_file"],dtype=np.float32).reshape(expected.shape),expected)
    assert report["outputs"][0]["tokens"]==tokens
    ids=tmp_path/"observe.json";ids.write_text(json.dumps([next(n["source_operator_id"] for n in program["nodes"] if n["kind"]=="split")]))
    result=subprocess.run([str(native[0]),str(p),str(tmp_path/"observe"),"none","1",str(ids)],capture_output=True,text=True,timeout=60)
    assert result.returncode==1 and "tuple-view observations" in result.stderr
    assert not (tmp_path/"observe/result.json").exists()


def test_system_abi_rejects_tuple_elision_before_building_tasks(tmp_path):
    program,_,_,_,_=capture(tmp_path)
    with pytest.raises(ValueError,match="has not registered tuple-view"):
        compile_graph(program,{})
