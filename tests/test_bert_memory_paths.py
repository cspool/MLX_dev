"""Actual C++ indexing, generation and layout paths needed by BERT QA."""
import copy
import json

import numpy as np
import pytest
import torch

from mlxsim.model_execution_inventory import ModelExecutionInventory
from mlxsim.model_tensor_compiler import compile_inventory, validate_event
from mlxsim.model_memory_program import Planner
from test_memory_window_scheduler import binary as memory_binary, job, run as run_memory
from test_model_tensor_semantics import literal, node, ref
from test_ready_graph import binary as graph_binary, execute as run_graph
from test_physical_model import binary as physical_binary, run as run_physical


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.int64, torch.bool])
@pytest.mark.parametrize("buffered", [False, True])
def test_nd_index_reads_actual_broadcast_negative_coordinates(memory_binary, tmp_path, dtype, buffered):
    data = torch.arange(15).reshape(3,5)
    data = data + 2**60 if dtype == torch.int64 else data.remainder(2) if dtype == torch.bool else data
    data = data.to(dtype)
    rows, cols = torch.tensor([[2],[-3]]), torch.tensor([[4,0,-1,2]])
    expected = data[rows,cols]
    spec = job({"data":literal(data), "rows":literal(torch.zeros_like(rows)), "cols":literal(torch.zeros_like(cols))},
               [node(0,"advanced_index",[ref("data"),[ref("rows"),ref("cols")]],expected)],
               physical_values={"rows":literal(rows),"cols":literal(cols)}, buffered_bridge=buffered,
               options={"response_period":7,"request_period":2})
    report, actual = run_memory(memory_binary,tmp_path/"index",spec)
    np.testing.assert_array_equal(actual,expected.numpy())
    count = report["windows"][0]["numeric_instructions"]
    assert count["profile"] == "mlx-memory-plan-v2" and count["index_reads"] == expected.numel()*2
    assert count["read_bytes"] == expected.numel()*(16+data.element_size())
    assert count["write_bytes"] == expected.numel()*data.element_size()
    assert count["instructions"] == expected.numel()*5


def test_nd_index_uses_strides_after_root_value_release(memory_binary, tmp_path):
    x=torch.arange(24).reshape(2,3,4); trans=x.permute(2,1,0)
    indices=[torch.tensor([3,-4]),torch.tensor([1,2]),torch.tensor([0,-1])]
    nodes=[node(0,"transpose",[ref("x"),0,2],trans),
           node(1,"advanced_index",[ref("v0"),[ref("a"),ref("b"),ref("c")]],trans[tuple(indices)])]
    spec=job({"x":literal(x),**{k:literal(v) for k,v in zip("abc",indices)}},nodes)
    spec["nodes"][0]["release"]=["x"]
    report,actual=run_memory(memory_binary,tmp_path/"strides",spec)
    np.testing.assert_array_equal(actual,trans[tuple(indices)].numpy())
    assert report["windows"][0]["view_elided"] and report["windows"][1]["numeric_instructions"]["index_reads"]==6


@pytest.mark.parametrize("bad", [-4,3,2**60])
def test_bad_nd_index_responses_cannot_be_used_as_addresses(memory_binary,tmp_path,bad):
    data=torch.arange(15).reshape(3,5)
    spec=job({"x":literal(data),"a":literal(torch.tensor([0])),"b":literal(torch.tensor([0]))},
             [node(0,"advanced_index",[ref("x"),[ref("a"),ref("b")]],torch.tensor([0]))],
             physical_values={"a":literal(torch.tensor([bad]))})
    run_memory(memory_binary,tmp_path/"bad-index",spec,"advanced index out of range")


def test_empty_and_scalar_index_broadcasts(memory_binary,tmp_path):
    x=torch.arange(15).reshape(3,5)
    for name,a,b in [("empty",torch.empty(0,1,dtype=torch.int64),torch.tensor([[1,2,3]])),
                     ("scalar",torch.tensor(-1),torch.tensor(2))]:
        expected=x[a,b]
        spec=job({"x":literal(x),"a":literal(a),"b":literal(b)},
                 [node(0,"advanced_index",[ref("x"),[ref("a"),ref("b")]],expected)])
        report,actual=run_memory(memory_binary,tmp_path/name,spec)
        np.testing.assert_array_equal(actual,expected.numpy())
        assert report["windows"][0]["dma_requests"]==expected.numel()*4


@pytest.mark.parametrize("rank",[1,8,9])
def test_nd_index_rank_capacity_is_enforced(memory_binary,tmp_path,rank):
    source=torch.tensor(47).reshape([1]*rank)
    assets={"x":literal(source),**{f"i{d}":literal(torch.tensor(-1)) for d in range(rank)}}
    item=node(0,"advanced_index",[ref("x"),[ref(f"i{d}") for d in range(rank)]],torch.tensor(47))
    if rank==9:
        with pytest.raises(ValueError,match="rank 1..8"):job(assets,[item])
        return
    report,actual=run_memory(memory_binary,tmp_path/"rank",job(assets,[item]))
    assert int(actual)==47 and report["windows"][0]["numeric_instructions"]["index_reads"]==rank


@pytest.mark.parametrize("dtype", [torch.float16,torch.float32,torch.int64,torch.bool])
@pytest.mark.parametrize("shape", [[],[0],[1,35]])
def test_new_ones_generates_fresh_data_without_reading_template(memory_binary,tmp_path,dtype,shape):
    source=torch.tensor([0,-3,8],dtype=torch.int64); expected=torch.ones(shape,dtype=dtype)
    spec=job({"x":literal(source)},[node(0,"new_ones",[ref("x"),shape],expected,dtype=str(dtype))],
             buffered_bridge=True,options={"response_period":5})
    report,actual=run_memory(memory_binary,tmp_path/"ones",spec)
    np.testing.assert_array_equal(actual,expected.numpy())
    count=report["windows"][0]["numeric_instructions"]
    assert count["read_bytes"]==0 and count["instructions"]==3*expected.numel()
    assert count["write_bytes"]==expected.numel()*expected.element_size() and count["allocations"]==1


@pytest.mark.parametrize("damage",["dtype","shape","pin","words","view"])
def test_new_ones_contract_is_checked_before_generation(memory_binary,tmp_path,damage):
    spec=job({"x":literal(torch.tensor([9]))},[node(0,"new_ones",[ref("x"),[2]],torch.ones(2,dtype=torch.bool),dtype="torch.bool")])
    item=spec["nodes"][0];p=item["memory_program"]
    if damage=="dtype":item["kwargs"]["dtype"]="torch.float32";error="dtype contract"
    if damage=="shape":item["args"][1]=[3];error="shape mismatch"
    if damage=="pin":item["kwargs"]["pin_memory"]=True;error="pinned memory"
    if damage=="words":p["words"].pop(0);error="template mismatch"
    if damage=="view":p["mode"]="view";p["words"]=[];error="affine view"
    run_memory(memory_binary,tmp_path/"bad-ones",spec,error)


@pytest.mark.parametrize("shape,axis", [([1,3,1],-1),([2,1,3],1),([1,3],1),([],0),([],-1),([0,1],1)])
def test_squeeze_proves_view_and_preserves_lifetime(memory_binary,tmp_path,shape,axis):
    x=torch.arange(max(1,int(np.prod(shape)))).reshape(shape) if np.prod(shape) else torch.empty(shape,dtype=torch.int64)
    expected=x.squeeze(axis)
    spec=job({"x":literal(x)},[node(0,"squeeze",[ref("x"),axis],expected),
                              node(1,"cast",[ref("v0"),"torch.int64",False,True],expected)])
    spec["nodes"][0]["release"]=["x"]
    report,actual=run_memory(memory_binary,tmp_path/"squeeze",spec)
    np.testing.assert_array_equal(actual,expected.numpy())
    assert report["windows"][0]["view_elided"] and report["windows"][0]["dma_requests"]==0
    assert not report["windows"][1]["view_elided"]


@pytest.mark.parametrize("damage", ["missing_index_word","wrong_profile","dtype","partial","boolean_index","shape"])
def test_index_plan_rejects_unregistered_or_corrupted_contract(memory_binary,tmp_path,damage):
    x=torch.arange(6).reshape(2,3);a=torch.tensor([0]);b=torch.tensor([1])
    assets={"x":literal(x),"a":literal(a),"b":literal(b)}
    item=node(0,"advanced_index",[ref("x"),[ref("a"),ref("b")]],torch.tensor([1]))
    if damage in {"partial","boolean_index","shape"}:
        if damage=="partial":item["args"][1].pop()
        if damage=="boolean_index":assets["a"]=literal(torch.tensor([True]))
        if damage=="shape":item["output"]["shape"]=[2]
        with pytest.raises(ValueError,match="index"):
            job(assets,[item])
        return
    spec=job(assets,[item]);p=spec["nodes"][0]["memory_program"]
    if damage=="missing_index_word":p["words"].pop(0);error="template mismatch"
    if damage=="wrong_profile":p["profile"]="mlx-memory-plan-v1";error="mismatched memory"
    if damage=="dtype":spec["assets"]["a"]["dtype"]="f32";error="runtime layout"
    run_memory(memory_binary,tmp_path/"bad-plan",spec,error)


def capture(tmp_path,copy_flag):
    (tmp_path/"model.safetensors.index.json").write_text(json.dumps({"weight_map":{}}))
    mask=torch.arange(64).remainder(3).eq(0).reshape(1,64)
    rows=torch.zeros(1,1,1,1,dtype=torch.int64);cols=torch.arange(63,-1,-1).reshape(1,1,1,64)
    trace=ModelExecutionInventory(torch.nn.Identity(),capture_bindings=True)
    for name,value in (("mask",mask),("rows",rows),("cols",cols)):trace.bind_input(name,value)
    with torch.inference_mode(), trace.step(0,"prefill"):
        selected=torch.ops.aten.index.Tensor(mask,[rows,cols])
        ones=selected.new_ones([],dtype=torch.bool,pin_memory=False)
        converted=torch.ops.aten.to.dtype_layout(selected,dtype=torch.bool,layout=torch.strided,device=torch.device("cpu"),copy=copy_flag)
        squeezed=converted.squeeze(1).squeeze(1)
        logits=(squeezed & ones).to(torch.float32)
        trace.phase="token_selection";token=logits.argmax(-1)
    report=trace.report();report.update(model_identity={"path":str(tmp_path),"family":"bert-memory-component","variant":"not-full-model","parameters":0,"files":{}},reference_checks=[{"forward_id":0}],input={"copy":copy_flag})
    return report,logits.numpy(),token.tolist()


@pytest.mark.parametrize("copy_flag", [False,True])
def test_actual_operator_capture_routes_to_serial_and_ready_cpp(memory_binary,physical_binary,graph_binary,tmp_path,copy_flag):
    inventory,expected,tokens=capture(tmp_path,copy_flag)
    program,coverage=compile_inventory(inventory,matrix_backend="scheduled",vector_backend="scheduled",memory_backend="scheduled",control_backend="scheduled")
    assert len(program["nodes"])==len(inventory["operations"])==coverage["source_calls"]
    layout=next(n for n in program["nodes"] if n["source_operator"]=="aten.to.dtype_layout")
    assert layout["memory_program"]["mode"]==("transfer" if copy_flag else "view")
    for runner,binary,name in ((run_physical,physical_binary,"serial"),(run_graph,graph_binary,"dag")):
        actual=runner(binary,tmp_path/name,program)
        np.testing.assert_array_equal(np.fromfile(actual["outputs"][0]["logits_file"],dtype=np.float32).reshape(expected.shape),expected)
        assert actual["outputs"][0]["tokens"]==tokens and actual["functional_entry_calls"]==actual["blas_calls"]==0


@pytest.mark.parametrize("kwargs", [{"pin_memory":True},{"pin_memory":0},{"layout":"torch.sparse_coo"},{"copy":1},{"dtype":"torch.float16"},{"unknown":True}])
def test_dtype_layout_overload_does_not_drop_unchecked_attributes(tmp_path,kwargs):
    inventory,_,_=capture(tmp_path,False)
    item=copy.deepcopy(next(e for e in inventory["operations"] if e["operator"]=="aten.to.dtype_layout"))
    item["kwargs"].update(kwargs)
    with pytest.raises(ValueError):validate_event(item)
