import copy

import numpy as np
import pytest
import torch

from mlxsim.model_event_windows import WindowCatalog,MissingWitness,broadcast_batch,element_address
from mlxsim.model_vector_program import vector_program
from mlxsim.model_control_program import control_program
from scripts.run_mlx_event_witness import witness_plan,decode_boolean_witness
from test_event_graph_plan import program
from test_physical_model import compiled_nodes
from test_model_tensor_semantics import literal,node,ref


@pytest.mark.parametrize("batch",range(10))
def test_full_matrix_broadcast_window_keeps_original_layout_and_offsets(batch):
    catalog=WindowCatalog(program());job,binding=catalog.window(1,batch)
    assert job["m"]==3 and job["n"]==6 and job["k"]==4 and job["a"]["shape"]==[3,4]
    assert binding["inputs"]["a"]["shape"]==[2,1,3,4]
    assert binding["matrix"]["a_batch_index"]==batch//5 and binding["matrix"]["b_batch_index"]==batch%5
    assert binding["matrix"]["output_flat_start"]==batch*18
    assert element_address(binding["inputs"]["a"],binding["matrix"]["a_flat_start"])["byte_offset"]==(batch//5)*12*4
    assert element_address(binding["outputs"]["v1"],binding["matrix"]["output_flat_start"])["byte_offset"]==batch*18*4
    event=catalog.compile(job);assert sum(b["source_operator_id"]==0 for b in event["blocks"])==2


def test_strided_alias_address_uses_root_offset_and_strides():
    layout=dict(root="root",dtype="f32",shape=[3,2],strides=[1,5],offset=6,storage_elements=20)
    view=np.arange(20).reshape(4,5)[1:3,1:4].T
    assert [element_address(layout,i)["byte_offset"]//4 for i in range(6)]==view.reshape(-1).tolist()


@pytest.mark.parametrize("index",[-1,6,True])
def test_invalid_logical_address_rejected(index):
    with pytest.raises(ValueError):element_address(dict(root="r",dtype="f32",shape=[2,3],strides=[3,1],offset=0,storage_elements=6),index)


def test_identical_vector_timing_patterns_keep_distinct_value_bindings():
    a=torch.ones(2,3);b=a*2
    nodes=[node(i,"neg",[ref(name)],-value) for i,(name,value) in enumerate([("a",a),("b",b)])]
    for n in nodes:n["vector_program"]=vector_program("neg",["f32"],"f32")
    p=compiled_nodes({"a":literal(a),"b":literal(b),"token":literal(torch.tensor(0))},nodes,"v1","token")
    c=WindowCatalog(p);first,fa=c.window(0);second,sa=c.window(1)
    assert first==second and fa["inputs"]!=sa["inputs"]
    assert fa["pattern_inputs"]=={"input0":"a"} and sa["pattern_inputs"]=={"input0":"b"}
    assert c.compile(first)==c.compile(second)


@pytest.mark.parametrize("source,witness",[(0,None),(2,None)])
def test_missing_dynamic_witness_is_not_filled_from_expected_output(source,witness):
    with pytest.raises(MissingWitness):WindowCatalog(program()).window(source,witness=witness)


def test_guard_value_and_batch_range_are_checked():
    c=WindowCatalog(program());job,_=c.window(0,witness=False)
    with pytest.raises(ValueError,match="guard mismatch"):c.compile(job)
    with pytest.raises(ValueError,match="batch outside"):c.window(1,10)


def boolean_program():
    mask=torch.tensor([True,False,True]);x=torch.arange(3).float()
    nodes=[node(0,"bitwise_and",[ref("mask"),ref("mask")],mask),node(1,"all",[ref("v0")],mask.all()),
           node(2,"guard",[ref("v1"),False],mask.all()),node(3,"where",[ref("v0"),ref("x"),1.0],torch.where(mask,x,1.0))]
    for n in nodes[:3]:n["control_program"]=control_program(n["kind"])
    return compiled_nodes({"mask":literal(mask),"x":literal(x),"token":literal(torch.tensor(0))},nodes,"v3","token")


def test_boolean_witness_plan_binds_actual_predecessor_sources():
    p=boolean_program();r=copy.deepcopy(p)
    for family,backend in zip(("matrix","vector","memory","control"),("microcode","microcode","planned","rv64_leaf")):r[family+"_backend"]=backend
    rows=witness_plan(p,r)
    assert [(x["source_operator_id"],x["producer"],x["count"]) for x in rows]==[(2,1,1),(3,0,3)]
    assert decode_boolean_witness(rows[0],b"\x00") is False
    assert decode_boolean_witness(rows[1],b"\x01\x00\x01")==[True,False,True]
    c=WindowCatalog(p)
    for source,value in [(2,False),(3,[True,False,True])]:
        job,binding=c.window(source,witness=value);assert c.compile(job)["controllers"] and not binding["addresses_executed"]
    r["assets"]["mask"]["values"][0]=False
    with pytest.raises(RuntimeError,match="differ beyond"):witness_plan(p,r)


@pytest.mark.parametrize("raw",[b"\x02",b"",b"\x00\x01"])
def test_corrupt_boolean_observation_is_rejected(raw):
    row=dict(input_shape=[],kind="actual_guard_boolean",expected_guard=False)
    with pytest.raises(RuntimeError):decode_boolean_witness(row,raw)


def test_where_scalar_observation_expands_only_by_real_broadcast():
    row=dict(input_shape=[1,2],output_shape=[3,2],kind="actual_where_predicate",count=6)
    assert decode_boolean_witness(row,b"\x00\x01")==[False,True]*3


def test_changed_memory_layout_is_rejected_before_window_lowering():
    p=boolean_program();p["nodes"][3]["memory_program"]["output_layout"]["strides"]=[2]
    with pytest.raises(ValueError,match="canonical memory"):WindowCatalog(p)


@pytest.mark.parametrize("args",[(-1,[2,5],[2,1]),(10,[2,5],[2,1]),(0,[2,5],[3,1])])
def test_invalid_broadcast_batch_is_rejected(args):
    with pytest.raises(ValueError):broadcast_batch(*args)
