import pytest
import torch

from mlxsim.model_event_graph_plan import graph_plan
from mlxsim.model_matrix_program import matrix_program
from mlxsim.model_control_program import control_program
from test_model_tensor_semantics import literal,node,ref
from test_physical_model import compiled_nodes


def program():
    a=torch.ones(2,1,3,4);b=torch.ones(1,5,4,6);y=a@b;condition=torch.tensor(True)
    guard=node(0,"guard",[ref("condition"),True],condition);guard["control_program"]=control_program("guard")
    mat=node(1,"matmul",[ref("a"),ref("b")],y);mat["matrix_program"]=matrix_program("f32","f32");mat["control_dependencies"]=[ref("v0")]
    token=node(2,"argmax",[ref("v1"),-1],y.argmax(-1));token["control_program"]=control_program("argmax","f32")
    return compiled_nodes({"a":literal(a),"b":literal(b),"condition":literal(condition)},[guard,mat,token],"v1","v2")


def test_broadcast_batches_control_edges_and_witnesses_are_preserved():
    p=graph_plan(program());m=p["sources"][1]
    assert p["source_calls"]==p["lowered_calls"]==3 and p["total_windows"]==12
    assert m["parents"]==[0] and m["control_inputs"]==["v0"] and m["window_count"]==10
    assert m["matrix"]["a_batch_shape"]==[2,1] and m["matrix"]["b_batch_shape"]==[1,5] and m["matrix"]["output_batch_shape"]==[2,5]
    assert [x["count"] for x in p["timing_witness_requirements"]]==[1,150]
    assert not p["full_event_lowering_complete"] and p["event_simulated_cycles"] is None


@pytest.mark.parametrize("damage",["value","reference","route","batch","k","source","output"])
def test_incomplete_full_graph_contract_is_rejected(damage):
    p=program()
    if damage=="value":p["nodes"][1]["id"]="a"
    elif damage=="reference":p["nodes"][1]["control_dependencies"]=[ref("missing")]
    elif damage=="route":p["nodes"][1].pop("matrix_program")
    elif damage=="batch":p["assets"]["b"]["shape"][0]=3
    elif damage=="k":p["assets"]["a"]["shape"][-1]=3
    elif damage=="source":p["nodes"][1]["source_operator_id"]=0
    else:p["outputs"][0]["token"]="missing"
    with pytest.raises(ValueError):graph_plan(p)
