import copy
import json
import struct
import subprocess
from pathlib import Path

import pytest
import torch

from mlxsim.model_memory_program import Planner
from system_sim.physical_host.lowering import BYTES
from system_sim.physical_device.memory_lowering import lower_memory
from test_model_tensor_semantics import literal,node,ref

ROOT=Path(__file__).resolve().parents[1]
BUILD=ROOT/"build/mlx-spike-matrix"


@pytest.fixture(scope="module")
def decoder():
    for cmd in (["cmake","-S",str(ROOT/"system_sim/physical_device"),"-B",str(BUILD),"-DCMAKE_BUILD_TYPE=Release"],["cmake","--build",str(BUILD),"--target","memory-wire-dump","-j4"]):
        result=subprocess.run(cmd,capture_output=True,text=True,timeout=180);assert result.returncode==0,result.stdout+result.stderr
    return BUILD/"memory-wire-dump"


def prepare(assets,nodes,same_device=True):
    planner=Planner()
    for name,tensor in assets.items():planner.add_asset(name,literal(tensor))
    for n in nodes:planner.register(n,planned=True,same_device=same_device)
    bindings={};base=2**32
    for layout in planner.layouts.values():
        root=layout["root"]
        if root in bindings:continue
        size=layout["storage_elements"]*BYTES[layout["dtype"]];bindings[root]={"base":base,"bytes":size,"writable":root not in assets};base+=max(64,(size+63)//64*64)
    return nodes[-1],planner.layouts,bindings


def decode(decoder,path,blob,error=False):
    path.mkdir();(path/"wire.bin").write_bytes(blob)
    result=subprocess.run([str(decoder),str(path/"wire.bin"),str(path/"decoded.json")],capture_output=True,text=True,timeout=30)
    if error:
        assert result.returncode!=0 and "MEMORY_WIRE_DECODE_FAIL" in result.stderr
        return
    assert result.returncode==0,result.stdout+result.stderr
    return json.loads((path/"decoded.json").read_text())["windows"][0]


@pytest.mark.parametrize("kind",["alias","dropout_inference","transpose","slice","select","unsqueeze","expand","reshape","contiguous","cast","cast_device"])
def test_view_and_transfer_distinctions_survive_memory_wire(decoder,tmp_path,kind):
    x=torch.arange(12,dtype=torch.float32).reshape(3,4);assets={"x":x}
    if kind=="alias":args,expected=[ref("x")],x
    elif kind=="dropout_inference":args,expected=[ref("x"),0.3,False],x
    elif kind=="transpose":args,expected=[ref("x"),0,1],x.T
    elif kind=="slice":args,expected=[ref("x"),1,None,None,2],x[:,::2]
    elif kind=="select":args,expected=[ref("x"),0,-1],x[-1]
    elif kind=="unsqueeze":args,expected=[ref("x"),0],x.unsqueeze(0)
    elif kind=="expand":args,expected=[ref("x"),[2,3,-1]],x.expand(2,3,-1)
    elif kind=="reshape":args,expected=[ref("x"),[-1]],x.reshape(-1)
    elif kind=="contiguous":args,expected=[ref("x")],x.contiguous()
    elif kind=="cast":args,expected=[ref("x"),"torch.float16",False,True],x.half()
    else:args,expected=[ref("x"),"cuda:0","torch.float32",False,False],x
    item,layouts,bindings=prepare(assets,[node(0,kind,args,expected)],same_device=kind!="cast_device")
    blob,route=lower_memory(item,layouts,bindings);result=decode(decoder,tmp_path/kind,blob)
    assert len(blob)==15872 and result["view_elided"]==(item["memory_program"]["mode"]=="view")
    assert result["node"]["memory_program"]["words"]==item["memory_program"]["words"] and result["backend_constructor_validated"]
    assert route["operand_slots"]==1


def test_nondense_reshape_materialization_preserves_source_layout(decoder,tmp_path):
    x=torch.arange(24).reshape(3,8);t=x.T
    item,layouts,bindings=prepare({"x":x},[node(0,"transpose",[ref("x"),0,1],t),node(1,"reshape",[ref("v0"),[-1]],t.reshape(-1))])
    blob,_=lower_memory(item,layouts,bindings);result=decode(decoder,tmp_path/"copy",blob)
    assert not result["view_elided"] and result["node"]["memory_program"]["input_layouts"]["a00"]["strides"]==[1,8]


@pytest.mark.parametrize("dtype",[torch.float16,torch.float32,torch.int64,torch.bool])
def test_where_and_literal_types_are_not_lost(decoder,tmp_path,dtype):
    p=torch.tensor([[True],[False]]);x=torch.tensor([1,2,3],dtype=dtype);scalar=2**60+1 if dtype==torch.int64 else False if dtype==torch.bool else -65504.0
    expected=torch.where(p,x,scalar);item,layouts,bindings=prepare({"p":p,"x":x},[node(0,"where",[ref("p"),ref("x"),scalar],expected)])
    blob,_=lower_memory(item,layouts,bindings);result=decode(decoder,tmp_path/"where",blob)
    assert result["node"]["args"][2]==scalar and result["node"]["memory_program"]["selector"]=="predicate_select"


@pytest.mark.parametrize("width",[0,7])
def test_embedding_indices_remain_tensor_operands(decoder,tmp_path,width):
    w=torch.empty(5,width);ids=torch.tensor([4,1]);expected=torch.empty(2,width)
    item,layouts,bindings=prepare({"w":w,"ids":ids},[node(0,"embedding",[ref("w"),ref("ids")],expected)])
    blob,_=lower_memory(item,layouts,bindings);result=decode(decoder,tmp_path/"embedding",blob)
    assert result["node"]["memory_program"]["words"][0]==4 and result["node"]["memory_program"]["input_layouts"]["a01"]["dtype"]=="i64"


def test_concat_repeated_ssa_is_not_duplicated_in_region_table(decoder,tmp_path):
    x=torch.arange(6).reshape(2,3);expected=torch.cat([x]*64,dim=1)
    item,layouts,bindings=prepare({"x":x},[node(0,"cat",[[ref("x")]*64,1],expected)])
    blob,route=lower_memory(item,layouts,bindings);result=decode(decoder,tmp_path/"cat",blob)
    assert route["operand_slots"]==1 and route["argument_count"]==64 and len(result["regions"])==2
    assert len(result["node"]["args"][0])==64 and all(arg=={"value":"a00"} for arg in result["node"]["args"][0])


@pytest.mark.parametrize("index,value",[(0,0),(3,7),(11,1),(6,65),(7,257),(8,5),(10,17),(309,1),(310,99),(1974,3),(1983,1),(16,1)])
def test_cpp_memory_wire_rejects_corruption(decoder,tmp_path,index,value):
    x=torch.arange(4).float();case=prepare({"x":x},[node(0,"cast",[ref("x"),"torch.float16"],x.half())]);blob,_=lower_memory(*case);words=list(struct.unpack("<1984Q",blob));words[index]=value
    decode(decoder,tmp_path/"corrupt",struct.pack("<1984Q",*words),True)


def test_memory_wire_refuses_wrong_source_dtype_and_changed_plan():
    x=torch.arange(4).float();item,layouts,bindings=prepare({"x":x},[node(0,"cast",[ref("x"),"torch.float16"],x.half())])
    item["args"][1]="torch.float32"
    with pytest.raises(ValueError,match="dtype argument"):lower_memory(item,layouts,bindings)
    item["args"][1]="torch.float16";item["memory_program"]["words"]=[3]
    with pytest.raises(ValueError,match="modified"):lower_memory(item,layouts,bindings)
