import copy
import json
import struct
import subprocess
from pathlib import Path

import pytest

from mlxsim.model_memory_program import make_layout
from mlxsim.model_vector_program import vector_program
from mlxsim.model_dtype_lowering import lower_softmax_input_cast
from system_sim.physical_device.vector_lowering import lower_vector,PHASES

ROOT=Path(__file__).resolve().parents[1]
BUILD=ROOT/"build/mlx-spike-matrix"


@pytest.fixture(scope="module")
def decoder():
    for command in (["cmake","-S",str(ROOT/"system_sim/physical_device"),"-B",str(BUILD),"-DCMAKE_BUILD_TYPE=Release"],["cmake","--build",str(BUILD),"--target","vector-wire-dump","-j4"]):
        result=subprocess.run(command,capture_output=True,text=True,timeout=180);assert result.returncode==0,result.stdout+result.stderr
    return BUILD/"vector-wire-dump"


def case(kind,precision="f16",output=None,width=65):
    output=output or precision;shape=[2,width] if kind in {"mean","softmax"} else [2,17]
    args=[{"value":"a"}];types=[precision];kw={}
    if kind in {"add","mul"}:args.append(0.125);types.append("f32")
    if kind=="add":kw["alpha"]=1.75
    if kind=="pow":args.append(2)
    if kind=="mean":args.extend([[-1],True])
    if kind=="softmax":args.extend([-1,"torch.float16" if output=="f16" else "torch.float32"])
    out_shape=[2,1] if kind=="mean" else shape
    layouts={"a":make_layout(precision,shape,"a"),"out":make_layout(output,out_shape,"out")}
    bindings={"a":{"base":2**32,"bytes":layouts["a"]["storage_elements"]*(2 if precision=="f16" else 4),"writable":False},"out":{"base":2**32+65536,"bytes":layouts["out"]["storage_elements"]*(2 if output=="f16" else 4),"writable":True}}
    node={"id":"out","kind":kind,"args":args,"kwargs":kw,"output":{"dtype":output,"shape":out_shape},"source_operator_id":3,"vector_program":vector_program(kind,types,output,width=width if kind in {"mean","softmax"} else None,alpha=kw.get("alpha",1))}
    lower_softmax_input_cast(node)
    return node,layouts,bindings


def decode(decoder,tmp_path,blob,error=False):
    file=tmp_path/"wire.bin";file.write_bytes(blob)
    result=subprocess.run([str(decoder),str(file),str(tmp_path/"decoded.json")],capture_output=True,text=True,timeout=30)
    if error:
        assert result.returncode!=0 and "VECTOR_WIRE_DECODE_FAIL" in result.stderr
        return
    assert result.returncode==0,result.stdout+result.stderr
    report=json.loads((tmp_path/"decoded.json").read_text());assert not report["mlx_system_verified"]
    return report["windows"][0]


@pytest.mark.parametrize("kind",["add","mul","pow","rsqrt","silu","cos","sin","neg","mean","softmax"])
@pytest.mark.parametrize("precision",["f16","f32"])
def test_all_vector_wire_modes_preserve_programs_and_bindings(decoder,tmp_path,kind,precision):
    node,layouts,bindings=case(kind,precision);blob,route=lower_vector(node,layouts,bindings);assert len(blob)==4288 and route["rom_words"]<=32
    decoded=decode(decoder,tmp_path,blob)
    assert decoded["node"]["vector_program"]==node["vector_program"]
    assert decoded["backend_constructor_validated"] and decoded["values"]["a"]["shape"]==layouts["a"]["shape"]
    assert decoded["regions"][0]["base"]==bindings["a"]["base"]
    if kind in {"add","mul"}:assert decoded["node"]["args"][1]==0.125


@pytest.mark.parametrize("width",[1,3,17,65,4096])
def test_narrowing_softmax_preserves_pre_reduction_conversions(decoder,tmp_path,width):
    node,layouts,bindings=case("softmax","f32","f16",width);blob,_=lower_vector(node,layouts,bindings)
    decoded=decode(decoder,tmp_path,blob)
    assert decoded["node"]["vector_program"]==node["vector_program"]
    assert decoded["node"]["vector_program"]["softmax_input_cast"]=="f32_to_f16_before_reduction"


@pytest.mark.parametrize("damage",["rom","phase","width","shape","readonly","overlap","scalar"])
def test_vector_lowering_does_not_drop_source_semantics(damage):
    node,layouts,bindings=copy.deepcopy(case("add"))
    if damage=="rom":node["vector_program"]["rom"][0]^=256
    elif damage=="phase":node["vector_program"]["phases"]["body"].pop()
    elif damage=="width":node["vector_program"]["width"]=19
    elif damage=="shape":node["output"]["shape"]=[17]
    elif damage=="readonly":bindings["out"]["writable"]=False
    elif damage=="overlap":bindings["out"]["base"]=bindings["a"]["base"]
    else:node["vector_program"]["input_dtypes"][1]="f16"
    with pytest.raises(ValueError):lower_vector(node,layouts,bindings)


@pytest.mark.parametrize("index,value",[(0,0),(2,99),(8,1),(5,33),(90,2**32),(121,1),(137,1),(138,33),(150,99),(181,1),(534,1),(16,99),(91,0x0c)])
def test_cpp_vector_wire_rejects_corrupted_binary(decoder,tmp_path,index,value):
    blob,_=lower_vector(*case("add"));words=list(struct.unpack("<536Q",blob));words[index]=value
    decode(decoder,tmp_path,struct.pack("<536Q",*words),True)
