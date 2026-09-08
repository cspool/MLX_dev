import copy
import json
import struct
import subprocess
from pathlib import Path

import pytest

from mlxsim.model_matrix_program import matrix_program
from mlxsim.model_memory_program import make_layout
from system_sim.physical_device.lowering import geometry,lower_matrix_window

ROOT=Path(__file__).resolve().parents[1]
BUILD=ROOT/"build/mlx-spike-matrix"


@pytest.fixture(scope="module")
def decoder():
    for command in (["cmake","-S",str(ROOT/"system_sim/physical_device"),"-B",str(BUILD),"-DCMAKE_BUILD_TYPE=Release"],["cmake","--build",str(BUILD),"--target","matrix-wire-dump","-j4"]):
        result=subprocess.run(command,capture_output=True,text=True,timeout=180);assert result.returncode==0,result.stdout+result.stderr
    return BUILD/"matrix-wire-dump"


def linear(precision="f16",bias=False):
    layouts={"a":make_layout(precision,[1,8,5],"a"),"b":make_layout(precision,[7,5],"b"),"out":make_layout(precision,[1,8,7],"out"),"bias":make_layout(precision,[7],"bias")}
    width=2 if precision=="f16" else 4
    bindings={name:{"base":2**32+index*0x1000,"bytes":layout["storage_elements"]*width,"writable":name=="out"} for index,(name,layout) in enumerate(layouts.items())}
    node={"id":"out","kind":"linear","args":[{"value":"a"},{"value":"b"}]+([{"value":"bias"}] if bias else []),"output":{"dtype":precision,"shape":[1,8,7]},"source_operator_id":5,"matrix_program":matrix_program(precision,precision,bias)}
    return node,layouts,bindings


def decode(decoder,tmp_path,blob,success=True):
    file=tmp_path/"wire.bin";file.write_bytes(blob)
    result=subprocess.run([str(decoder),str(file),str(tmp_path/"decoded.json")],capture_output=True,text=True,timeout=30)
    if not success:
        assert result.returncode!=0 and "MATRIX_WIRE_DECODE_FAIL" in result.stderr
        return
    assert result.returncode==0,result.stdout+result.stderr
    report=json.loads((tmp_path/"decoded.json").read_text())
    assert not report["mlx_system_verified"]
    return report["windows"][0]


@pytest.mark.parametrize("precision",["f16","f32"])
@pytest.mark.parametrize("bias",[False,True])
def test_linear_wire_roundtrip_preserves_microcode_and_bindings(decoder,tmp_path,precision,bias):
    node,layouts,bindings=linear(precision,bias);blob,route=lower_matrix_window(node,layouts,bindings)
    assert len(blob)==1088 and route["m"]==8 and route["n"]==7 and route["k"]==5
    result=decode(decoder,tmp_path,blob)
    assert result["program"]=={k:v for k,v in node["matrix_program"].items() if k not in {"numeric_contract","target_status"}}
    assert result["a"]["base"]==2**32 and result["output"]["base"]==2**32+0x2000
    assert result["has_bias"]==bias and result["backend_constructor_validated"]


def test_batch_broadcast_and_transpose_stride_are_encoded_per_window(decoder,tmp_path):
    a=make_layout("f32",[2,1,3,5],"a");b=make_layout("f32",[1,4,5,7],"b");b["strides"]=[140,35,1,5]
    out=make_layout("f32",[2,4,3,7],"out");layouts={"a":a,"b":b,"out":out}
    bindings={name:{"base":2**32+i*4096,"bytes":layout["storage_elements"]*4,"writable":name=="out"} for i,(name,layout) in enumerate(layouts.items())}
    node={"id":"out","kind":"matmul","args":[{"value":"a"},{"value":"b"}],"output":{"dtype":"f32","shape":out["shape"]},"source_operator_id":0,"matrix_program":matrix_program("f32","f32")}
    assert geometry(node,layouts)["batches"]==8
    for batch in range(8):
        path=tmp_path/str(batch);path.mkdir();blob,route=lower_matrix_window(node,layouts,bindings,batch);result=decode(decoder,path,blob)
        assert route["a_batch"]==result["a_batch"]==batch//4 and route["b_batch"]==result["b_batch"]==batch%4
        assert result["output_batch"]==batch and result["b"]["strides"]==[140,35,1,5] and not result["transpose_b"]


@pytest.mark.parametrize("damage",["microcode","output-shape","bias","readonly","overlap","stride","batch","capacity"])
def test_matrix_lowering_rejects_unsupported_changes(damage):
    node,layouts,bindings=copy.deepcopy(linear());batch=0
    if damage=="microcode":node["matrix_program"]["body"][0]^=256
    elif damage=="output-shape":node["output"]["shape"]=[1,8,8]
    elif damage=="bias":node["args"].append(7)
    elif damage=="readonly":bindings["out"]["writable"]=False
    elif damage=="overlap":bindings["out"]["base"]=bindings["a"]["base"]
    elif damage=="stride":layouts["a"]["strides"]=[40,5,-1]
    elif damage=="batch":batch=1
    else:bindings["a"]["bytes"]-=2
    with pytest.raises(ValueError):lower_matrix_window(node,layouts,bindings,batch)


@pytest.mark.parametrize("index,value",[(0,0),(2,8),(12,1),(9,33),(104,2**32),(135,1),(60,1),(87,1),(17,2)])
def test_cpp_wire_validation_rejects_corrupt_binary(decoder,tmp_path,index,value):
    blob,_=lower_matrix_window(*linear());words=list(struct.unpack("<136Q",blob));words[index]=value
    decode(decoder,tmp_path,struct.pack("<136Q",*words),False)
