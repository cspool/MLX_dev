import copy
import json
import struct
import subprocess
from pathlib import Path

import pytest

from mlxsim.model_control_program import control_program
from system_sim.physical_host.lowering import MAGIC,lower_control

ROOT=Path(__file__).resolve().parents[1]


def command():
    layout=lambda root:{"root":root,"dtype":"i64","shape":[2],"strides":[1],"offset":0,"storage_elements":2}
    layouts={"a":layout("a"),"out":layout("out")}
    bindings={"a":{"base":2**32,"bytes":16,"writable":False},"out":{"base":2**32+64,"bytes":16,"writable":True}}
    node={"id":"out","kind":"add","args":[{"value":"a"},3],"kwargs":{},"source_operator_id":7,"control_program":control_program("add")}
    return node,layouts,bindings


def test_binary_host_command_layout_preserves_64_bit_addresses():
    blob,route=lower_control(*command());words=struct.unpack("<80Q",blob)
    assert words[:6]==(MAGIC,1,2,0,0,0) and words[10]==2**32 and words[58]==2**32+64
    assert words[32:36]==(2,2,3,0) and route["command_bytes"]==640
    assert not route["pe_opcode"] and not route["mlx_system_verified"]


@pytest.mark.parametrize("damage",["leaf","readonly","overflow","capacity","view","missing","scalar"])
def test_host_lowering_rejects_changed_semantics_and_invalid_bindings(damage):
    node,layouts,bindings=copy.deepcopy(command())
    if damage=="leaf":node["control_program"]["phases"]["body"][0]^=128
    elif damage=="readonly":bindings["out"]["writable"]=False
    elif damage=="overflow":bindings["a"]["base"]=2**64-8
    elif damage=="capacity":bindings["a"]["bytes"]=8
    elif damage=="view":layouts["a"]["strides"]=[2]
    elif damage=="missing":del bindings["a"]
    else:node["args"][1]=2**63
    with pytest.raises(ValueError):lower_control(node,layouts,bindings)


def test_actual_rv64_host_elf_executes_control_data_paths(tmp_path):
    out=tmp_path/"host"
    result=subprocess.run([str(ROOT/".venv/bin/python"),"-m","scripts.verify_mlx_physical_host","--output",str(out)],cwd=ROOT,capture_output=True,text=True,timeout=180)
    assert result.returncode==0,result.stdout+result.stderr
    report=json.loads((out/"report.json").read_text())
    assert report["all_passed"] and report["cases"]==40 and report["half_conversion_patterns"]==65536
    assert report["serialized_commands_match_c_abi"]==22 and report["data_base"]==2**32
    assert report["host_load_store_and_loops_executed"] and not report["model_runner_integrated"] and not report["mlx_system_verified"]
