"""Actual RV64 control + versioned memory + PE compute, not a full BERT claim."""
import json
from pathlib import Path
import subprocess

import pytest
import torch

from mlxsim.model_control_program import control_program
from mlxsim.model_vector_program import vector_program
from test_model_tensor_semantics import literal, node, ref
from test_physical_model import compiled_nodes
from test_pair_graph import storage_binary, lifetime

ROOT = Path(__file__).resolve().parents[1]


def primitive_chain(perturb=False, wrong_guard=False):
    mask = torch.tensor([True, not wrong_guard, True])
    prototype = torch.tensor([9.], dtype=torch.float32)
    weight = torch.arange(9, dtype=torch.float32).reshape(3,3)
    if perturb: weight[1,1] = 20
    columns, indices = torch.tensor([1]), torch.arange(3)
    ones = torch.ones(1,1,3)
    selected = weight[indices, columns]
    logits = ones.squeeze(1) + selected
    items = [node(0,"all",[ref("mask")],mask.all()),
             node(1,"guard",[ref("v0"),True],torch.tensor(True)),
             node(2,"arange",[3],indices),
             node(3,"new_ones",[ref("prototype"),[1,1,3]],ones),
             node(4,"squeeze",[ref("v3"),1],ones.squeeze(1)),
             node(5,"advanced_index",[ref("weight"),[ref("v2"),ref("columns")]],selected),
             node(6,"add",[ref("v4"),ref("v5")],logits),
             node(7,"argmax",[ref("v6"),-1],logits.argmax(-1))]
    for i in (0,1,2,7): items[i]["control_program"] = control_program(items[i]["kind"],"f32" if i==7 else "i64")
    for item in items[2:]: item["control_dependencies"] = [ref("v1")]
    items[6]["vector_program"] = vector_program("add",["f32","f32"],"f32")
    assets = {name:literal(value) for name,value in (("mask",mask),("prototype",prototype),("weight",weight),("columns",columns))}
    return compiled_nodes(assets,items,"v6","v7"), logits


@pytest.mark.parametrize("variant", ["normal", "changed-data", "guard-mismatch"])
def test_cpu_dispatches_real_index_loads_and_rejects_wrong_branch(storage_binary, tmp_path, variant):
    program, expected = primitive_chain(variant=="changed-data", variant=="guard-mismatch")
    input_dir = tmp_path / "input"
    lifetime(storage_binary,input_dir,program)
    ref_file = tmp_path / "logits.bin"
    ref_file.write_bytes(expected.numpy().tobytes())
    reference = {"outputs":[{"forward_id":0,"dtype":"f32","shape":[1,3],
                             "logits_file":str(ref_file),"tokens":expected.argmax(-1).tolist()}]}
    (tmp_path / "reference.json").write_text(json.dumps(reference))
    out = tmp_path / "run"
    result = subprocess.run([str(ROOT / ".venv/bin/python"),"-m","scripts.run_mlx_spike_graph",
                             "--program",str(input_dir / "program.json"),"--lifetimes",str(input_dir / "life.json"),
                             "--reference",str(tmp_path / "reference.json"),"--output",str(out)],
                            cwd=ROOT,capture_output=True,text=True,timeout=300)
    if variant=="guard-mismatch":
        assert result.returncode != 0 and "generic graph ELF failed code 10" in result.stderr
        device = json.loads((out / "device.json").read_text())
        assert device["launches"] == 0 and not device["windows"] and device["memory_idle"]
        assert not (out / "report.json").exists()
        return
    assert result.returncode==0, result.stdout+result.stderr
    report = json.loads((out / "report.json").read_text())
    assert report["source_calls"]==8 and report["device_windows"]==3
    assert report["family_source_calls"]["control"]==4 and report["family_source_calls"]["view"]==1
    assert report["actual_cpu_dispatch"] and report["output_bytes_checked_in_elf"]
    assert not report["full_model_execution_verified"] and not report["rocket_execution_verified"]
    device = json.loads((out / "device.json").read_text())
    assert device["windows"][0]["numeric_instructions"]["read_bytes"]==0
    assert device["windows"][1]["numeric_instructions"]["index_reads"]==6
