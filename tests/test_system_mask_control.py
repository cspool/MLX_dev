"""BERT mask/control ABI executes RV64 C, not sampled Python branch values."""
import copy
import json
from pathlib import Path
import struct
import subprocess

import pytest

from mlxsim.model_control_program import control_program
from mlxsim.model_memory_program import make_layout
from scripts.verify_mlx_physical_host import cases, mask_cases
from system_sim.physical_host.lowering import lower_control
from system_sim.physical_host.pair_graph import schedule

ROOT = Path(__file__).resolve().parents[1]


def control(kind):
    layouts = {"a": make_layout("bool", [1], "a"), "b": make_layout("bool", [1], "b"),
               "out": make_layout("bool", [] if kind in {"all", "guard"} else [1], "out")}
    bindings = {name: {"base": 2**32 + 64*i, "bytes": 1, "writable": name == "out"}
                for i, name in enumerate(layouts)}
    args = [{"value": "a"}]
    if kind == "guard": args.append(True)
    elif kind != "all": args.append({"value": "b"})
    return {"id": "out", "source_operator_id": 21, "kind": kind, "args": args,
            "kwargs": {}, "control_program": control_program(kind)}, layouts, bindings


@pytest.mark.parametrize("kind,opcode", [("ge", 6), ("bitwise_and", 7), ("all", 8), ("guard", 9)])
def test_mask_controls_have_distinct_versioned_cpu_route(kind, opcode):
    raw, route = lower_control(*control(kind))
    words = struct.unpack("<80Q", raw)
    assert words[1:6] == (2, opcode, 0, 0, 0) and len(raw) == 640
    assert words[10] == 2**32 and words[58] == 2**32+128
    assert route["abi_version"] == 2 and not route["pe_opcode"]
    if kind == "guard": assert words[32:36] == (2, 3, 1, 0)


@pytest.mark.parametrize("damage", ["leaf", "expected", "input_dtype", "multiple", "output", "scalar"])
def test_guard_lowering_cannot_replace_runtime_condition(damage):
    node, layouts, bindings = copy.deepcopy(control("guard"))
    if damage == "leaf": node["control_program"]["phases"]["body"][0] ^= 128
    elif damage == "expected": node["args"][1] = 1
    elif damage == "input_dtype": layouts["a"]["dtype"] = "i64"
    elif damage == "multiple": layouts["a"]["shape"] = [2]
    elif damage == "output": layouts["out"]["shape"] = [1]
    else: node["args"][0] = True
    with pytest.raises(ValueError): lower_control(node, layouts, bindings)


def test_and_rejects_scalar_operand_even_if_boolean():
    node, layouts, bindings = control("bitwise_and")
    node["args"][1] = False
    with pytest.raises(ValueError, match="Boolean tensors"): lower_control(node, layouts, bindings)


def test_actual_rv64_mask_paths_guards_and_legacy_contracts(tmp_path):
    out = tmp_path / "rv64"
    result = subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "scripts.verify_mlx_physical_host",
                             "--include-mask-control", "--output", str(out)], cwd=ROOT,
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((out / "report.json").read_text())
    assert report["cases"] == len(cases(True)) == 71
    assert report["serialized_commands_match_c_abi"] == sum(c["status"] == 0 for c in cases(True))
    assert report["mask_control_v2"]["cases"] == len(mask_cases()) == 31
    assert report["mask_control_v2"]["guard_mismatch_cases"] == 2
    assert report["mask_control_v2"]["failure_outputs_remained_poisoned"]
    assert report["data_base"] == 2**32 and not report["rocket_execution_verified"]


def test_cpu_pair_graph_keeps_guard_edges_for_independent_later_data():
    # Later arithmetic reads an unrelated asset; the guard is a control edge,
    # not inferable from tensor arguments. It must enter the CPU source table.
    guard, _, _ = control("guard")
    guard.update(id="v0", source_operator_id=0, output={"dtype":"bool","shape":[]})
    later = {"id":"v1","source_operator_id":1,"kind":"add","args":[{"value":"x"},1],
             "kwargs":{},"output":{"dtype":"i64","shape":[1]},
             "control_program":control_program("add"),"control_dependencies":[{"value":"v0"}]}
    program = {"schema":"mlx_tensor_semantics_v1","assets":{"a":{},"x":{}},"nodes":[guard,later],"outputs":[]}
    groups, dependencies, _ = schedule(program, 2)
    assert groups == [[0],[1]] and dependencies == [[],[0]]
    del later["control_dependencies"]
    with pytest.raises(ValueError, match="guard dependency"): schedule(program, 2)
