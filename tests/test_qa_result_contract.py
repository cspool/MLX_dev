"""QA task-head protocol components; these are not full BERT acceptance."""
import copy
import ctypes
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from mlxsim.model_execution_inventory import ModelExecutionInventory
from mlxsim.model_tensor_compiler import compile_inventory
from mlxsim.model_result_contract import QA_CONTRACT, QA_SELECTION, validate_result_contract, verify_qa_result
from mlxsim.model_value_outputs import require_value_contract
from mlxsim.model_block_pipeline import compile_block_pipelines
from mlxsim.model_ready_evidence import verify_ready_execution
from mlxsim.model_physical_evidence import verify_physical_execution
from system_sim.physical_host.graph_lowering import compile_graph

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def binaries():
    build = ROOT / "build/qa-runtime"
    for command in (["cmake", "-S", str(ROOT / "simulator_ext/model_system"), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"],
                    ["cmake", "--build", str(build), "--target", "mlx-tensor-semantics", "mlx-physical-model", "mlx-ready-graph", "-j4"]):
        r = subprocess.run(command, capture_output=True, text=True, timeout=240)
        assert r.returncode == 0, r.stdout + r.stderr
    return {"tensor": build / "model-storage/tensor-model/mlx-tensor-semantics",
            "physical": build / "mlx-physical-model", "paired": build / "mlx-ready-graph"}


def capture(path, length=28, unicode=False, perturb=False, composite=False):
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    context = ("阵列" if unicode else "ab") * (length - 2)
    offsets = [[0, 0]] + [[2 * i, 2 * i + 2] for i in range(length - 2)] + [[0, 0]]
    mask = [False] + [True] * (length - 2) + [False]
    torch.manual_seed(413)
    x = torch.randn(1, length, 2)
    x[0, 2 if perturb else 1] = torch.tensor([8., 8.])
    inputs = dict(input_ids=torch.arange(length).reshape(1, length),
                  token_type_ids=torch.tensor([[0] + [1] * (length - 2) + [0]]),
                  attention_mask=torch.tensor([[1] * (length - 1) + [0]]))
    trace = ModelExecutionInventory(torch.nn.Identity(), capture_bindings=True)
    trace.bind_input("x", x)
    for name, value in inputs.items(): trace.bind_input(name, value)
    input_tensors = {name: trace.tensor(value) for name, value in inputs.items()}
    with torch.inference_mode(), trace.step(0, "qa"):
        y = x * 2 + 1
        if composite: y = torch.nn.functional.gelu(y)
        a, b = y.split(1, -1)
        a, b = a.squeeze(-1).contiguous(), b.squeeze(-1).contiguous()
    outputs = {"start_logits": trace.tensor(a), "end_logits": trace.tensor(b)}
    inventory = trace.report()
    inventory.update(model_identity=dict(path=str(path), family="qa-protocol-component", variant="not-model-validation", parameters=0, files={}),
        reference_checks=[dict(forward_id=0, outputs=outputs, context=context, context_mask=mask, offsets=offsets,
                               input_tensors=input_tensors, input_values={k: v.tolist() for k, v in inputs.items()},
                               span_selection=QA_SELECTION, span={"text": "GOLDEN MUST NOT ENTER PROGRAM", "start": -100})])
    program, coverage = compile_inventory(inventory, matrix_backend="scheduled", vector_backend="scheduled",
                                          memory_backend="scheduled", control_backend="scheduled")
    require_value_contract(program)
    return program, inventory, coverage, (a.numpy(), b.numpy())


def run(binaries, mode, path, program, error=False):
    path.mkdir()
    (path / "program.json").write_text(json.dumps(program))
    options = dict(base=2**32, bytes=4 * 1024 * 1024, max_cycles=10**8, memory=dict(latency=3, accept_period=2, nack_every=5))
    if mode == "paired": options.update(tile_pipeline=True, template_load_timing=True)
    (path / "options.json").write_text(json.dumps(options))
    command = [str(binaries[mode]), str(path / "program.json")]
    command += [str(path / "out"), "none", "1"] if mode == "tensor" else [str(path / "options.json"), str(path / "out")]
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    (path / "run.log").write_text(result.stdout + result.stderr)
    if error:
        assert result.returncode != 0, result.stdout + result.stderr
        assert not (path / "out/result.json").exists()
        return
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((path / "out/result.json").read_text())
    for spec, row in zip(program["outputs"], report["outputs"], strict=True): verify_qa_result(program, spec, row)
    if mode == "physical": verify_physical_execution(program, report, options)
    if mode == "paired": verify_ready_execution(program, report, options)
    return report


@pytest.mark.parametrize("length,unicode,composite", [(28, False, False), (64, False, False), (8, True, True)])
def test_qa_named_results_actual_span_and_counted_readback(binaries, tmp_path, length, unicode, composite):
    program, inventory, coverage, expected = capture(tmp_path, length, unicode, composite=composite)
    assert program["schema"] == "mlx_tensor_semantics_v4" and program["output_contract"] == QA_CONTRACT
    assert coverage["source_calls"] == len(inventory["operations"])
    assert "GOLDEN MUST NOT ENTER PROGRAM" not in json.dumps(program)
    assert all("token" not in o and "logits" not in o for o in program["outputs"])
    paired = compile_block_pipelines(program, event_slots=4)
    for mode in binaries:
        report = run(binaries, mode, tmp_path / mode, paired if mode == "paired" else program)
        row = report["outputs"][0]
        assert row["span"]["start"] == row["span"]["end"] == 1
        assert row["span"]["text"] == ("阵列" if unicode else "ab")
        for role, values in zip(("start_logits", "end_logits"), expected, strict=True):
            actual = np.fromfile(row["outputs"][role]["file"], dtype="<f4").reshape(values.shape)
            if composite: np.testing.assert_allclose(actual, values, atol=3e-6, rtol=0)
            else: np.testing.assert_array_equal(actual, values)
        if mode != "tensor":
            assert report["host_readback_requests"] == 5 * length
            assert report["arena_drained"]["reserved_bytes"] == 0 and report["memory"]["nacks"] > 0
        bad = copy.deepcopy(row); bad["span"]["start"] += 1
        with pytest.raises(RuntimeError, match="actual computed logits"): verify_qa_result(program, program["outputs"][0], bad)


def test_qa_input_perturbation_changes_actual_answer(binaries, tmp_path):
    p, _, _, _ = capture(tmp_path, perturb=True)
    r = run(binaries, "physical", tmp_path / "perturb", p)
    assert r["outputs"][0]["span"]["start"] == r["outputs"][0]["span"]["end"] == 2


@pytest.mark.parametrize("damage", ["schema", "contract", "role", "asset_output", "owner_output", "forward", "release", "mask", "offset", "policy", "utf8", "golden"])
def test_qa_python_and_cpp_reject_bad_contracts(binaries, tmp_path, damage):
    p, _, _, _ = capture(tmp_path, unicode=True)
    out = p["outputs"][0]
    if damage == "schema": p["schema"] = "mlx_tensor_semantics_v2"
    if damage == "contract": p.pop("output_contract")
    if damage == "role": out["token"] = out.pop("end_logits")
    if damage == "asset_output": out["start_logits"] = next(iter(p["assets"]))
    if damage == "owner_output": out["start_logits"] = next(n["id"] for n in p["nodes"] if n["kind"] == "split")
    if damage == "forward": out["forward_id"] = 1
    if damage == "release": p["nodes"][-1]["release"].append(out["start_logits"])
    if damage == "mask": p["assets"][out["context_mask"]]["values"][1] = 2
    if damage == "offset": p["assets"][out["offsets_utf8"]]["values"][1] = [6, 3]
    if damage == "policy": out["max_answer_tokens"] = 31
    if damage == "utf8": p["assets"][out["offsets_utf8"]]["values"][1] = [1, 6]
    if damage == "golden": out["span"] = {"start": 1, "end": 1}
    with pytest.raises((ValueError, KeyError, TypeError)): require_value_contract(p)
    run(binaries, "paired", tmp_path / "bad", p, error=True)


def test_qa_system_abi_not_silently_relabelled(tmp_path):
    p, _, _, _ = capture(tmp_path)
    with pytest.raises(ValueError, match="validated lifetime"): compile_graph(p, {})


@pytest.mark.parametrize("damage", ["padding", "input_binding", "selection", "missing_output", "uncomputed"])
def test_capture_metadata_and_outputs_are_checked(tmp_path, damage):
    _, inv, _, _ = capture(tmp_path)
    check = inv["reference_checks"][0]
    if damage == "padding": check["context_mask"][-1] = True
    if damage == "input_binding": check["input_values"]["attention_mask"][0][0] = 0
    if damage == "selection": check["span_selection"] = "golden"
    if damage == "missing_output": check["outputs"].pop("end_logits")
    if damage == "uncomputed": check["outputs"]["start_logits"]["tensor_id"] = "absent"
    with pytest.raises(ValueError): compile_inventory(inv, matrix_backend="scheduled", vector_backend="scheduled", memory_backend="scheduled", control_backend="scheduled")


def test_portable_c_span_f64_ties_limits_masks_and_nonfinite(tmp_path):
    lib = tmp_path / "qa.so"
    subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-ffp-contract=off", "-fPIC", "-shared",
                    str(ROOT / "simulator_ext/control_model/qa_span.c"), "-o", str(lib)], check=True)
    class Span(ctypes.Structure):
        _fields_ = [("start", ctypes.c_size_t), ("end", ctypes.c_size_t), ("score", ctypes.c_double), ("candidates", ctypes.c_uint64)]
    fn = ctypes.CDLL(str(lib)).mlx_qa_select_span
    fn.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(Span)]
    rng = np.random.default_rng(941)
    for n in (1, 28, 31, 64):
        for trial in range(12):
            a = rng.normal(size=n).astype(np.float32); b = rng.normal(size=n).astype(np.float32)
            mask = (rng.random(n) > .2).astype(np.uint8)
            if trial == 0: a.fill(-2); b.fill(-3); mask.fill(1)
            if trial == 1: a[0] = 2**24; b.fill(1); b[-1] = 2; mask.fill(1)
            if trial == 2: a[0] = np.nan; mask[0] = 0
            if trial == 3: b[-1] = np.inf
            if trial == 4: mask[0] = 2
            if trial == 5: mask.fill(0)
            out = Span()
            status = fn(a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), n, 30, ctypes.byref(out))
            if trial in (2, 3, 4): assert status == 1; continue
            candidates = [(float(a[i]) + float(b[j]), i, j) for i in range(n) for j in range(i, min(n, i + 30)) if mask[i:j + 1].all()]
            if not candidates: assert status == 2; continue
            assert status == 0
            score, i, j = max(candidates, key=lambda v: v[0])
            assert (out.start, out.end, out.score, out.candidates) == (i, j, score, len(candidates))
