"""Real memory request/response progression, not a whole-model timing claim."""
import copy
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from mlxsim.model_tensor_compiler import compile_inventory
from test_model_memory_program import planned
from test_model_tensor_semantics import NUMPY, literal, node, ref

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/mlx-memory-window"


@pytest.fixture(scope="module")
def binary():
    for command in (["cmake", "-S", str(ROOT / "simulator_ext/memory_model"), "-B", str(BUILD), "-DCMAKE_BUILD_TYPE=Release"],
                    ["cmake", "--build", str(BUILD), "--target", "memory-external-memory", "-j4"]):
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
    return BUILD / "memory-external-memory"


def job(assets, nodes, **kwargs):
    return {"assets": assets, "nodes": planned(assets, nodes), **kwargs}


def run(binary, path, spec, error=None):
    path.mkdir(); (path / "job.json").write_text(json.dumps(spec))
    process = subprocess.run([str(binary), str(path / "job.json"), str(path / "out")], capture_output=True, text=True, timeout=60)
    if error:
        assert process.returncode != 0 and error in process.stderr, process.stdout + process.stderr
        assert not (path / "out/result.json").exists()
        return
    assert process.returncode == 0, process.stdout + process.stderr
    report = json.loads((path / "out/result.json").read_text()); windows = report["windows"]
    output = spec["nodes"][-1]["output"]
    values = np.fromfile(path / "out/output.bin", dtype=NUMPY[output["dtype"]]).reshape(output["shape"])
    assert report["functional_oracle_equal"]
    assert all(w["done"] and w["dma_requests"] == w["dma_responses"] and w["inflight_transactions"] == 0 for w in windows)
    assert all(w["cycles"] == 0 and w["dma_requests"] == 0 for w in windows if w["view_elided"])
    if spec.get("external", True):
        assert report["virtual_tensor_backing_used"] and all(w["external_memory_port"] for w in windows)
        commits = report["physical_commits"]
        requests = sum(w["dma_requests"] for w in windows)
        assert commits["reads"] + commits["writes"] == requests
        assert commits["read_bytes"] == sum(w["dma_read_bytes"] for w in windows)
        assert commits["write_bytes"] == sum(w["dma_write_bytes"] for w in windows)
        retries = report.get("transport", {}).get("nacks", 0)
        assert len(report["physical_requests"]) == requests + retries
        if not retries:
            ids = [r["id"] for r in report["physical_requests"]]
            assert len(set(ids)) == len(ids)
        assert all(r["address"] >= 2**32 for r in report["physical_requests"])
    for w in windows:
        issued = {}
        for e in w["trace"]:
            if e["event"] == "request":
                issued[e["request_id"]] = e
            if e["event"] == "response":
                assert e["cycle"] > issued[e["request_id"]]["cycle"]
        if w["trace"] and not w["trace_truncated"]:
            assert len(issued) == w["dma_requests"]
            assert len([e for e in w["trace"] if e["event"] == "response"]) == w["dma_responses"]
    assert not report["mlx_system_verified"] and not report["inference_performance_eligible"]
    return report, values


@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("target", [torch.float16, torch.float32, torch.int64, torch.bool])
def test_cast_layout_and_source_owner_survive_release(binary, tmp_path, external, target):
    x = torch.linspace(-3, 3, 30).reshape(5, 6); trans = x.T; expected = trans.to(target, copy=True)
    spec = job({"x": literal(x)}, [node(0, "transpose", [ref("x"), 0, 1], trans),
                node(1, "cast", [ref("v0"), str(target), False, True], expected)], external=external,
               options={"response_period": 7, "request_period": 2, "convert_latency": 3})
    spec["nodes"][0]["release"] = ["x"]
    report, actual = run(binary, tmp_path / "cast", spec)
    np.testing.assert_array_equal(actual, expected.numpy())
    assert report["windows"][1]["dma_requests"] == 60
    assert report["windows"][1]["response_stalls"] > 0


def test_reshape_copy_expand_contiguous_and_concat_aliases(binary, tmp_path):
    x = torch.arange(24).reshape(3, 8).half(); t = x.T; r = t.reshape(-1); e = r.unsqueeze(0).expand(2, -1)
    c = e.contiguous(); expected = torch.cat([c, c], -1)
    spec = job({"x": literal(x)}, [node(0, "transpose", [ref("x"), 0, 1], t),
        node(1, "reshape", [ref("v0"), [-1]], r), node(2, "unsqueeze", [ref("v1"), 0], r.unsqueeze(0)),
        node(3, "expand", [ref("v2"), [2, -1]], e), node(4, "contiguous", [ref("v3")], c),
        node(5, "cat", [[ref("v4"), ref("v4")], -1], expected)], buffered_bridge=True,
        options={"response_period": 5})
    report, actual = run(binary, tmp_path / "layout", spec)
    np.testing.assert_array_equal(actual.view(np.uint16), expected.numpy().view(np.uint16))
    assert report["transport"]["nacks"] > 0 and report["transport"]["idle"]
    assert report["transport"]["consumed"] == sum(w["dma_requests"] for w in report["windows"])


def test_embedding_indices_are_fetched_from_external_responses(binary, tmp_path):
    w = torch.arange(45).reshape(5, 9).float(); fake = torch.tensor([[0, 0]]); ids = torch.tensor([[4, 1]])
    expected = torch.nn.functional.embedding(ids, w)
    spec = job({"w": literal(w), "ids": literal(fake)}, [node(0, "embedding", [ref("w"), ref("ids")], expected)],
               physical_values={"ids": literal(ids)}, options={"response_period": 11})
    report, actual = run(binary, tmp_path / "embedding", spec)
    np.testing.assert_array_equal(actual, expected.numpy())
    assert report["windows"][0]["numeric_instructions"]["index_reads"] == expected.numel()
    assert report["physical_commits"]["reads"] == 2 * expected.numel()


@pytest.mark.parametrize("width", [0, 7])
@pytest.mark.parametrize("index", [-1, 5])
def test_invalid_embedding_response_cannot_be_ignored_even_for_empty_output(binary, tmp_path, width, index):
    w = torch.empty(5, width); ids = torch.tensor([0, index])
    spec = job({"w": literal(w), "ids": literal(ids)}, [node(0, "embedding", [ref("w"), ref("ids")], torch.empty(2, width))])
    run(binary, tmp_path / "invalid", spec, "embedding index out of range")


def test_zero_width_embedding_only_loads_indices(binary, tmp_path):
    spec = job({"w": literal(torch.empty(5, 0)), "ids": literal(torch.tensor([4, 0]))},
               [node(0, "embedding", [ref("w"), ref("ids")], torch.empty(2, 0))])
    report, actual = run(binary, tmp_path / "empty", spec)
    assert actual.shape == (2, 0) and report["physical_commits"]["reads"] == 2 and report["physical_commits"]["writes"] == 0
    assert report["windows"][0]["numeric_instructions"]["opcode_counts"]["4"] == 2


@pytest.mark.parametrize("kind", ["tensor", "literal"])
def test_where_uses_actual_predicate_and_reads_only_selected_operand(binary, tmp_path, kind):
    x = torch.tensor([2**60 + 1, 2**60 + 3]); p = torch.tensor([[True], [False]])
    other = torch.tensor([2**60 + 7, 2**60 + 9]) if kind == "tensor" else 2**60 + 13
    assets = {"x": literal(x), "p": literal(~p)}
    if kind == "tensor": assets["z"] = literal(other)
    expected = torch.where(p, x, other)
    spec = job(assets, [node(0, "where", [ref("p"), ref("x"), ref("z") if kind == "tensor" else other], expected)], physical_values={"p": literal(p)})
    report, actual = run(binary, tmp_path / "where", spec)
    np.testing.assert_array_equal(actual, expected.numpy())
    assert report["physical_commits"]["reads"] == (8 if kind == "tensor" else 6)
    assert report["windows"][0]["numeric_instructions"]["predicate_reads"] == 4


def test_bit_preserving_copy_and_tail(binary, tmp_path):
    raw = np.tile(np.array([0x7C01, 0x7E55, 0xFC01, 0x8000, 0], dtype=np.uint16), 7)
    file = tmp_path / "raw.bin"; file.write_bytes(raw.tobytes())
    assets = {"x": {"kind": "mapped_file", "path": str(file), "byte_offset": 0, "bytes": raw.nbytes, "shape": [35], "dtype": "f16"}}
    spec = job(assets, [node(0, "cast", [ref("x"), "torch.float16", False, True], torch.empty(35).half())])
    _, actual = run(binary, tmp_path / "bits", spec)
    np.testing.assert_array_equal(actual.view(np.uint16), raw)


def test_strided_slice_select_alias_and_device_copy(binary, tmp_path):
    x = torch.arange(24).reshape(3, 8).half(); sliced = x[:, 1:8:2]; selected = sliced[-1]
    assets = {"x": literal(x)}
    items = [node(0, "slice", [ref("x"), 1, 1, 8, 2], sliced),
        node(1, "select", [ref("v0"), 0, -1], selected), node(2, "alias", [ref("v1")], selected),
        node(3, "dropout_inference", [ref("v2"), 0.5, False], selected),
        node(4, "cast_device", [ref("v3"), "cuda:0", "torch.float16"], selected)]
    spec = {"assets": assets, "nodes": planned(assets, items, same_device=False)}
    items[0]["release"] = ["x"]; items[1]["release"] = ["v0"]
    report, actual = run(binary, tmp_path / "views", spec)
    np.testing.assert_array_equal(actual.view(np.uint16), selected.numpy().view(np.uint16))
    assert sum(w["view_elided"] for w in report["windows"]) == 4
    assert report["physical_commits"]["reads"] == report["physical_commits"]["writes"] == 4


def test_empty_valid_transfer_has_no_fabricated_requests(binary, tmp_path):
    x = torch.empty(0); spec = job({"x": literal(x)}, [node(0, "cast", [ref("x"), "torch.float16"], x.half())])
    report, actual = run(binary, tmp_path / "empty", spec)
    assert not actual.size and report["windows"][0]["cycles"] == report["windows"][0]["dma_requests"] == 0


@pytest.mark.parametrize("raw", [0x7F800000, 0x7FC00000, 0x5F000000])
def test_unregistered_float_to_integer_conversion_fails(binary, tmp_path, raw):
    file = tmp_path / "raw.bin"; file.write_bytes(np.array([raw], dtype=np.uint32).tobytes())
    assets = {"x": {"kind": "mapped_file", "path": str(file), "byte_offset": 0, "bytes": 4, "shape": [1], "dtype": "f32"}}
    spec = job(assets, [node(0, "cast", [ref("x"), "torch.int64"], torch.empty(1, dtype=torch.int64))])
    run(binary, tmp_path / "invalid-conversion", spec, "outside registered finite range")


@pytest.mark.parametrize("fault, message", [("token", "owner mismatch"), ("error", "reported error"),
    ("unsolicited", "owner mismatch"), ("unstable", "backpressure"), ("withdraw", "backpressure")])
@pytest.mark.parametrize("raw_port", [False, True])
def test_response_fault_does_not_produce_a_success_certificate(binary, tmp_path, fault, message, raw_port):
    x = torch.arange(5).float()
    spec = job({"x": literal(x)}, [node(0, "cast", [ref("x"), "torch.float16"], x.half())],
               fault=fault, raw_logical_port=raw_port, options={"response_period": 19})
    run(binary, tmp_path / "fault", spec, message)


@pytest.mark.parametrize("overrides, message", [({"0": {"readable": False}}, "permission"),
    ({"1": {"writable": False}}, "permission"), ({"0": {"bytes": 2}}, "out of bounds"),
    ({"1": {"base": 2**32}}, "overlap"), ({"0": {"base": 2**32 + 1}}, "misaligned")])
def test_physical_access_contract(binary, tmp_path, overrides, message):
    x = torch.arange(5).float()
    spec = job({"x": literal(x)}, [node(0, "cast", [ref("x"), "torch.float16"], x.half())], region_overrides=overrides)
    run(binary, tmp_path / "bounds", spec, message)


@pytest.mark.parametrize("words", [[], [3], [1, 258, 3], [4, 1, 2, 3]])
def test_empty_output_still_validates_the_whole_template(binary, tmp_path, words):
    x = torch.empty(0); spec = job({"x": literal(x)}, [node(0, "cast", [ref("x"), "torch.float16"], x.half())])
    spec["nodes"][0]["memory_program"]["words"] = words
    run(binary, tmp_path / "words", spec, "memory transfer")


def test_backpressure_changes_cycles_without_changing_data(binary, tmp_path):
    x = torch.linspace(-1, 1, 65)
    spec = job({"x": literal(x)}, [node(0, "cast", [ref("x"), "torch.float16"], x.half())])
    fast, values = run(binary, tmp_path / "fast", spec)
    slow_spec = copy.deepcopy(spec); slow_spec["options"] = {"request_period": 7, "response_period": 13, "convert_latency": 9}
    slow, slow_values = run(binary, tmp_path / "slow", slow_spec)
    np.testing.assert_array_equal(values.view(np.uint16), slow_values.view(np.uint16))
    assert slow["windows"][0]["cycles"] > fast["windows"][0]["cycles"]
    assert fast["windows"][0]["numeric_instructions"] == slow["windows"][0]["numeric_instructions"]


@pytest.mark.parametrize("options", [{"request_period": 0}, {"convert_latency": -1}, {"typo": 1}])
def test_invalid_timing_is_rejected(binary, tmp_path, options):
    x = torch.ones(1); spec = job({"x": literal(x)}, [node(0, "cast", [ref("x"), "torch.float16"], x.half())], options=options)
    run(binary, tmp_path / "options", spec, "memory")


def test_cycle_bound_is_not_a_silent_partial_result(binary, tmp_path):
    x = torch.ones(1); spec = job({"x": literal(x)}, [node(0, "cast", [ref("x"), "torch.float16"], x.half())], options={"max_cycles": 2})
    run(binary, tmp_path / "timeout", spec, "max_cycles")


def test_schedule_options_cannot_be_attached_to_an_unselected_backend():
    with pytest.raises(ValueError, match="memory timing options"):
        compile_inventory({}, memory_backend="planned", memory_schedule_options={})
