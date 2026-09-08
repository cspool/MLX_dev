import copy
import json
import subprocess

import numpy as np
import pytest
import torch

from mlxsim.model_memory_program import Planner, make_layout, reshape_strides
from mlxsim.model_execution_inventory import ModelExecutionInventory
from mlxsim.model_tensor_compiler import compile_inventory
from test_model_tensor_semantics import execute_nodes, literal, native, node, ref


def planned(assets, nodes, *, same_device=True):
    planner = Planner()
    for identifier, spec in assets.items():
        planner.add_asset(identifier, spec)
    for item in nodes:
        planner.register(item, planned=True, same_device=same_device)
    return nodes


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.int64, torch.bool])
def test_views_and_copy_after_root_release_preserve_logical_values(native, tmp_path, dtype):
    x = torch.arange(48).reshape(2, 6, 4).to(dtype)
    trans = x.transpose(0, 2)
    sliced = trans[:, 1::2, :]
    reshaped = sliced.reshape(-1, 2)
    assets = {"x": literal(x)}
    nodes = planned(assets, [node(0, "transpose", [ref("x"), 0, 2], trans), node(1, "slice", [ref("v0"), 1, 1, 6, 2], sliced), node(2, "reshape", [ref("v1"), [-1, 2]], reshaped)])
    nodes[0]["release"] = ["x"]; nodes[1]["release"] = ["v0"]
    assert nodes[0]["memory_program"]["mode"] == nodes[1]["memory_program"]["mode"] == "view"
    assert nodes[2]["memory_program"]["mode"] == "transfer"
    result = execute_nodes((native[0], "none"), tmp_path / "views", assets, nodes)
    np.testing.assert_array_equal(result[-1], reshaped.numpy())
    report = json.loads((tmp_path / "views/out/result.json").read_text())["memory_programs"]
    assert report["calls"] == 3 and report["view_elisions"] == 2 and report["allocations"] == 1
    assert report["write_bytes"] == reshaped.numel() * reshaped.element_size()


def test_strided_view_is_not_unnecessarily_materialized(native, tmp_path):
    x = torch.arange(24).reshape(3, 8)
    sliced = x[:, ::2]
    view = sliced.view(-1)
    assets = {"x": literal(x)}
    nodes = [node(0, "slice", [ref("x"), 1, 0, 8, 2], sliced), node(1, "reshape", [ref("v0"), [-1]], view)]
    nodes[1]["source_operator"] = "aten.view.default"
    planned(assets, nodes)
    assert nodes[1]["memory_program"]["mode"] == "view"
    assert nodes[1]["memory_program"]["output_layout"]["strides"] == [2]
    actual = execute_nodes((native[0], "none"), tmp_path / "strided", assets, nodes)[-1]
    np.testing.assert_array_equal(actual, view.numpy())


def test_invalid_view_cannot_hide_a_copy():
    x = torch.arange(24).reshape(3, 8)
    t = x.transpose(0, 1)
    assets = {"x": literal(x)}
    items = [node(0, "transpose", [ref("x"), 0, 1], t), node(1, "reshape", [ref("v0"), [-1]], t.reshape(-1))]
    items[1]["source_operator"] = "aten.view.default"
    with pytest.raises(ValueError, match="no implicit copy"):
        planned(assets, items)


@pytest.mark.parametrize("target", [torch.float16, torch.float32, torch.int64, torch.bool])
def test_cast_preserves_dense_layout_and_requests_real_conversion(native, tmp_path, target):
    x = torch.linspace(-3, 3, 12).reshape(3, 4)
    trans = x.transpose(0, 1)
    expected = trans.to(target, copy=True, memory_format=torch.preserve_format)
    assets = {"x": literal(x)}
    nodes = planned(assets, [node(0, "transpose", [ref("x"), 0, 1], trans), node(1, "cast", [ref("v0"), str(target), False, True], expected, memory_format="torch.preserve_format")])
    assert nodes[1]["memory_program"]["mode"] == "transfer"
    assert nodes[1]["memory_program"]["output_layout"]["strides"] == list(expected.stride())
    actual = execute_nodes((native[0], "none"), tmp_path / "cast", assets, nodes)[-1]
    np.testing.assert_array_equal(actual, expected.numpy())


def test_same_dtype_contiguous_format_is_not_an_alias_for_transpose(native, tmp_path):
    x = torch.arange(12).reshape(3, 4).float()
    t = x.transpose(0, 1)
    expected = t.to(dtype=torch.float32, memory_format=torch.contiguous_format)
    assets = {"x": literal(x)}
    nodes = planned(assets, [node(0, "transpose", [ref("x"), 0, 1], t), node(1, "cast", [ref("v0"), "torch.float32"], expected, memory_format="torch.contiguous_format")])
    assert nodes[1]["memory_program"]["mode"] == "transfer"
    np.testing.assert_array_equal(execute_nodes((native[0], "none"), tmp_path / "contiguous", assets, nodes)[-1], expected.numpy())


def test_concat_empty_cache_and_strided_inputs(native, tmp_path):
    x = torch.arange(24).reshape(2, 3, 4).half()
    t = x.transpose(0, 2)
    empty = torch.empty(0, dtype=torch.float16)
    expected = torch.cat([empty, t, t], dim=1)
    assets = {"x": literal(x), "empty": literal(empty)}
    nodes = planned(assets, [node(0, "transpose", [ref("x"), 0, 2], t), node(1, "cat", [[ref("empty"), ref("v0"), ref("v0")], 1], expected)])
    actual = execute_nodes((native[0], "none"), tmp_path / "cat", assets, nodes)[-1]
    np.testing.assert_array_equal(actual.view(np.uint16), expected.numpy().view(np.uint16))


def test_embedding_uses_actual_indices_and_rejects_out_of_bounds(native, tmp_path):
    w = torch.arange(35).reshape(5, 7).half()
    ids = torch.tensor([[4, 1, 3]])
    expected = torch.nn.functional.embedding(ids, w)
    assets = {"w": literal(w), "ids": literal(ids)}
    nodes = planned(assets, [node(0, "embedding", [ref("w"), ref("ids")], expected)])
    np.testing.assert_array_equal(execute_nodes((native[0], "none"), tmp_path / "embedding", assets, nodes)[0], expected.numpy())
    bad = copy.deepcopy(assets); bad["ids"]["values"] = [[4, -1, 3]]
    error = execute_nodes((native[0], "none"), tmp_path / "bad-index", bad, nodes, success=False)
    assert "embedding index" in error


def test_zero_width_embedding_still_validates_indices(native, tmp_path):
    weight, ids = torch.empty(5,0), torch.tensor([1,4])
    expected = torch.nn.functional.embedding(ids, weight)
    assets = {"weight": literal(weight), "ids": literal(ids)}
    items = planned(assets, [node(0, "embedding", [ref("weight"), ref("ids")], expected)])
    actual = execute_nodes((native[0], "none"), tmp_path / "empty", assets, items)[0]
    assert actual.shape == expected.shape
    assets["ids"]["values"] = [1,-1]
    error = execute_nodes((native[0], "none"), tmp_path / "invalid-empty", assets, items, success=False)
    assert "embedding index" in error


@pytest.mark.parametrize("dtype", [torch.int64, torch.float16, torch.float32])
def test_where_copies_the_selected_value_without_float_narrowing(native, tmp_path, dtype):
    x = torch.tensor([2**60 + 1, 2**60 + 3], dtype=dtype) if dtype == torch.int64 else torch.tensor([1.5, -2.5], dtype=dtype)
    predicate = torch.tensor([[True], [False]])
    other = 2**60 + 9 if dtype == torch.int64 else -65504.0
    expected = torch.where(predicate, x, other)
    assets = {"x": literal(x), "p": literal(predicate)}
    nodes = planned(assets, [node(0, "where", [ref("p"), ref("x"), other], expected)])
    actual = execute_nodes((native[0], "none"), tmp_path / "where", assets, nodes)[0]
    np.testing.assert_array_equal(actual, expected.numpy())


def test_same_dtype_copy_preserves_nan_payload_and_signed_zero(native, tmp_path):
    raw = np.array([0x7C01, 0x7E55, 0xFC01, 0x8000, 0x0000], dtype=np.uint16)
    file = tmp_path / "raw.bin"; file.write_bytes(raw.tobytes())
    assets = {"x": {"kind": "mapped_file", "path": str(file), "byte_offset": 0, "bytes": raw.nbytes, "shape": [5], "dtype": "f16"}}
    nodes = planned(assets, [node(0, "cast", [ref("x"), "torch.float16", False, True], torch.zeros(5).half())])
    actual = execute_nodes((native[0], "none"), tmp_path / "payload", assets, nodes)[0]
    np.testing.assert_array_equal(actual.view(np.uint16), raw)


@pytest.mark.parametrize("corruption", ["stride", "offset", "word", "dropout"])
def test_memory_plan_corruption_is_rejected(native, tmp_path, corruption):
    x = torch.arange(4).float()
    assets = {"x": literal(x)}
    item = planned(assets, [node(0, "cast", [ref("x"), "torch.float16"], x.half())])[0]
    if corruption == "stride":
        item["memory_program"]["input_layouts"]["x"]["strides"] = [2]
    elif corruption == "offset":
        item["memory_program"]["output_layout"]["offset"] = 1
    elif corruption == "word":
        item["memory_program"]["words"] = [3]
    else:
        item = planned(assets, [node(0, "dropout_inference", [ref("x"), 0.1, False], x)])[0]
        item["args"][2] = True
    execute_nodes((native[0], "none"), tmp_path / "corrupt", assets, [item], success=False)


def test_memory_format_is_recorded_as_an_attribute_not_lost_python_type():
    tracer = ModelExecutionInventory(torch.nn.Identity())
    assert tracer.small(torch.contiguous_format) == "torch.contiguous_format"
    assert tracer.small(torch.preserve_format) == "torch.preserve_format"


@pytest.mark.parametrize("scheduled",[False,True])
def test_all_lowering_families_execute_one_generation_and_cache_graph(native, tmp_path, scheduled):
    model = torch.nn.Identity()
    trace = ModelExecutionInventory(model, capture_bindings=True)
    weight = torch.tensor([[3,1,2,0],[0,3,5,2],[1,0,5,2],[2,1,0,6]], dtype=torch.float16)
    projection = torch.eye(4, dtype=torch.float16)
    current = torch.tensor([[1]])
    cache = torch.empty(1,0,4, dtype=torch.float16)
    for name, value in (("weight",weight),("projection",projection),("input",current),("cache",cache)):
        trace.bind_input(name,value)
    expected = []
    for step in range(3):
        with torch.inference_mode(), trace.step(step, "prefill" if not step else "decode"):
            hidden = torch.nn.functional.embedding(current,weight)
            cache = torch.cat([cache,hidden],dim=1)
            pooled = cache.transpose(1,2).mean(-1)
            logits = torch.nn.functional.linear(pooled,projection)*2+1
            positions = torch.arange(4)
            limit = positions[:1]+step+1
            logits = torch.where(positions<=limit,logits,-65504.0)
            trace.phase = "token_selection"
            token = logits.argmax(-1)
            current = token.unsqueeze(1)
        expected.append((logits.numpy().copy(),token.tolist()))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map":{}}))
    inventory = trace.report()
    inventory.update(model_identity={"path":str(tmp_path),"family":"unit","variant":"unit","parameters":0,"files":{}},input={"batch":1},reference_checks=[{"forward_id":i} for i in range(3)])
    timing={"rows":1,"columns":1,"trace":False}
    program, compilation = compile_inventory(inventory,matrix_backend="scheduled" if scheduled else "microcode",vector_backend="scheduled" if scheduled else "microcode",schedule_options=timing if scheduled else None,vector_schedule_options=timing if scheduled else None,memory_backend="scheduled" if scheduled else "planned",control_backend="scheduled" if scheduled else "rv64_leaf",control_schedule_options={"response_period":3} if scheduled else None)
    assert all(sum(field in n for field in ("matrix_program","vector_program","memory_program","control_program"))==1 for n in program["nodes"])
    (tmp_path / "program.json").write_text(json.dumps(program))
    result = subprocess.run([str(native[0]),str(tmp_path/"program.json"),str(tmp_path/"out"),"none","1"],capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
    report = json.loads((tmp_path/"out/result.json").read_text())
    assert report["functional_entry_calls"]==0 and report["blas_calls"]==0
    assert report["control_programs"]["calls"]>0 and report["memory_programs"]["calls"]>0 and report["vector_microcode"]["calls"]>0
    if scheduled:
        assert report["matrix_windows"] and report["vector_windows"] and report["memory_windows"] and report["control_windows"]
        assert all(window["done"] and window["cycles"]>0 for window in report["matrix_windows"]+report["vector_windows"])
        assert all(w["done"] and w["dma_requests"]==w["dma_responses"] for w in report["memory_windows"])
        assert all(w["cycles"]==0 and w["dma_requests"]==0 for w in report["memory_windows"] if w["view_elided"])
        assert all(r["memory_plan_entry"]=="mlx::memory_model::Simulator" for r in compilation["routes"] if r["memory_plan_entry"])
        assert all(r["control_entry"]=="mlx::control_schedule::Simulator" for r in compilation["routes"] if r["control_entry"])
        assert all(w["done"] and w["dma_requests"]==w["dma_responses"] for w in report["control_windows"])
        assert sum(w["instructions"] for w in report["control_windows"])==report["control_programs"]["instructions"]
        assert report["timing_mode"]=="unmodeled"  # real host and cross-operator timing still missing
    for actual,(logits,tokens) in zip(report["outputs"],expected,strict=True):
        got=np.fromfile(actual["logits_file"],dtype=np.float16).reshape(actual["shape"])
        np.testing.assert_array_equal(got.view(np.uint16),logits.view(np.uint16))
        assert actual["tokens"]==tokens
    assert not report["mlx_system_verified"] and not report["performance_eligible"]
    if scheduled:
        for backend, message in (("memory", "strict memory program"), ("control", "strict controller program")):
            missing = copy.deepcopy(program)
            field = backend+"_program"
            del next(n for n in missing["nodes"] if field in n)[field]
            file = tmp_path / f"missing-{backend}-program.json"; file.write_text(json.dumps(missing))
            failed = subprocess.run([str(native[0]),str(file),str(tmp_path/f"missing-{backend}-out"),"none","1"],capture_output=True,text=True,timeout=60)
            assert failed.returncode != 0 and message+" has a missing lowering" in failed.stderr
