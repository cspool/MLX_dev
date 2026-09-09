"""Versioned memory descriptors consume actual response data on the clocked bus."""
import copy
import struct

import pytest
import torch

from system_sim.physical_host.lowering import BYTES
from system_sim.physical_device.memory_lowering import lower_memory
from test_clocked_device import BASE, binary, run
from test_memory_wire_lowering import decoder, prepare, decode
from test_model_tensor_semantics import node, ref


def clocked_job(assets, nodes, *, initialize=True):
    item, layouts, bindings = prepare(assets, nodes)
    for binding in bindings.values(): binding["base"] += 65536
    blob, route = lower_memory(item, layouts, bindings)
    output = layouts[item["id"]]
    job = {"base": BASE, "bytes": 1048576, "memory_options": {"trace": True},
           "initial": [{"address": bindings[name]["base"], "data": list(value.numpy().tobytes())}
                       for name, value in assets.items()] if initialize else [],
           "commands": [{"address": BASE+4096, "data": list(blob), "source_id": item["source_operator_id"]}],
           "outputs": [{"address": bindings[output["root"]]["base"] + output["offset"]*BYTES[output["dtype"]],
                        "bytes": output["storage_elements"]*BYTES[output["dtype"]]}]}
    return job, item, blob, route


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.int64, torch.bool])
@pytest.mark.parametrize("retry", [False, True])
def test_indexed_values_and_negative_indices_come_from_system_memory(binary, decoder, tmp_path, dtype, retry):
    source = torch.arange(12).reshape(3,4).to(dtype)
    rows, cols = torch.tensor([[0],[-1]]), torch.tensor([[1,3,0]])
    expected = source[rows, cols]
    item = node(0, "advanced_index", [ref("x"), [ref("rows"), ref("cols")]], expected)
    job, _, blob, route = clocked_job({"x": source, "rows": rows, "cols": cols}, [item])
    assert len(blob) == 15936 and route["wire_version"] == 2
    if retry: job.update(latency=5, accept_period=3, nack_every=3)
    decoded = decode(decoder, tmp_path / "decode", blob)
    assert decoded["node"]["memory_program"]["profile"] == "mlx-memory-plan-v2"
    assert decoded["node"]["memory_program"]["words"] == item["memory_program"]["words"]
    actual = run(binary, tmp_path / "clocked", job)
    assert actual["outputs"] == [list(expected.numpy().tobytes())]
    kernel = actual["launches"][0]["kernel"]
    assert kernel["numeric_instructions"]["index_reads"] == 2*expected.numel()
    assert actual["launches"][0]["descriptor_bytes_fetched"] == 15936
    assert actual["launches"][0]["transport"]["idle"]


def test_eight_dimensional_index_template_keeps_repeated_source_loads(binary, decoder, tmp_path):
    source = torch.arange(256, dtype=torch.float32).reshape(*([2]*8))
    ids = torch.tensor([-1, 0])
    expected = source[tuple([ids]*8)]
    item = node(0, "advanced_index", [ref("x"), [ref("i")]*8], expected)
    job, _, blob, route = clocked_job({"x": source, "i": ids}, [item])
    assert route["operand_slots"] == 2 and route["argument_count"] == 9
    decoded = decode(decoder, tmp_path / "decode", blob)
    assert len(decoded["node"]["memory_program"]["words"]) == 11
    actual = run(binary, tmp_path / "clocked", job)
    assert actual["outputs"] == [list(expected.numpy().tobytes())]
    assert actual["launches"][0]["kernel"]["numeric_instructions"]["index_reads"] == 16


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.int64, torch.bool])
def test_new_ones_preserves_explicit_dtype_and_does_not_read_prototype(binary, decoder, tmp_path, dtype):
    expected = torch.ones((2,3), dtype=dtype)
    item = node(0, "new_ones", [ref("prototype"), [2,3]], expected)
    item["kwargs"]["dtype"] = str(dtype)
    job, _, blob, _ = clocked_job({"prototype": torch.tensor([9.], dtype=torch.float32)}, [item], initialize=False)
    # The prototype region deliberately has no initialized data. Only its
    # metadata is needed; any attempted source read must fail the test bus.
    decode(decoder, tmp_path / "decode", blob)
    actual = run(binary, tmp_path / "clocked", job)
    assert actual["outputs"] == [list(expected.numpy().tobytes())]
    tensors = [event for event in actual["commits"] if event["address"] >= BASE+65536]
    assert len(tensors) == expected.numel() and all(event["write"] for event in tensors)


@pytest.mark.parametrize("shape,axis", [([2,1,3],1), ([2,3],0), ([],0), ([], -1)])
def test_squeeze_view_preserves_data_without_tensor_dma(binary, decoder, tmp_path, shape, axis):
    source = torch.ones(shape, dtype=torch.float32)
    expected = source.squeeze(axis)
    job, _, blob, _ = clocked_job({"x": source}, [node(0, "squeeze", [ref("x"), axis], expected)])
    result = decode(decoder, tmp_path / "decode", blob)
    assert result["view_elided"]
    actual = run(binary, tmp_path / "clocked", job)
    assert actual["outputs"] == [list(expected.numpy().tobytes())]
    assert actual["launches"][0]["kernel"]["dma_requests"] == 0


def test_bad_runtime_index_does_not_publish_success(binary, tmp_path):
    source, ids = torch.arange(4, dtype=torch.float32), torch.tensor([4])
    item = node(0, "advanced_index", [ref("x"), [ref("i")]], torch.zeros(1))
    job, _, _, _ = clocked_job({"x": source, "i": ids}, [item])
    actual = run(binary, tmp_path / "bad-index", job)
    window = actual["launches"][0]
    assert not window["done"] and "index" in window["error"] and window["transport"]["idle"]
    assert not actual["outputs"] and not any(event["write"] for event in actual["commits"])


@pytest.mark.parametrize("word,value", [(1,1), (1,3), (2,1), (4,7), (5,64), (8,13), (11,1), (1985,1), (1991,1)])
def test_v2_wire_rejects_changed_version_capacity_and_reserved_fields(decoder, tmp_path, word, value):
    source, ids = torch.arange(4).float(), torch.tensor([1])
    _, _, blob, _ = clocked_job({"x":source,"i":ids}, [node(0,"advanced_index",[ref("x"),[ref("i")]],source[ids])])
    words = list(struct.unpack("<1992Q",blob)); words[word] = value
    decode(decoder, tmp_path / "bad", struct.pack("<1992Q",*words), True)


@pytest.mark.parametrize("damage", ["pinning", "layout", "instructions"])
def test_new_ones_does_not_drop_unsupported_source_semantics(damage):
    item, layouts, bindings = prepare({"x": torch.zeros(1)}, [node(0, "new_ones", [ref("x"),[2]],torch.ones(2))])
    item = copy.deepcopy(item)
    if damage == "pinning": item["kwargs"]["pin_memory"] = True
    elif damage == "layout": item["kwargs"]["layout"] = "torch.sparse_coo"
    else: item["memory_program"]["words"] = [1,3]
    with pytest.raises(ValueError): lower_memory(item, layouts, bindings)
