from __future__ import annotations

from mlxsim.dsagen_dma import ElfSymbol
from mlxsim.dsagen_full_block import compile_full_block, stage_graph
from mlxsim.dsagen_overlay import canonical_json


def symbols() -> dict[str, ElfSymbol]:
    return {
        "mlx_dma_cold_region": ElfSymbol(address=0x100000, size=131072),
        "mlx_dma_write_region": ElfSymbol(address=0x200000, size=4096),
    }


def test_frozen_stage_graph_and_operator_coverage() -> None:
    document, metadata = compile_full_block(symbols(), memory_backend="dsagen_dma")
    assert metadata["stage_count"] == len(stage_graph()) == 28
    assert metadata["stage_groups"][0] == "pre_attention_rmsnorm"
    assert metadata["stage_groups"][-1] == "final_residual_store"
    assert metadata["logical_lanes"] == 4
    assert document["active_window"] == 4
    assert document["start_in_roi"] is True
    assert set(metadata["operation_counts"]) == {
        "add",
        "mul",
        "fma",
        "fmax",
        "fexp",
        "fdiv",
        "frsqrt",
        "shuffle",
    }


def test_tags_events_placement_and_memory_are_bounded() -> None:
    document, metadata = compile_full_block(symbols(), memory_backend="dsagen_dma")
    events = {
        instruction["emit_event"]: block["tag"]
        for block in document["blocks"]
        for instruction in block["instructions"]
        if instruction.get("emit_event")
    }
    assert sorted({block["tag"] for block in document["blocks"]}) == list(range(1, 29))
    assert len(metadata["final_events"]) == 4
    for block in document["blocks"]:
        assert block["pe"] == [int(block["id"].rsplit("lane", 1)[1]), (block["tag"] - 1) % 4]
        assert block["trip_count"] == 2
        assert all(events[event] == block["tag"] - 1 for event in block["wait_events"])
        pipelines = [instruction["pipeline"] for instruction in block["instructions"]]
        if "load" in pipelines:
            assert pipelines[0] == "load"
        if "store" in pipelines and "xfer" in pipelines:
            assert pipelines.index("store") < pipelines.index("xfer")
        for instruction in block["instructions"]:
            if instruction["pipeline"] in {"load", "store"}:
                assert instruction["memory_bytes"] == 16
                assert len(instruction["memory_address_sequence"]) == 2
                assert all(address % 16 == 0 for address in instruction["memory_address_sequence"])
    assert metadata["memory_requests"] == 40


def test_fixed_and_dma_documents_differ_only_by_backend_and_are_deterministic() -> None:
    first, _ = compile_full_block(symbols(), memory_backend="dsagen_dma")
    second, _ = compile_full_block(symbols(), memory_backend="dsagen_dma")
    fixed, _ = compile_full_block(symbols(), memory_backend="fixed")
    assert canonical_json(first) == canonical_json(second)
    assert first.pop("memory_backend") == "dsagen_dma"
    assert fixed.pop("memory_backend") == "fixed"
    assert first == fixed
