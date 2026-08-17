from __future__ import annotations

import json

import pytest

from mlxsim.fig10_mapping import (
    DEFAULT_FIXTURE,
    canonical_json,
    compile_fig10_mapping,
    fixed_memory_control,
)
from scripts.compile_fig10_mapping import digest


@pytest.mark.parametrize("operator,instructions", [("bsmm", 5), ("fft", 6)])
def test_figure10_loop_conservation(operator: str, instructions: int) -> None:
    document, metadata = compile_fig10_mapping(operator, 1024)  # type: ignore[arg-type]
    assert metadata["outer_i0_trip"] == 16
    assert metadata["local_i1_trip"] == 4
    assert metadata["spatial_i2_trip"] == 16
    assert metadata["outputs_per_pe_per_stage"] == 64
    assert metadata["output_instances"] == 1024 * 10
    assert metadata["instruction_count"] == 1024 * 10 * instructions
    assert len(document["blocks"]) == 16 * 10
    assert {block["trip_count"] for block in document["blocks"]} == {64}


def test_bsmm_template_and_cdc_memory_boundaries() -> None:
    document, metadata = compile_fig10_mapping("bsmm", 128)
    first = document["blocks"][0]
    assert [instruction["operation"] for instruction in first["instructions"]] == [
        "load",
        "load",
        "mul",
        "fma",
        "xfer",
    ]
    assert first["instructions"][0]["memory_external"] is True
    internal = document["blocks"][16]
    assert internal["instructions"][0]["memory_external"] is False
    boundary = document["blocks"][5 * 16]
    assert boundary["instructions"][-1]["operation"] == "store"
    assert metadata["cdc_starts"] == [0, 6]
    assert metadata["cdc_ends"] == [5, 6]
    assert metadata["memory_requests"] == 2 * 128 * 2 + 128 * 2
    assert sum(metadata["expected_pipeline_instructions"].values()) == metadata[
        "instruction_count"
    ]


def test_stride_mapping_matches_figure10_axes_and_self_hops() -> None:
    document, _ = compile_fig10_mapping("bsmm", 64)
    routes = document["metadata"]["routes"]
    stage0 = next(item for item in routes if item["stage"] == 0 and item["source"] == [0, 0])
    stage1 = next(item for item in routes if item["stage"] == 1 and item["source"] == [0, 0])
    stage2 = next(item for item in routes if item["stage"] == 2 and item["source"] == [0, 0])
    stage3 = next(item for item in routes if item["stage"] == 3 and item["source"] == [0, 0])
    stage4 = next(item for item in routes if item["stage"] == 4 and item["source"] == [0, 0])
    assert stage0["destination"] == [1, 0]
    assert stage1["destination"] == [2, 0]
    assert stage2["destination"] == [0, 1]
    assert stage3["destination"] == [0, 2]
    assert stage4["destination"] == [0, 0]
    assert document["metadata"]["route_hops"] == 4 * 64


def test_instruction_footprint_fits_disclosed_store() -> None:
    for operator in ("bsmm", "fft"):
        _, metadata = compile_fig10_mapping(operator, 8192)
        assert metadata["max_active_instruction_footprint_per_pe"] <= 32
        assert metadata["simd_width"] == 8
        assert metadata["paper_performance_targets_consumed"] is False


def test_compiler_and_fixed_control_are_deterministic() -> None:
    first, _ = compile_fig10_mapping("fft", 256)
    second, _ = compile_fig10_mapping("fft", 256)
    assert canonical_json(first) == canonical_json(second)
    fixed = fixed_memory_control(first)
    changed = json.loads(canonical_json(fixed))
    assert changed["memory_backend"] == "fixed"
    assert changed["blocks"] == first["blocks"]


def test_invalid_width_is_rejected() -> None:
    with pytest.raises(ValueError):
        compile_fig10_mapping("bsmm", 32)
    with pytest.raises(ValueError):
        compile_fig10_mapping("bsmm", 96, DEFAULT_FIXTURE)


def test_compiler_digest_is_callable(tmp_path) -> None:
    path = tmp_path / "sample.json"
    path.write_text("{}\n", encoding="utf-8")
    value = digest(path)
    assert value["bytes"] == 3
    assert len(value["sha256"]) == 64
