from __future__ import annotations

from collections import defaultdict

import pytest

from mlxsim.dsagen_overlay import (
    OverlayFixture,
    canonical_json,
    compile_aggregate_radix2_cdc,
    compile_radix2_cdc,
    pair_indices,
)


@pytest.mark.parametrize(
    ("operator_kind", "width"),
    [("bsmm", 8), ("bsmm", 16), ("bsmm", 64), ("fft", 8), ("fft", 256)],
)
def test_aggregate_conserves_pairwise_counts(operator_kind: str, width: int) -> None:
    _, pairwise = compile_radix2_cdc(operator_kind, width)
    config, aggregate = compile_aggregate_radix2_cdc(operator_kind, width)
    for key in (
        "stages",
        "pairs_per_stage",
        "total_pairs",
        "instruction_count",
        "memory_requests",
        "transfers",
        "operation_counts",
    ):
        assert aggregate[key] == pairwise[key]
    per_stage: dict[int, int] = defaultdict(int)
    for block in config["blocks"]:
        per_stage[int(block["tag"]) - 1] += int(block["trip_count"])
    assert set(per_stage.values()) == {width // 2}


def test_address_sequences_match_every_assigned_radix_pair() -> None:
    fixture = OverlayFixture()
    config, _ = compile_aggregate_radix2_cdc("bsmm", 64, fixture)
    for block in config["blocks"]:
        stage = int(block["tag"]) - 1
        pairs = block["logical_pairs"]
        loads = [
            item for item in block["instructions"] if item["pipeline"] == "load"
        ]
        store = next(
            item for item in block["instructions"] if item["pipeline"] == "store"
        )
        expected_first = []
        expected_second = []
        expected_output = []
        for pair in pairs:
            first, second = pair_indices(64, stage, pair)
            expected_first.append((stage * 64 + first) * fixture.scalar_bytes)
            expected_second.append((stage * 64 + second) * fixture.scalar_bytes)
            expected_output.append(((stage + 1) * 64 + first) * fixture.scalar_bytes)
        assert loads[0]["memory_address_sequence"] == expected_first
        assert loads[1]["memory_address_sequence"] == expected_second
        assert store["memory_address_sequence"] == expected_output


def test_assigned_pairs_share_fixed_route() -> None:
    config, _ = compile_aggregate_radix2_cdc("fft", 256)
    capacity = config["routing"]["mesh_width"] * config["routing"]["mesh_height"]
    for block in config["blocks"]:
        assert len({pair % capacity for pair in block["logical_pairs"]}) == 1
        xfer = block["instructions"][-1]
        assert xfer["pipeline"] == "xfer"
        assert xfer["route"]


def test_fft8192_is_bounded_and_exact() -> None:
    config, metadata = compile_aggregate_radix2_cdc("fft", 8192)
    assert metadata["stages"] == 13
    assert metadata["pairs_per_stage"] == 4096
    assert metadata["total_pairs"] == 53248
    assert metadata["block_count"] == 208
    assert metadata["max_trip_count"] == 256
    assert metadata["instruction_count"] == 53248 * 7
    assert metadata["memory_requests"] == 53248 * 3
    assert metadata["max_active_instruction_footprint_per_pe"] == 21
    assert len(canonical_json(config).encode()) < 5_000_000


def test_b64_reduces_static_blocks_without_padding() -> None:
    config, metadata = compile_aggregate_radix2_cdc("bsmm", 64)
    assert metadata["total_pairs"] == 192
    assert metadata["block_count"] == 96
    assert metadata["max_trip_count"] == 2
    assigned = [pair for block in config["blocks"] for pair in block["logical_pairs"]]
    assert len(assigned) == 192


def test_aggregate_compiler_is_deterministic() -> None:
    first, _ = compile_aggregate_radix2_cdc("fft", 256)
    second, _ = compile_aggregate_radix2_cdc("fft", 256)
    assert canonical_json(first) == canonical_json(second)
