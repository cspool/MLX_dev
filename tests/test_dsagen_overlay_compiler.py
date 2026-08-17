from __future__ import annotations

import json

import pytest

from mlxsim.dsagen_overlay import (
    OverlayFixture,
    canonical_json,
    compile_radix2_cdc,
    greedy_route_steps,
    pair_indices,
)


def test_radix2_pair_indices() -> None:
    assert [pair_indices(8, 0, pair) for pair in range(4)] == [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
    ]
    assert [pair_indices(8, 1, pair) for pair in range(4)] == [
        (0, 2),
        (1, 3),
        (4, 6),
        (5, 7),
    ]
    assert [pair_indices(8, 2, pair) for pair in range(4)] == [
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]


@pytest.mark.parametrize(
    ("operator_kind", "expected_counts"),
    [
        ("bsmm", {"parameters": 48, "scalar_multiplies": 48, "scalar_adds": 24}),
        (
            "fft",
            {
                "complex_multiplies": 12,
                "complex_adds": 24,
                "real_multiplies": 48,
                "real_adds": 72,
            },
        ),
    ],
)
def test_b8_l8_exact_counts(operator_kind: str, expected_counts: dict[str, int]) -> None:
    config, metadata = compile_radix2_cdc(operator_kind, 8)
    assert metadata["stages"] == 3
    assert metadata["pairs_per_stage"] == 4
    assert metadata["total_pairs"] == 12
    assert metadata["block_count"] == 12
    assert metadata["instruction_count"] == (72 if operator_kind == "bsmm" else 84)
    assert metadata["memory_requests"] == 36
    assert metadata["transfers"] == 12
    assert metadata["operation_counts"] == expected_counts
    assert len(config["blocks"]) == 12


def test_block_layout_events_addresses_and_routes() -> None:
    fixture = OverlayFixture()
    config, metadata = compile_radix2_cdc("bsmm", 8, fixture)
    emitted: set[str] = set()
    waited: set[str] = set()
    for block in config["blocks"]:
        pipelines = [instruction["pipeline"] for instruction in block["instructions"]]
        assert pipelines == ["load", "load", "compute", "compute", "store", "xfer"]
        for instruction in block["instructions"]:
            if instruction["pipeline"] in {"load", "store"}:
                assert instruction["memory_address"] % fixture.scalar_bytes == 0
                assert instruction["memory_address"] < 16 * 1024 * 1024
            if event := instruction.get("emit_event"):
                assert event not in emitted
                emitted.add(event)
        waited.update(block["wait_events"])
    assert waited.issubset(emitted)
    assert len(metadata["event_edges"]) == 8
    assert all(
        edge["consumer_stage"] == edge["producer_stage"] + 1
        for edge in metadata["event_edges"]
    )
    assert all(route["steps"] for route in metadata["routes"])


def test_compiler_is_byte_deterministic() -> None:
    first, _ = compile_radix2_cdc("fft", 8)
    second, _ = compile_radix2_cdc("fft", 8)
    assert canonical_json(first) == canonical_json(second)
    assert json.loads(canonical_json(first)) == first
    assert [item["pipeline"] for item in first["blocks"][0]["instructions"]] == [
        "load",
        "load",
        "compute",
        "compute",
        "compute",
        "store",
        "xfer",
    ]


def test_signed_greedy_route() -> None:
    assert [step["step"] for step in greedy_route_steps((3, 3), (0, 0), (2, 1))] == [
        -2,
        -1,
        -2,
        -1,
    ]


def test_invalid_width_rejected() -> None:
    with pytest.raises(ValueError, match="power of two"):
        compile_radix2_cdc("bsmm", 6)


def test_b16_memory_stress_has_more_ready_pairs_than_request_buffer_slots() -> None:
    config, metadata = compile_radix2_cdc("bsmm", 16)
    assert metadata["stages"] == 4
    assert metadata["pairs_per_stage"] == 8
    assert metadata["total_pairs"] == 32
    assert metadata["memory_requests"] == 96
    assert sum(block["tag"] == 1 for block in config["blocks"]) == 8
