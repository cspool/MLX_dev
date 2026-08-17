from __future__ import annotations

from mlxsim.dsagen_dma import ElfSymbol, compile_dma_microtrace
from mlxsim.dsagen_overlay import canonical_json


def fixture_symbols() -> dict[str, ElfSymbol]:
    return {
        "mlx_dma_cold_region": ElfSymbol(address=0x100000, size=131072),
        "mlx_dma_write_region": ElfSymbol(address=0x200000, size=4096),
    }


def test_dma_microtrace_exact_counts_and_addresses() -> None:
    document, metadata = compile_dma_microtrace(
        fixture_symbols(), memory_backend="dsagen_dma"
    )
    assert document["memory_backend"] == "dsagen_dma"
    assert document["start_in_roi"] is True
    assert document["active_window"] == 16
    assert metadata["reads"] == metadata["stores"] == metadata["computes"] == 64
    assert len(document["blocks"]) == 16
    for index, block in enumerate(document["blocks"]):
        assert block["tag"] == index + 1
        assert block["pe"] == [index % 4, index // 4]
        assert block["trip_count"] == 4
        assert [item["pipeline"] for item in block["instructions"]] == [
            "load",
            "compute",
            "store",
        ]
        for instruction in (block["instructions"][0], block["instructions"][2]):
            assert len(instruction["memory_address_sequence"]) == 4
            assert all(address % 8 == 0 for address in instruction["memory_address_sequence"])


def test_fixed_control_changes_only_backend() -> None:
    dma, _ = compile_dma_microtrace(fixture_symbols(), memory_backend="dsagen_dma")
    fixed, _ = compile_dma_microtrace(fixture_symbols(), memory_backend="fixed")
    assert dma.pop("memory_backend") == "dsagen_dma"
    assert fixed.pop("memory_backend") == "fixed"
    assert dma == fixed


def test_dma_compiler_is_byte_deterministic() -> None:
    first, _ = compile_dma_microtrace(fixture_symbols(), memory_backend="dsagen_dma")
    second, _ = compile_dma_microtrace(fixture_symbols(), memory_backend="dsagen_dma")
    assert canonical_json(first) == canonical_json(second)
