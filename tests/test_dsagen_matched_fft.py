from mlxsim.dsagen_dma import ElfSymbol
from mlxsim.dsagen_matched_fft import compile_matched_fft


def _symbols() -> dict[str, ElfSymbol]:
    return {
        "mlx_dma_cold_region": ElfSymbol(0x1000, 0x100000),
        "mlx_dma_write_region": ElfSymbol(0x200000, 0x100000),
    }


def test_n256_unit_signature() -> None:
    document, metadata = compile_matched_fft(
        name="N256-q1",
        forward_stages=8,
        inverse_stages=7,
        scale=1,
        symbols=_symbols(),
    )
    assert metadata["stage_count"] == 16
    assert metadata["stage_trip_counts"] == [2] * 9 + [1] * 7
    assert metadata["operation_counts"] == {
        "fma": 1104,
        "add": 1656,
        "shuffle": 24,
    }
    assert len(document["blocks"]) == 192
    assert metadata["operation_counts"]["fma"] * 8192 * 8 == 72_351_744
    assert metadata["operation_counts"]["add"] * 8192 * 8 == 108_527_616
    assert metadata["operation_counts"]["shuffle"] * 8192 * 8 == 1_572_864


def test_n8192_unit_signature() -> None:
    _, metadata = compile_matched_fft(
        name="N8192-q1",
        forward_stages=13,
        inverse_stages=12,
        scale=1,
        symbols=_symbols(),
    )
    assert metadata["stage_count"] == 26
    assert metadata["stage_trip_counts"] == [2] * 14 + [1] * 12
    assert metadata["operation_counts"] == {
        "fma": 1824,
        "add": 2736,
        "shuffle": 24,
    }
    assert metadata["operation_counts"]["fma"] * 262144 * 8 == 3_825_205_248
    assert metadata["operation_counts"]["add"] * 262144 * 8 == 5_737_807_872
    assert metadata["operation_counts"]["shuffle"] * 262144 * 8 == 50_331_648
