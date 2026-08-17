from __future__ import annotations

from mlxsim.dsagen_dma import ElfSymbol
from mlxsim.dsagen_operator_sweep import compile_operator_proxy, operator_stages
from mlxsim.dsagen_overlay import canonical_json

SYMBOLS = {
    "mlx_dma_cold_region": ElfSymbol(0x100000, 131072),
    "mlx_dma_write_region": ElfSymbol(0x200000, 4096),
}
CASE = {"name": "BERT_512", "sequence": 512, "trip_count": 1}


def test_operator_stage_depths_are_source_derived() -> None:
    fft = {"name": "fft_cmp", "family": "fft", "stages": 7}
    assert len(operator_stages(fft)) == 7
    for block_size, depth in ((16, 4), (32, 5), (64, 6)):
        operator = {
            "name": f"qkv_bsmm_b{block_size}",
            "family": "qkv_bsmm",
            "block_size": block_size,
            "stages": depth,
        }
        assert len(operator_stages(operator)) == depth


def test_swa_primitive_repeats_and_memory_counts() -> None:
    operator = {
        "name": "swa_w256_q64",
        "family": "swa",
        "window": 256,
        "query_tile": 64,
        "fma_repeats": 2,
    }
    document, metadata = compile_operator_proxy(operator, CASE, SYMBOLS)
    assert metadata["stage_count"] == 4
    assert metadata["operation_counts"] == {
        "fma": 16,
        "fmax": 4,
        "fexp": 4,
        "add": 4,
        "fdiv": 4,
    }
    assert metadata["memory_requests"] == 12
    assert document["memory_backend"] == "dsagen_dma"


def test_case_trip_count_scales_all_dynamic_counts_without_targets() -> None:
    operator = {"name": "qkv_bsmm", "family": "qkv_bsmm", "block_size": 16, "stages": 4}
    small, small_meta = compile_operator_proxy(operator, CASE, SYMBOLS)
    large_case = {"name": "BERT_8K", "sequence": 8192, "trip_count": 16}
    large, large_meta = compile_operator_proxy(operator, large_case, SYMBOLS)
    assert large_meta["pipeline_counts"] == {
        name: count * 16 for name, count in small_meta["pipeline_counts"].items()
    }
    assert large_meta["paper_target_values_consumed"] is False
    again, _ = compile_operator_proxy(operator, large_case, SYMBOLS)
    assert canonical_json(large) == canonical_json(again)
    assert small["active_window"] == large["active_window"] == 4
