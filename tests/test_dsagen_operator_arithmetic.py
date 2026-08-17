from __future__ import annotations

from mlxsim.dsagen_dma import ElfSymbol
from mlxsim.dsagen_operator_sweep import compile_operator_proxy

SYMBOLS = {
    "mlx_dma_cold_region": ElfSymbol(0x100000, 131072),
    "mlx_dma_write_region": ElfSymbol(0x200000, 4096),
}
CASE = {"name": "BERT_512", "sequence": 512, "trip_count": 1}


def test_bsmm_and_fft_use_registered_pair_arithmetic() -> None:
    bsmm = {"name": "qkv_bsmm", "family": "qkv_bsmm", "block_size": 16, "stages": 4}
    _, bsmm_meta = compile_operator_proxy(bsmm, CASE, SYMBOLS, arithmetic_expanded=True)
    assert bsmm_meta["operation_counts"] == {"fma": 192, "add": 96}

    fft = {"name": "fft_cmp", "family": "fft", "stages": 7}
    _, fft_meta = compile_operator_proxy(fft, CASE, SYMBOLS, arithmetic_expanded=True)
    assert fft_meta["operation_counts"] == {"fma": 288, "add": 432, "shuffle": 12}


def test_swa_uses_tile_fma_groups_and_kv_load_waves() -> None:
    operator = {
        "name": "swa_w128_q32",
        "family": "swa",
        "window": 128,
        "query_tile": 32,
        "fma_repeats": 1,
        "score_fma_groups": 32,
        "sv_fma_groups": 32,
        "kv_load_waves": 4,
    }
    _, metadata = compile_operator_proxy(operator, CASE, SYMBOLS, arithmetic_expanded=True)
    assert metadata["operation_counts"]["fma"] == 256
    assert metadata["pipeline_counts"]["load"] == 32
    assert metadata["pipeline_counts"]["store"] == 4
    assert metadata["arithmetic_expanded"] is True
