from __future__ import annotations

from scripts.compile_fig24_cross_simulator import compiler_operator, gpu_fma_count


def test_gpu_fma_counts_follow_frozen_source_formulas() -> None:
    fft = {
        "family": "fft",
        "gpu_parameter": 6,
        "gpu_fma_per_element_stage": 2,
    }
    assert gpu_fma_count(fft, 1024) == 12288
    bsmm = {
        "family": "qkv_bsmm",
        "gpu_parameter": 5,
        "gpu_fma_per_element_stage": 3,
    }
    assert gpu_fma_count(bsmm, 2048) == 30720
    swa = {
        "family": "swa",
        "gpu_parameter": 32,
        "gpu_fma_per_element_parameter": 1,
    }
    assert gpu_fma_count(swa, 4096) == 131072


def test_compiler_operator_does_not_consume_target_values() -> None:
    specification = {
        "name": "qkv_bsmm_b64",
        "family": "qkv_bsmm",
        "block_size": 64,
        "mlx_stages": 6,
    }
    assert compiler_operator(specification) == {
        "name": "qkv_bsmm_b64",
        "family": "qkv_bsmm",
        "block_size": 64,
        "stages": 6,
    }
