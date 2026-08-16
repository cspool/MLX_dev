import pytest

from mlxsim.algorithm import (
    AttentionShape,
    hierarchical_bsmm_density,
    hybrid_compute_remaining,
    mixed_layer_compute_remaining,
)


def test_hierarchical_bsmm_density_matches_disclosed_block_sizes() -> None:
    assert hierarchical_bsmm_density(16) == pytest.approx(0.5)
    assert hierarchical_bsmm_density(32) == pytest.approx(0.3125)
    assert hierarchical_bsmm_density(64) == pytest.approx(0.1875)


def test_grouped_query_attention_reduces_kv_projection_width() -> None:
    internlm = AttentionShape("internlm2", 2048, 4096, query_heads=32, key_value_heads=8)
    llama = AttentionShape("llama2", 2048, 4096, query_heads=32, key_value_heads=32)
    assert internlm.qkv_projection_coefficient == pytest.approx(1.5)
    assert llama.qkv_projection_coefficient == pytest.approx(3.0)


def test_compute_remaining_decreases_with_stronger_structure() -> None:
    shape = AttentionShape("llama2", 512, 4096, query_heads=32, key_value_heads=32)
    by_block = [
        hybrid_compute_remaining(shape, block_size=block_size, compression_ratio=0.75)
        for block_size in (16, 32, 64)
    ]
    assert by_block[0] > by_block[1] > by_block[2]


def test_mixed_layer_sweep_has_dense_and_fully_modified_endpoints() -> None:
    modified = 0.3
    assert mixed_layer_compute_remaining(modified, total_layers=12, modified_layers=0) == 1.0
    assert mixed_layer_compute_remaining(
        modified, total_layers=12, modified_layers=12
    ) == pytest.approx(modified)
