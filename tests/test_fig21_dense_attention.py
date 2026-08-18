from mlxsim.fig21_dense_attention import compile_dense_attention


def test_dense_attention_full_scale_counts() -> None:
    _, metadata = compile_dense_attention(
        name="N256-u1", sequence_length=256, hidden_dimension=4096, scale=1
    )
    full = metadata["full_scale"]
    assert full == 4096
    assert metadata["operation_counts"]["fma"] * full * 32 == 2 * 8 * 256 * 256 * 4096
    assert metadata["operation_counts"]["fmax"] * full * 32 == 8 * 256 * 256
    assert metadata["operation_counts"]["fdiv"] * full * 32 == 8 * 256 * 4096
    assert metadata["offchip_bytes"] * full == 4 * 8 * 256 * 4096 * 2
