from mlxsim.dsagen_combined_attention import compile_combined_attention


def _compile(shape: str, scale: int = 1):
    if shape == "N256":
        return compile_combined_attention(
            name=f"{shape}-u{scale}",
            sequence_length=256,
            retained_length=128,
            hidden_dimension=4096,
            forward_stages=8,
            inverse_stages=7,
            fft_scale=16 * scale,
            attention_scale=scale,
        )
    return compile_combined_attention(
        name=f"{shape}-u{scale}",
        sequence_length=8192,
        retained_length=4096,
        hidden_dimension=4096,
        forward_stages=13,
        inverse_stages=12,
        fft_scale=scale,
        attention_scale=2 * scale,
    )


def test_n256_full_work_and_bytes() -> None:
    _, metadata = _compile("N256")
    full = 128
    assert metadata["offchip_bytes"] * full == 7_340_032
    assert metadata["boundary_bytes"] * full == 3_145_728
    assert metadata["max_active_instruction_footprint_per_pe"] <= 32
    assert metadata["operation_counts"]["fma"] * full * 32 == (
        72_351_744 + 134_217_728
    )
    assert metadata["operation_counts"]["fdiv"] * full * 32 == 524_288


def test_n8192_full_work_and_bytes() -> None:
    _, metadata = _compile("N8192")
    full = 65_536
    assert metadata["offchip_bytes"] * full == 234_881_024
    assert metadata["boundary_bytes"] * full == 100_663_296
    assert metadata["max_active_instruction_footprint_per_pe"] <= 32
    assert metadata["operation_counts"]["fma"] * full * 32 == (
        3_825_205_248 + 137_438_953_472
    )
    assert metadata["operation_counts"]["fdiv"] * full * 32 == 16_777_216


def test_inverse_fft_waits_for_two_retained_packets() -> None:
    document, _ = _compile("N256")
    inverse = [
        block
        for block in document["blocks"]
        if "_fft_s9_" in block["id"]
    ]
    assert len(inverse) == 12
    assert all(list(block["wait_event_multiplicities"].values()) == [2] for block in inverse)
