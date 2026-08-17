from mlxsim.attention_signature import attention_work_signature

FFT_TEMPLATE = {
    "fma_per_pair": 4,
    "add_per_pair": 6,
    "analytical_flops_per_pair": 10,
    "shuffle_per_retained_element": 1,
}


def test_n256_attention_signature() -> None:
    signature = attention_work_signature(
        sequence_length=256,
        hidden_dimension=4096,
        batch=1,
        projections=3,
        compression_ratio=0.5,
        fft_template=FFT_TEMPLATE,
    )
    fft = signature["fft_compression"]
    attention = signature["compressed_attention"]
    assert fft["butterfly_pairs"] == 18_087_936
    assert fft["analytical_operations"] == 180_879_360
    assert fft["tagged_stage_count"] == 16
    assert attention["analytical_operations_excluding_fdiv"] == 268_533_760
    assert attention["fu_instruction_instances"]["fdiv"] == 524_288


def test_n8192_attention_signature() -> None:
    signature = attention_work_signature(
        sequence_length=8192,
        hidden_dimension=4096,
        batch=1,
        projections=3,
        compression_ratio=0.5,
        fft_template=FFT_TEMPLATE,
    )
    fft = signature["fft_compression"]
    attention = signature["compressed_attention"]
    assert fft["butterfly_pairs"] == 956_301_312
    assert fft["analytical_operations"] == 9_563_013_120
    assert fft["tagged_stage_count"] == 26
    assert attention["analytical_operations_excluding_fdiv"] == 274_978_570_240
    assert attention["fu_instruction_instances"]["fdiv"] == 16_777_216
