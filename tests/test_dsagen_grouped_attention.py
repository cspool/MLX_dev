from mlxsim.dsagen_grouped_attention import compile_grouped_attention


def test_n256_grouped_attention_work() -> None:
    document, metadata = compile_grouped_attention(
        name="N256-q1",
        retained_length=128,
        hidden_dimension=4096,
        scale=1,
    )
    assert metadata["operation_counts"] == {
        "fma": 32768,
        "fmax": 4,
        "fexp": 4,
        "add": 4,
        "fdiv": 128,
    }
    assert len(document["blocks"]) == 20
    full_scale = 512
    assert metadata["operation_counts"]["fma"] * full_scale * 8 == 134_217_728
    assert metadata["operation_counts"]["fmax"] * full_scale * 8 == 16_384
    assert metadata["operation_counts"]["fdiv"] * full_scale * 8 == 524_288


def test_n8192_grouped_attention_work() -> None:
    document, metadata = compile_grouped_attention(
        name="N8192-q1",
        retained_length=4096,
        hidden_dimension=4096,
        scale=1,
    )
    assert metadata["operation_counts"] == {
        "fma": 32768,
        "fmax": 4,
        "fexp": 4,
        "add": 4,
        "fdiv": 4,
    }
    assert all(
        block.get("wait_event_period", 1) == 4096
        for block in document["blocks"]
        if "_sv_l" in block["id"]
    )
    full_scale = 524288
    assert metadata["operation_counts"]["fma"] * full_scale * 8 == 137_438_953_472
    assert metadata["operation_counts"]["fmax"] * full_scale * 8 == 16_777_216
    assert metadata["operation_counts"]["fdiv"] * full_scale * 8 == 16_777_216
