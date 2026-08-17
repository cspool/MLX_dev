from scripts.compile_xavier_matched_attention import work


def test_n256_full_work_signature() -> None:
    shape = {
        "retained_length": 128,
        "hidden_dimension": 4096,
        "forward_stages": 8,
        "inverse_stages": 7,
    }
    assert work("fftcmp", 1_572_864, shape) == {
        "fma": 72_351_744,
        "add": 108_527_616,
        "shuffle": 1_572_864,
    }
    assert work("qk", 16_384, shape) == {"fma": 67_108_864}
    assert work("softmax", 128, shape) == {
        "fmax": 16_384,
        "fexp": 16_384,
        "add": 16_384,
    }
    assert work("sv", 524_288, shape) == {
        "fma": 67_108_864,
        "fdiv": 524_288,
    }


def test_n8192_full_work_signature() -> None:
    shape = {
        "retained_length": 4096,
        "hidden_dimension": 4096,
        "forward_stages": 13,
        "inverse_stages": 12,
    }
    assert work("fftcmp", 50_331_648, shape) == {
        "fma": 3_825_205_248,
        "add": 5_737_807_872,
        "shuffle": 50_331_648,
    }
    assert work("qk", 16_777_216, shape) == {"fma": 68_719_476_736}
    assert work("softmax", 4096, shape) == {
        "fmax": 16_777_216,
        "fexp": 16_777_216,
        "add": 16_777_216,
    }
    assert work("sv", 16_777_216, shape) == {
        "fma": 68_719_476_736,
        "fdiv": 16_777_216,
    }
