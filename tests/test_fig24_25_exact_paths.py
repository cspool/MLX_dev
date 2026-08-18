from mlxsim.fig24_25_exact_paths import compile_fft_cmp_path, compile_swa_path


def test_fft_full_scale_is_batch32_work() -> None:
    _, m = compile_fft_cmp_path(name="fft", sequence_length=512, hidden_dimension=1024, batch=32, scale=1)
    assert m["full_scale"] == 32768
    assert m["stage_count"] == 18


def test_swa_full_scale_is_windowed_work() -> None:
    _, m = compile_swa_path(name="swa", sequence_length=512, hidden_dimension=1024, batch=32,
                            window=128, query_tile=32, scale=1)
    assert m["full_scale"] == 16384
    assert m["stage_count"] == 4
