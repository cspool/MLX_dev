from mlxsim.fig19_source_paths import compile_fft2d_path


def test_fft2d_full_work_and_packets() -> None:
    _, metadata = compile_fft2d_path(name="N128-fft2d-q1", sequence_length=128, scale=1)
    full = metadata["full_scale"]
    stages = metadata["stage_count"]
    pairs = 512 * 128 * stages
    assert metadata["operation_counts"]["fma"] * full * 32 == 4 * pairs
    assert metadata["operation_counts"]["add"] * full * 32 == 6 * pairs
    assert metadata["offchip_bytes"] * full == 2 * 128 * 1024 * 2
