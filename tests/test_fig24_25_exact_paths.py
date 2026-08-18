from mlxsim.fig24_25_exact_paths import compile_fft_cmp_path, compile_swa_path
from scripts.audit_fig24_25_exact_paths import reconstruct_full_work


def test_fft_full_scale_is_batch32_work() -> None:
    _, m = compile_fft_cmp_path(name="fft", sequence_length=512, hidden_dimension=1024, batch=32, scale=1)
    assert m["full_scale"] == 32768
    assert m["stage_count"] == 18


def test_swa_full_scale_is_windowed_work() -> None:
    _, m = compile_swa_path(name="swa", sequence_length=512, hidden_dimension=1024, batch=32,
                            window=128, query_tile=32, scale=1)
    assert m["full_scale"] == 16384
    assert m["stage_count"] == 4


def test_fft_reconstructs_full_scalar_work_and_bytes() -> None:
    _, metadata = compile_fft_cmp_path(
        name="fft", sequence_length=512, hidden_dimension=1024, batch=32, scale=4
    )
    contract = {
        "family": "fft",
        "full_scale": 32768,
        "case": {"batch": 32, "n": 512, "d": 1024},
        "actual": {
            "stage_count": 18,
            "fu": {
                "fma": 1308622848,
                "add": 1962934272,
                "shuffle": 25165824,
            },
        },
    }
    reconstructed = reconstruct_full_work(
        contract=contract, metadata=metadata, vector_bytes=64, simd_width=32
    )
    assert reconstructed["pass"]
    assert reconstructed["reconstructed_load_bytes"] == 100663296
    assert reconstructed["reconstructed_store_bytes"] == 50331648


def test_swa_reconstructs_full_scalar_work_and_bytes() -> None:
    _, metadata = compile_swa_path(
        name="swa",
        sequence_length=512,
        hidden_dimension=1024,
        batch=32,
        window=128,
        query_tile=32,
        scale=4,
    )
    contract = {
        "family": "swa",
        "full_scale": 16384,
        "case": {"batch": 32, "n": 512, "d": 1024},
        "actual": {
            "stage_count": 4,
            "query_tile": 32,
            "fu": {
                "fma": 4294967296,
                "fmax": 2097152,
                "fexp": 2097152,
                "add": 2097152,
                "fdiv": 16777216,
            },
        },
    }
    reconstructed = reconstruct_full_work(
        contract=contract, metadata=metadata, vector_bytes=64, simd_width=32
    )
    assert reconstructed["pass"]
    assert reconstructed["reconstructed_load_bytes"] == 100663296
    assert reconstructed["reconstructed_store_bytes"] == 33554432
