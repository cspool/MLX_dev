from mlxsim.fig21_timed_paths import compile_timed_path, normalize_path
from mlxsim.fig24_25_exact_paths import compile_fft_cmp_path, compile_swa_path
from mlxsim.fig24_25_full_mesh_paths import (
    compile_full_mesh_fft_cmp_path,
    compile_full_mesh_swa_path,
    compile_full_mesh_timed_path,
    normalize_full_mesh_path,
)


def _bert_b16_normalized():
    return normalize_path(
        fu_counts={"fma": 25769803776},
        load_bytes=36700160,
        store_bytes=100663296,
        stage_count=4,
        simd_width=32,
        vector_bytes=64,
        lanes=4,
    )


def test_qkv_full_mesh_conserves_four_strip_work() -> None:
    base = _bert_b16_normalized()
    full_mesh = normalize_full_mesh_path(base)
    _, old = compile_timed_path(name="old", normalized=base, scale=4)
    document, new = compile_full_mesh_timed_path(
        name="new", normalized=full_mesh, scale=4
    )
    assert new["operation_counts"] == old["operation_counts"]
    assert new["pipeline_counts"] == old["pipeline_counts"]
    assert new["memory_requests"] == old["memory_requests"]
    assert new["event_names_balanced"]
    assert new["dynamic_event_count"] == new["dynamic_event_demand_count"]
    assert all(len(coordinates) == 16 for coordinates in new["compute_coordinates_by_step"])
    assert len({tuple(block["pe"]) for block in document["blocks"]}) == 16
    assert new["max_active_instructions_per_pe"] == 2


def test_fft_full_mesh_conserves_four_strip_work() -> None:
    _, old = compile_fft_cmp_path(
        name="old", sequence_length=512, hidden_dimension=1024, batch=32, scale=4
    )
    document, new = compile_full_mesh_fft_cmp_path(
        name="new", sequence_length=512, hidden_dimension=1024, batch=32, scale=4
    )
    assert new["operation_counts"] == old["operation_counts"]
    assert new["pipeline_counts"] == old["pipeline_counts"]
    assert new["memory_requests"] == old["memory_requests"]
    assert new["dynamic_event_count"] == old["dynamic_event_count"]
    assert new["event_names_balanced"]
    assert new["dynamic_event_count"] == new["dynamic_event_demand_count"]
    assert all(
        len({tuple(coord) for coord in phase["coordinates"]}) == 16
        for phase in new["compute_coordinates_by_phase"]
    )
    assert new["max_active_instructions_per_pe"] <= 32
    assert all(0 <= value < 4 for block in document["blocks"] for value in block["pe"])


def test_swa_full_mesh_conserves_four_strip_work() -> None:
    arguments = {
        "sequence_length": 512,
        "hidden_dimension": 1024,
        "batch": 32,
        "window": 128,
        "query_tile": 32,
        "scale": 4,
    }
    _, old = compile_swa_path(name="old", **arguments)
    document, new = compile_full_mesh_swa_path(name="new", **arguments)
    assert new["operation_counts"] == old["operation_counts"]
    assert new["pipeline_counts"] == old["pipeline_counts"]
    assert new["memory_requests"] == old["memory_requests"]
    assert new["dynamic_event_count"] == 192
    assert new["dynamic_event_count"] > old["dynamic_event_count"]
    assert new["event_names_balanced"]
    assert new["dynamic_event_count"] == new["dynamic_event_demand_count"]
    assert all(
        len({tuple(coord) for coord in phase["coordinates"]}) == 16
        for phase in new["compute_coordinates_by_phase"]
    )
    assert len({tuple(block["pe"]) for block in document["blocks"]}) == 16
    assert new["max_active_instructions_per_pe"] == 5
