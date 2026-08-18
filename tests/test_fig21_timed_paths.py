from mlxsim.fig21_timed_paths import normalize_path


def test_normalize_path_round_trip() -> None:
    normalized = normalize_path(
        fu_counts={"fma": 5 * 4 * 32 * 100},
        load_bytes=4 * 64 * 20,
        store_bytes=4 * 64 * 10,
        stage_count=5,
    )
    assert normalized["full_scale"] == 10
    assert normalized["unit_load_trip_per_lane"] == 2
    assert normalized["unit_store_trip_per_lane"] == 1
    assert normalized["unit_compute_steps"] == [
        {"operation": "fma", "trip_per_lane": 10}
    ] * 5
