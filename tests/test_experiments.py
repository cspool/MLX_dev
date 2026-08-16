from mlxsim.experiments import (
    geometric_mean,
    reproduce_fig2,
    reproduce_fig3,
    reproduce_fig15_compute,
    reproduce_fig16_compute,
    reproduce_fig18,
    reproduce_fig20,
    reproduce_fig21,
    reproduce_fig22,
    reproduce_fig24,
    reproduce_fig25,
    reproduce_tables_and_fig19,
    run_h2_ablations,
)


def test_geometric_mean() -> None:
    assert geometric_mean([1.0, 4.0]) == 2.0


def test_fig22_manifest_shapes() -> None:
    result = reproduce_fig22()
    assert result["figure"] == 22
    assert len(result["sizes"]) == 8
    assert len(result["actual"]["bsmm"]) == 8
    assert len(result["actual"]["chunk_fft"]) == 8


def test_h2_ablations_do_not_improve_over_baseline() -> None:
    result = run_h2_ablations()
    for name, regression in result["cycle_regression"].items():
        if name != "baseline":
            assert regression >= 1.0


def test_fig25_calibration_surface_replays_anchor_points() -> None:
    result = reproduce_fig25()
    assert result["classification"] == "calibration-replay"
    assert result["validation_eligible"] is False
    assert result["fit_degrees_of_freedom"]["per_surface"] == 4
    assert result["fit_degrees_of_freedom"]["anchors_per_surface"] == 4
    assert all(audit["pass_10pct"] for audit in result["audit"].values())


def test_fig24_calibrated_gpu_proxy_replays_ratios() -> None:
    result = reproduce_fig24()
    assert result["classification"] == "calibration-replay-gpu-proxy"
    assert result["validation_eligible"] is False
    assert result["fit_degrees_of_freedom"]["per_operator_gpu_surface"] == 7
    assert result["fit_degrees_of_freedom"]["anchors_per_operator"] == 7
    assert all(audit["pass_10pct"] for audit in result["audit"].values())


def test_fig20_is_classified_as_a_held_out_prediction() -> None:
    result = reproduce_fig20()
    assert result["classification"] == "held-out-cross-device-prediction"
    assert result["validation_eligible"] is True
    assert len(result["actual"]["versus_dense_tcu"]["speedup"]) == 9


def test_fig21_memory_model_and_capacity_labels() -> None:
    result = reproduce_fig21()
    assert result["classification"] == "held-out-cross-device-prediction"
    assert len(result["actual"]["dense_memory_gb"]) == 5
    assert result["xavier_execution_status"][:3] == ["within-xavier-capacity"] * 3
    assert result["xavier_execution_status"][3:] == ["projected-over-xavier-capacity"] * 2


def test_fig18_cross_figure_affinity_audit_is_not_simulator_validation() -> None:
    result = reproduce_fig18()
    assert result["simulator_validation_eligible"] is False
    assert len(result["actual"]["algorithm_normalized_speedup"]) == 7


def test_reported_table_arithmetic() -> None:
    result = reproduce_tables_and_fig19()
    assert result["table2"]["audit"]["pass_10pct"]
    assert result["table5"]["audit"]["pass_10pct"]
    assert result["fig19"]["audit"]["pass_10pct"]


def test_fig2_stack_totals_reproduce_annotated_speedups() -> None:
    result = reproduce_fig2()
    assert result["native_profile_reproduced"] is False
    assert result["audit"]["fft_speedup"]["pass_10pct"]


def test_fig3_points_respect_roofline() -> None:
    result = reproduce_fig3()
    assert result["native_profile_reproduced"] is False
    assert result["physical_audit"]["all_positive"]
    assert result["physical_audit"]["all_at_or_below_roofline_with_raster_tolerance"]
    assert result["audit"]["to_qkv_implied_tensor_efficiency"]["pass_10pct"]


def test_fig15_disclosed_compute_formulas_pass_digitized_mlx_bars() -> None:
    result = reproduce_fig15_compute()
    assert result["training_reproduced"] is False
    assert result["validation_eligible"] is False
    assert result["audit"]["all_mlx_compute_bars"]["pass_10pct"]


def test_fig16_block_size_formula_passes_digitized_compute_bars() -> None:
    result = reproduce_fig16_compute()
    assert result["training_reproduced"] is False
    assert result["audit"]["all_mlx_compute_bars"]["pass_10pct"]
