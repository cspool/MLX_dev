from mlxsim.experiments import geometric_mean, reproduce_fig22, run_h2_ablations


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
