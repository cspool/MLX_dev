from mlxsim.repeat_folding import fit_affine


def test_projection_anchor_fit() -> None:
    model = fit_affine(4, 205, 8, 405)
    assert model.intercept == 5
    assert model.slope == 50
    assert model.predict(16) == 805
