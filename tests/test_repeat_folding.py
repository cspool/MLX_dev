import pytest

from mlxsim.repeat_folding import fit_affine


def test_affine_fit_recovers_intercept_and_slope() -> None:
    model = fit_affine(2, 13, 4, 23)
    assert model.intercept == pytest.approx(3)
    assert model.slope == pytest.approx(5)
    assert model.predict(8) == pytest.approx(43)


def test_invalid_repeat_fit_is_rejected() -> None:
    with pytest.raises(ValueError):
        fit_affine(2, 10, 2, 11)
