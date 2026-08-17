import pytest

from scripts.audit_fig10_fig23_transfer import relative_error


def test_relative_error_uses_target_denominator() -> None:
    assert relative_error(3.9, 4.0) == pytest.approx(0.025)
