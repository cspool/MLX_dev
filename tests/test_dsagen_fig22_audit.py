from __future__ import annotations

from scripts.audit_dsagen_fig22 import relative_error


def test_relative_error() -> None:
    assert relative_error(0.9, 0.9) == 0.0
    assert abs(relative_error(0.81, 0.9) - 0.1) < 1e-12
