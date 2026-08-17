from __future__ import annotations

import pytest

from scripts.audit_dsagen_fig25_transfer import relative_error


def test_relative_error_is_symmetric_in_residual_and_target_normalized() -> None:
    assert relative_error(0.55, 0.5) == pytest.approx(0.1)
    assert relative_error(0.45, 0.5) == pytest.approx(0.1)
