from __future__ import annotations

from scripts.audit_dsagen_fig23 import rel_error


def test_relative_error() -> None:
    assert rel_error(4.0, 4.0) == 0
