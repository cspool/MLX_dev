from __future__ import annotations

from scripts.audit_dsagen_fig25_arithmetic import build_audit


def test_arithmetic_auditor_is_callable() -> None:
    assert callable(build_audit)
