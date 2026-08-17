from __future__ import annotations

from scripts.audit_fig24_cross_simulator import build_audit


def test_cross_simulator_auditor_is_callable() -> None:
    assert callable(build_audit)
