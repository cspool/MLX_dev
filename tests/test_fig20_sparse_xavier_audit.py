from __future__ import annotations

from scripts.audit_fig20_sparse_xavier import build_audit


def test_fig20_sparse_auditor_is_callable() -> None:
    assert callable(build_audit)
