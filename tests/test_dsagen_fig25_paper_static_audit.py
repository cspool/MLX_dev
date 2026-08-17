from __future__ import annotations

from scripts.audit_dsagen_fig25_paper_static import build_audit


def test_paper_static_fig25_auditor_is_callable() -> None:
    assert callable(build_audit)
