from __future__ import annotations

from scripts.audit_gpgpusim_xavier_proxy import build_audit


def test_xavier_auditor_is_callable() -> None:
    assert callable(build_audit)
