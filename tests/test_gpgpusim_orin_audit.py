from __future__ import annotations

from scripts.audit_gpgpusim_orin_proxy import build_audit


def test_orin_auditor_is_callable() -> None:
    assert callable(build_audit)
