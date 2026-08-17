from __future__ import annotations

from scripts.audit_mlx_pe_semantics import standalone_audit


def test_formal_paper_static_summary_has_no_inferred_hazard_stalls() -> None:
    result = standalone_audit()
    assert result["pass"] is True
    assert result["summary"]["pe_dependency_model"] == "paper_static"
    assert result["summary"]["stalls_by_reason"] == {
        "event_dependency": 4580,
        "pipeline_contention": 252,
    }
