import yaml

from scripts.audit_fig20_matched_evidence_closure import DEFAULT_CONFIG, build_audit


def test_figure20_closure_accounts_for_all_cells() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    report = build_audit(config)
    assert report["audit_integrity"] is True
    assert report["summary"]["status_counts"] == {
        "reproduced": 0,
        "numerical_failure": 6,
        "execution_incomplete": 2,
    }
    assert report["figure20_reproduced_within_10pct"] is False
