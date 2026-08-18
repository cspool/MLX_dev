import yaml

from scripts.audit_core_architecture_claims import DEFAULT_CONFIG, build_audit


def test_core_architecture_claim_certificate() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["summary"]["primary_claims"] == 5
    assert report["summary"]["primary_claims_reproduced"] == 5
    assert report["summary"]["supporting_claims"] == 3
    assert report["summary"]["supporting_claims_reproduced"] == 3
    assert report["summary"]["minimum_primary_speedup"] >= 1.2
    assert report["summary"]["maximum_primary_speedup"] > 7.9
    assert report["summary"]["core_architecture_goal_complete"]
    assert not report["summary"]["full_figure_required"]
    assert not report["summary"]["strict_10pct_required"]
    assert report["summary"]["acceptance_gates_passed"] == 10
