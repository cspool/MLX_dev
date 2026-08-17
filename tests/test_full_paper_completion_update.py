import yaml

from scripts.audit_full_paper_completion_update import DEFAULT_CONFIG, build_audit


def test_updated_certificate_has_no_full_passes() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    report = build_audit(config)
    assert report["audit_integrity"] is True
    assert report["summary"]["status_counts"] == {
        "reproduced_within_10pct": 0,
        "attempt_rejected": 11,
        "calibration_replay_only": 0,
        "publicly_blocked": 7,
    }
    assert report["summary"]["all_paper_experiments_reproduced_within_10pct"] is False
