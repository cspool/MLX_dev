import yaml

from scripts.audit_xavier_fft_regime import DEFAULT_CONFIG, build_audit


def test_xavier_fft_regime_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == (
        "supported" if all(report["acceptance_gates"]) else "rejected"
    )
    assert report["summary"]["new_runs"] == 4
    assert report["summary"]["holdouts"] == 2
    assert report["summary"]["full_estimates"] == 2
    assert report["summary"]["acceptance_gates_total"] == 10
