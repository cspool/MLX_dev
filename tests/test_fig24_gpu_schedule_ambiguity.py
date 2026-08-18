import yaml

from scripts.audit_fig24_gpu_schedule_ambiguity import DEFAULT_CONFIG, build_audit


def test_fig24_gpu_schedule_ambiguity_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == (
        "supported" if all(report["acceptance_gates"]) else "rejected"
    )
    assert report["summary"]["runs"] == 3
    assert report["summary"]["scalar_fma_per_run"] == 393216
    assert report["summary"]["acceptance_gates_total"] == 10
