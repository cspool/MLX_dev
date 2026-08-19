import yaml

from scripts.audit_full_workload_coverage import DEFAULT_CONFIG, build_audit


def test_full_workload_coverage_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["summary"]["figure23_units"] == 40
    assert report["summary"]["figure19_units"] == 12
    assert report["summary"]["figure20_units"] == 8
    assert report["summary"]["composition_units"] == 2
    assert report["summary"]["executable_units"] == 62
    assert report["summary"]["lowering_replay_passes"] == 62
    assert report["summary"]["executions"] == 124
    assert report["summary"]["execution_replay_passes"] == 62
    assert report["summary"]["llama_layers"] == 32
    assert report["summary"]["fabnet_layers"] == 24
    assert report["summary"]["single_entrypoint"]
    assert report["summary"]["full_workload_coverage_complete"]
