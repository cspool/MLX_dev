import yaml

from mlxsim.simict_dpu_contract import historical_fixtures, semantic_scenarios
from scripts.audit_simict_dpu_contract import DEFAULT_CONFIG, build_audit


def test_simict_dpu_contract_documents() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    scenarios = semantic_scenarios()
    fixtures = historical_fixtures(config["fixtures"])
    assert len(scenarios) == 10
    assert len(fixtures) == 3
    assert scenarios["frfo_ready_age"]["pe_dependency_model"] == "dpu_frfo"
    assert fixtures["dpu_2022_4x4"]["routing"]["network_planes"] == 4
    assert fixtures["dpu_2018_8x8"]["metadata"]["source_contract"][
        "noc_planes"
    ] is None


def test_simict_dpu_contract_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["summary"]["executions"] == 78
    assert report["summary"]["acceptance_gates_passed"] == 12
    assert report["summary"]["acceptance_gates_total"] == 12
    assert report["patch_checks"]["legacy_pre_patch_exact"]
    assert report["summary"]["h52_trace_semantics_exact"]
    assert report["summary"]["gem5_enabled_disabled_569_cycles"]
    assert report["paper_reproduction_claim"].startswith("none_")
