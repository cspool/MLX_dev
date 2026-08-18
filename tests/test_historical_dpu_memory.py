import yaml

from mlxsim.historical_dpu_memory import (
    invalid_relative_address_case,
    scenarios,
)
from scripts.audit_historical_dpu_memory import DEFAULT_CONFIG, build_audit


def test_historical_dpu_memory_documents() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    documents = scenarios(config)
    assert len(documents) == 6
    assert documents["non_stop_four_tiles"]["memory"]["mode"] == "non_stop"
    assert documents["baseline_four_tiles"]["memory"]["mode"] == "baseline"
    assert (
        documents["non_stop_four_tiles"]["memory"]["dma_bytes_per_cycle"] == 64
    )
    auxiliary = invalid_relative_address_case(config)
    assert "relative address" in auxiliary["expected_failure"]
    assert config["fixtures"]["dpu_2018"]["spm_bytes"] == 8 * 1024 * 1024
    assert config["fixtures"]["dpu_2022"]["data_spm_banks"] == 16
    assert config["fixtures"]["dpu_2022"]["instruction_spm_banks"] == 8


def test_historical_dpu_memory_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["summary"]["acceptance_gates_passed"] == 12
    assert report["summary"]["acceptance_gates_total"] == 12
    assert report["summary"]["main_executions"] == 36
    assert report["summary"]["auxiliary_executions"] == 4
    assert report["measurements"]["non_stop_end_to_end_cycles"] == 37
    assert report["measurements"]["baseline_end_to_end_cycles"] == 59
    assert report["measurements"]["offchip_read_bytes"] == 512
    assert report["measurements"]["offchip_write_bytes"] == 256
    assert report["patch_checks"]["h105_source_exact"]
    assert report["paper_reproduction_claim"].startswith("none_")

