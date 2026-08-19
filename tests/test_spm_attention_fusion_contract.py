import yaml

from scripts.audit_spm_attention_fusion_contract import DEFAULT_CONFIG, build_audit


def test_spm_attention_fusion_contract_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["paper_performance_targets_consumed"] is False
    assert report["performance_improvement_claimed"] is False
    assert report["summary"]["shapes"] == 5
    assert report["summary"]["one_kernel_shapes"] == 4
    assert report["summary"]["two_kernel_shapes"] == 1
    assert report["summary"]["current_matches"] == 4
    assert report["summary"]["current_mismatches"] == 1
    assert report["summary"]["timing_eligible_rows"] == 4
    assert report["summary"]["timing_blocked_rows"] == 1
    assert report["summary"]["blocked_sequence_lengths"] == [2048]
    assert report["summary"]["acceptance_gates_passed"] == 10


def test_spm_attention_fusion_footprints_and_timing() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    rows = {row["sequence_length"]: row for row in report["rows"]}
    assert [rows[n]["resident_footprint_mib"] for n in sorted(rows)] == [
        1.0,
        2.0,
        4.0,
        8.0,
        16.0,
    ]
    assert rows[1024]["fits_spm"]
    assert rows[1024]["patent_kernel_count"] == 1
    assert not rows[2048]["fits_spm"]
    assert rows[2048]["patent_kernel_count"] == 2
    assert rows[2048]["corrected_structured_attention_cycles"] is None
    assert rows[2048]["corrected_dense_attention_cycles"] is None
    assert len(rows[2048]["missing_fields"]) == 4
