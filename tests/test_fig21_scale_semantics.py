import yaml

from scripts.audit_fig21_scale_semantics import DEFAULT_CONFIG, build_audit


def test_fig21_scale_semantics_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["summary"]["h92_models"] == 45
    assert report["summary"]["h92_runs"] == 180
    assert report["summary"]["h92_max_simultaneous_pipeline_issues"] == 4
    assert report["summary"]["h92_implied_peak_gops"] == 256
    assert report["summary"]["paper_full_peak_gops"] == 1000
    assert report["summary"]["mlx_peak_correction_requirement"] == 3.90625
    assert report["summary"]["recorded_fma_per_trace_hmma"] == 4096
    assert report["summary"]["corrected_fma_per_trace_hmma"] == 256
    assert report["summary"]["xavier_cycle_correction_requirement"] == 16
    assert report["summary"]["h95_serialized_rows"] == 5
    assert report["summary"]["target_free_repair_paths"] == 3
    assert report["summary"]["active_simulator_figures_reproduced"] == 3
    assert report["summary"]["acceptance_gates_passed"] == 10
