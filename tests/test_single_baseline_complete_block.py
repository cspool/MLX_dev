import yaml

from scripts.audit_single_baseline_complete_block import DEFAULT_CONFIG, build_audit
from scripts.compile_single_baseline_complete_block import build_documents


def test_single_baseline_complete_block_compiler_pairs() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    documents = build_documents(config)
    assert len(documents) == 16
    for prefix in config["prefixes"]:
        baseline = documents[
            f"{prefix}--single_layer_serial_spatial_baseline--enabled"
        ]
        mlx = documents[f"{prefix}--multi_layer_execution--enabled"]
        assert baseline["active_window"] == 1
        assert mlx["active_window"] == 13
        assert baseline["metadata"]["schedule_counts"] == mlx["metadata"][
            "schedule_counts"
        ]
        assert baseline["functional_execution"]["memory"] == mlx[
            "functional_execution"
        ]["memory"]


def test_single_baseline_complete_block_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "rejected"
    assert report["paper_performance_targets_consumed"] is False
    assert report["summary"]["prefixes"] == 4
    assert report["summary"]["configs"] == 16
    assert report["summary"]["executions"] == 48
    assert report["summary"]["both_architectures_functionally_correct"]
    assert report["summary"]["same_input_and_work"]
    assert report["summary"]["all_prefixes_mlx_non_regression"]
    assert report["summary"]["complete_functional_operations"] == 466
    assert report["summary"]["complete_outputs"] == 8
    assert not report["summary"]["complete_block_clear_improvement"]
    assert report["summary"]["goal_complete"] is False
    assert report["summary"]["acceptance_gates_passed"] == 9


def test_single_baseline_speedup_depth_curve() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["performance"]["bsmm_fft"]["speedup"] > 1.3
    assert report["performance"]["attention"]["speedup"] > 1.2
    assert report["performance"]["swa"]["speedup"] > 1.1
    assert 1.1 < report["performance"]["complete"]["speedup"] < 1.2
    assert all(
        item["baseline_max_active_tags"] == 1
        for item in report["performance"].values()
    )
    assert report["performance"]["complete"]["mlx_max_active_tags"] > 1
