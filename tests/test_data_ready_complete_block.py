import pytest
import yaml

from scripts.audit_data_ready_complete_block import DEFAULT_CONFIG, build_audit
from scripts.compile_data_ready_complete_block import build_documents


def test_data_ready_complete_block_compiler() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    documents = build_documents(config)
    assert len(documents) == 16
    baseline = documents[
        "complete--single_layer_serial_spatial_baseline--enabled"
    ]
    mlx = documents["complete--multi_layer_execution_data_ready--enabled"]
    assert baseline["active_window"] == 1
    assert mlx["active_window"] == 13
    assert baseline["metadata"]["data_ready"]["event_definitions"] == 18
    assert mlx["metadata"]["data_ready"]["event_emissions"] == 24
    assert baseline["metadata"]["data_ready"]["coarse_predecessors_removed"] == 0
    assert mlx["metadata"]["data_ready"]["coarse_predecessors_removed"] == 21
    assert baseline["metadata"]["schedule_counts"] == mlx["metadata"][
        "schedule_counts"
    ]
    assert mlx["metadata"]["schedule_counts"]["boundary_events"] == 97


def test_data_ready_complete_block_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["paper_performance_targets_consumed"] is False
    assert report["goal_claim"] == "complete_one_baseline_functional_performance"
    assert report["summary"]["configs"] == 16
    assert report["summary"]["executions"] == 48
    assert report["summary"]["both_architectures_functionally_correct"]
    assert report["summary"]["same_input_and_work"]
    assert report["summary"]["clear_improvement_prefixes"] == 4
    assert report["summary"]["clear_improvement_prefix_total"] == 4
    assert report["summary"]["complete_block_clear_improvement"]
    assert report["summary"]["complete_functional_operations"] == 466
    assert report["summary"]["complete_boundary_events"] == 97
    assert report["summary"]["complete_route_hops"] == 139
    assert report["summary"]["complete_outputs"] == 8
    assert report["summary"]["baseline_complete_max_active_tags"] == 1
    assert report["summary"]["mlx_complete_max_active_tags"] == 13
    assert report["summary"]["mlx_data_ready_issues_before_tag_complete"] > 0
    assert report["summary"]["acceptance_gates_passed"] == 10
    assert report["summary"]["goal_complete"]


def test_data_ready_complete_block_speedups() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    expected = {
        "bsmm_fft": 1.625,
        "attention": 306 / 221,
        "swa": 391 / 306,
        "complete": 426 / 341,
    }
    for prefix, speedup in expected.items():
        assert report["performance"][prefix]["speedup"] == pytest.approx(speedup)
        assert report["performance"][prefix]["clear_improvement"]
        assert report["performance"][prefix]["same_instructions"]
        assert report["performance"][prefix]["same_events"]
        assert report["performance"][prefix]["same_routes"]
