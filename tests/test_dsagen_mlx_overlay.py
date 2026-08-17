from __future__ import annotations

from scripts import audit_dsagen_mlx_overlay as audit


def test_dsagen_metric_parser_separates_overlay_presence() -> None:
    text = """
MLX_OVERLAY_SUMMARY {"done":true}
Cycles: 569
CGRA Instances: 256 -- Activity Ratio: 1
CGRA Insts / Cycle: 1024 / 569 = 1.8
[single-core] sanity check passed successfully!
Exiting @ tick 114277000 because exiting with last active thread context
"""
    assert audit.parse_dsagen_metrics(text) == {
        "roi_cycles": 569,
        "cgra_instances": 256,
        "cgra_instructions": 1024,
        "sanity_check_passed": True,
        "normal_exit": True,
        "overlay_summary_present": True,
    }


def test_overlay_summary_parser_uses_last_summary() -> None:
    text = """MLX_OVERLAY_SUMMARY {"cycles":2,"done":false}
MLX_OVERLAY_SUMMARY {"cycles":5,"done":true}"""
    assert audit.parse_overlay_summary(text) == {"cycles": 5, "done": True}


def test_driver_report_rejects_missing_semantic_scenario() -> None:
    report = {
        "schema_version": 1,
        "audit_integrity": True,
        "scenario_count": 0,
        "assertion_count": 0,
        "paper_target_values_consumed": False,
        "scenarios": [],
    }
    evaluation = audit.evaluate_driver_report(report)
    assert evaluation["pass"] is False
    assert evaluation["checks"]["scenario_ids"] is False


def test_driver_report_accepts_registered_shape() -> None:
    ids = [
        "lower_tag_compute_contention",
        "four_pipeline_overlap",
        "active_window_bound",
        "register_raw_and_bank_pressure",
        "fu_initiation_interval",
        "greedy_skip_hop",
        "adjacent_layer_dependency",
    ]
    scenarios = []
    assertion_counts = [3, 2, 2, 2, 2, 6, 1]
    for scenario_id, count in zip(ids, assertion_counts, strict=True):
        summary = {
            "max_pipeline_issues_in_cycle": 4,
            "max_active_tags": 3,
            "skip_hops": 1,
            "link_stalls": 1,
            "stalls_by_reason": {"fu_initiation": 1},
        }
        scenarios.append(
            {
                "id": scenario_id,
                "pass": True,
                "deterministic_replay": True,
                "summary": summary,
                "assertions": [{"pass": True}] * count,
            }
        )
    report = {
        "schema_version": 1,
        "audit_integrity": True,
        "scenario_count": 7,
        "assertion_count": 25,
        "paper_target_values_consumed": False,
        "scenarios": scenarios,
    }
    assert audit.evaluate_driver_report(report)["pass"] is True
