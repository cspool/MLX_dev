from __future__ import annotations

from scripts import audit_dsagen_mlx_cdc_memory as audit


def test_prefixed_json_parser_uses_exact_prefix_and_last_value() -> None:
    text = """MLX_OVERLAY_SUMMARY {"cycles":5}
noise {"cycles":9}
MLX_OVERLAY_SUMMARY {"cycles":7}"""
    assert audit.parse_prefixed_json(text, "MLX_OVERLAY_SUMMARY") == {"cycles": 7}
    assert audit.parse_prefixed_json(text, "MLX_SPAD_ADAPTER_SUMMARY") is None


def test_micro_report_requires_event_overlap_and_memory_backpressure() -> None:
    report = {
        "audit_integrity": True,
        "scenario_count": 2,
        "assertion_count": 10,
        "paper_performance_targets_consumed": False,
        "scenarios": [
            {
                "id": "event_counted_cross_layer_overlap",
                "pass": True,
                "deterministic_replay": True,
                "assertions": [{"pass": True}] * 4,
                "summary": {"event_unblocked_issues_before_tag_complete": 2},
            },
            {
                "id": "memory_adapter_backpressure",
                "pass": True,
                "deterministic_replay": True,
                "assertions": [{"pass": True}] * 4,
                "summary": {
                    "external_memory_requests": 3,
                    "external_memory_completions": 3,
                    "stalls_by_reason": {"memory_queue_full": 1},
                },
            },
        ],
    }
    assert audit.evaluate_micro_report(report)["pass"] is True
    report["scenarios"][1]["summary"]["stalls_by_reason"] = {}
    assert audit.evaluate_micro_report(report)["pass"] is False


def test_dsagen_parser_requires_normal_exit_separately() -> None:
    text = """Cycles: 569
CGRA Instances: 256 -- Activity Ratio: 1
CGRA Insts / Cycle: 1024 / 569 = 1.8
[single-core] sanity check passed successfully!
Exiting @ tick 1 because exiting with last active thread context
"""
    metrics = audit.parse_dsagen_metrics(text)
    assert metrics["roi_cycles"] == 569
    assert metrics["normal_exit"] is True
    assert metrics["sanity_check_passed"] is True
