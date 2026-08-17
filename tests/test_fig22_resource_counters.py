from scripts.audit_fig22_resource_counters import (
    parse_log,
    relative_error,
    summarize_errors,
)


def test_relative_error_and_summary_use_strict_all_point_gate() -> None:
    assert relative_error(0.9, 1.0) <= 0.10
    summary = summarize_errors([0.01, 0.11])
    assert summary["passing_points"] == 1
    assert summary["total_points"] == 2
    assert not summary["all_within_10pct"]


def test_parse_log_extracts_last_counter_summary(tmp_path) -> None:
    path = tmp_path / "run.log"
    path.write_text(
        'MLX_OVERLAY_SUMMARY {"cycles":1}\n'
        'MLX_OVERLAY_SUMMARY {"cycles":2,"physical_pe_count":16}\n'
        'MLX_SPAD_ADAPTER_SUMMARY {"requests":3}\n'
        '[single-core] sanity check passed successfully!\n'
        'Exiting @ tick 1 because exiting with last active thread context\n',
        encoding="utf-8",
    )
    parsed = parse_log(path)
    assert parsed["overlay"] == {"cycles": 2, "physical_pe_count": 16}
    assert parsed["adapter"] == {"requests": 3}
    assert parsed["sanity"]
    assert parsed["normal_exit"]
    assert not parsed["watchdog_abort"]
