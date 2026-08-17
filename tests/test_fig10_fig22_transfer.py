from scripts.audit_fig10_fig22_transfer import (
    parse_log,
    relative_error,
    summarize,
)


def test_strict_metric_summary() -> None:
    assert relative_error(0.9, 1.0) <= 0.10
    summary = summarize([0.01, 0.11])
    assert summary["passing_points"] == 1
    assert not summary["all_within_10pct"]


def test_log_parser_uses_last_summary(tmp_path) -> None:
    path = tmp_path / "run.log"
    path.write_text(
        'MLX_OVERLAY_SUMMARY {"cycles":1}\n'
        'MLX_OVERLAY_SUMMARY {"cycles":2}\n'
        'MLX_SPAD_ADAPTER_SUMMARY {"requests":3}\n'
        '[single-core] sanity check passed successfully!\n'
        'Exiting @ tick 1 because exiting with last active thread context\n',
        encoding="utf-8",
    )
    parsed = parse_log(path)
    assert parsed["overlay"] == {"cycles": 2}
    assert parsed["adapter"] == {"requests": 3}
    assert parsed["sanity"] and parsed["normal_exit"]
