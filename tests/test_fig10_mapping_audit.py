from mlxsim.fig10_mapping import compile_fig10_mapping
from scripts.audit_fig10_mapping import parse_log, structural_checks


def test_structural_audit_accepts_source_derived_mapping() -> None:
    document, metadata = compile_fig10_mapping("bsmm", 128)
    assert all(structural_checks(document, metadata).values())


def test_log_parser_uses_last_summaries(tmp_path) -> None:
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
    assert parsed["sanity"]
    assert parsed["normal_exit"]
