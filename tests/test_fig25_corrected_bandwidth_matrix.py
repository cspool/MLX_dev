import yaml

from scripts.audit_fig25_corrected_bandwidth_matrix import (
    DEFAULT_CONFIG,
    build_audit,
    relative_error,
    within_limit,
)


def test_relative_error_is_inclusive_at_ten_percent() -> None:
    assert within_limit(relative_error(0.55, 0.5), 0.10)
    assert within_limit(relative_error(0.45, 0.5), 0.10)


def test_fixed_bandwidth_matrix_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    uniform = report["summary"]["uniform_bandwidths_passing_all_24"]
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == (
        "supported" if uniform else "rejected"
    )
    assert report["summary"]["bandwidths"] == 5
    assert report["summary"]["points_per_bandwidth"] == 24
    assert report["summary"]["matrix_points"] == 120
    assert report["summary"]["selected_mlx_bandwidth_bytes_per_cycle"] is None
    assert report["summary"]["oracle_is_diagnostic_only"] is True
    assert report["acceptance_gate_checks"]["uniform_24_of_24"] == bool(
        uniform
    )
    assert report["summary"]["acceptance_gates_passed"] in (11, 12)
    assert report["summary"]["acceptance_gates_total"] == 12
    assert all(report["parent_checks"].values())
    assert all(report["target_checks"].values())
    assert all(report["selection_checks"].values())
