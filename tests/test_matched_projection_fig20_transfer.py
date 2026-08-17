import yaml

from scripts.audit_matched_projection_fig20_transfer import (
    DEFAULT_CONFIG,
    build_audit,
)


def test_matched_projection_transfer_has_complete_partition() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    report = build_audit(config)
    assert report["audit_integrity"] is True
    assert report["summary"]["covered_cells"] == 6
    assert report["summary"]["cells_within_10pct"] == 0
    assert report["hypothesis_status"] == "rejected"
