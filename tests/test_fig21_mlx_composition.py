import yaml

from scripts.audit_fig21_mlx_composition import DEFAULT_CONFIG, build_audit


def test_mlx_composition_has_five_rows_without_speedup() -> None:
    report = build_audit(yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8")))
    assert report["audit_integrity"] is True
    assert report["summary"]["shape_count"] == 5
    assert report["summary"]["mlx_composition_available"] is True
    assert report["summary"]["figure21_speedup_available"] is False
