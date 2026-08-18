import yaml

from scripts.audit_mlx_author_simulator_lineage import (
    DEFAULT_CONFIG,
    LINEAGE_RECORDS,
    build_audit,
)


def test_author_simulator_lineage_audit() -> None:
    report = build_audit(yaml.safe_load(DEFAULT_CONFIG.read_text()))
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert len(LINEAGE_RECORDS) == 15
    assert report["conclusions"]["simulator_framework"] == "SimICT"
    assert report["conclusions"]["exact_parent_chip"] == "unresolved"
    assert report["conclusions"]["simulator_source_code_reuse"] == "not_supported"
