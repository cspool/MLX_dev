from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts import audit_full_paper_completion as audit

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    return yaml.safe_load(
        (PROJECT_ROOT / "configs/analysis/full_paper_completion_v1.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_json_pointer_supports_objects_lists_and_escaping() -> None:
    document = {"a/b": [{"x~y": False}]}
    assert audit.resolve_pointer(document, "/a~1b/0/x~0y") is False
    with pytest.raises(ValueError):
        audit.resolve_pointer(document, "a/b")


def test_frozen_inventory_labels_are_exact_and_ordered() -> None:
    report = audit.evaluate_inventory(load_config())
    assert report["pass"] is True
    assert len(report["actual_labels"]) == 18
    assert report["actual_labels"][0] == "Fig. 2"
    assert report["actual_labels"][-1] == "Fig. 25"


def test_all_item_status_invariants_and_counts_pass() -> None:
    report = audit.evaluate_items(load_config())
    assert report["pass"] is True
    assert report["actual_status_counts"] == {
        "publicly_blocked": 7,
        "attempt_rejected": 7,
        "reproduced_within_10pct": 1,
        "calibration_replay_only": 3,
    }


def test_status_invariant_rejects_promoting_replay_to_reproduction() -> None:
    config = load_config()
    item = next(item for item in config["items"] if item["id"] == "fig24")
    item["status"] = "reproduced_within_10pct"
    report = audit.evaluate_items(config)
    fig24 = next(entry for entry in report["items"] if entry["id"] == "fig24")
    assert fig24["checks"]["status_invariant"] is False
    assert report["pass"] is False


def test_preflight_binds_all_frozen_evidence_and_tracks_outputs() -> None:
    config = load_config()
    suite_output = PROJECT_ROOT / config["run"]["suite_output"]
    output = PROJECT_ROOT / config["run"]["output"]
    report = audit.preflight(
        config,
        require_outputs_absent=not suite_output.exists() and not output.exists(),
        require_clean=False,
    )
    assert report["checks"]["all_frozen_files"] is True
    assert report["checks"]["inventory"] is True
    assert report["checks"]["items"] is True
    assert report["actual_outputs_absent"] is (not suite_output.exists() and not output.exists())
    assert report["pass"] is True


def test_certificate_summary_never_counts_blocked_or_replay_as_pass() -> None:
    summary = audit.certificate_summary(load_config())
    assert summary["inventory_item_count"] == 18
    assert summary["reproduced_within_10pct_count"] == 1
    assert summary["not_fully_reproduced_count"] == 17
    assert summary["all_paper_experiments_reproduced_within_10pct"] is False
    assert summary["exact_mlx_author_artifact_used"] is False
