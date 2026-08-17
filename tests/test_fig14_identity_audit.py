from __future__ import annotations

from pathlib import Path

import yaml

from scripts import audit_mlx_fig14_identity as audit

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    return yaml.safe_load(
        (PROJECT_ROOT / "configs/analysis/mlx_fig14_identity_v1.yaml").read_text(encoding="utf-8")
    )


def base_observation() -> dict:
    config = load_config()
    figure = config["local_sources"]["figure14"]
    return {
        "observation": {
            "run_id": "run_041",
            "image": {
                "path": figure["path"],
                "bytes": figure["bytes"],
                "sha256": figure["sha256"],
                "width": figure["width"],
                "height": figure["height"],
                "mode": figure["mode"],
            },
            "inspection_tool": "view_image:original",
            "inspected_at_utc": "2026-08-17T00:00:00Z",
            "neutral_description": "A small floorplan raster.",
            "clear_text_entries": [],
            "too_small_or_blurred_text": [],
            "layout_resemblance_used_for_identity": False,
        }
    }


def test_preflight_binds_exact_figure_and_tracks_frozen_artifacts() -> None:
    config = load_config()
    observation = PROJECT_ROOT / config["run"]["observation"]
    output = PROJECT_ROOT / config["run"]["output"]
    report = audit.preflight(
        config, require_observation_absent=not observation.exists()
    )
    assert report["checks"]["observation_state"] is True
    assert report["checks"]["output_absent"] is (not output.exists())
    assert all(
        value for name, value in report["checks"].items() if name != "output_absent"
    )
    assert report["pass"] is (not output.exists())
    assert report["figure14"]["actual_image"] == {
        "width": 266,
        "height": 213,
        "mode": "RGB",
        "format": "JPEG",
    }


def test_empty_clear_text_rejects_identifier_hypothesis() -> None:
    evaluation = audit.evaluate_observation(load_config(), base_observation())
    assert evaluation["pass"] is True
    assert evaluation["decision"]["pass"] is False
    assert evaluation["registered_candidate_labels"] == []


def test_generic_mlx_label_cannot_pass_identity_gate() -> None:
    observation = base_observation()
    observation["observation"]["clear_text_entries"] = [
        {"text": "MLX", "category": "project_or_family_identifier", "confidence": "clear"}
    ]
    evaluation = audit.evaluate_observation(load_config(), observation)
    assert evaluation["decision"]["clear_non_generic_identifier_count"] == 0
    assert evaluation["decision"]["pass"] is False


def test_registered_candidate_label_passes_and_is_exposed() -> None:
    observation = base_observation()
    observation["observation"]["clear_text_entries"] = [
        {"text": "DFU-E", "category": "chip_identifier", "confidence": "clear"}
    ]
    evaluation = audit.evaluate_observation(load_config(), observation)
    assert evaluation["decision"]["identifier_path"] is True
    assert evaluation["registered_candidate_labels"][0]["text"] == "DFU-E"
    assert evaluation["exact_parent_candidate_labels"][0]["text"] == "DFU-E"


def test_method_label_cannot_establish_exact_parent() -> None:
    observation = base_observation()
    observation["observation"]["clear_text_entries"] = [
        {"text": "SimICT", "category": "project_or_family_identifier", "confidence": "clear"}
    ]
    evaluation = audit.evaluate_observation(load_config(), observation)
    assert evaluation["decision"]["identifier_path"] is True
    assert evaluation["registered_candidate_labels"][0]["text"] == "SimICT"
    assert evaluation["exact_parent_candidate_labels"] == []


def test_two_numeric_labels_pass_h36_but_not_candidate_identity() -> None:
    observation = base_observation()
    observation["observation"]["clear_text_entries"] = [
        {"text": "12 nm", "category": "process_node", "confidence": "clear"},
        {"text": "1 GHz", "category": "frequency", "confidence": "clear"},
    ]
    evaluation = audit.evaluate_observation(load_config(), observation)
    assert evaluation["decision"]["numeric_path"] is True
    assert evaluation["decision"]["pass"] is True
    assert evaluation["registered_candidate_labels"] == []


def test_numeric_category_without_a_digit_fails_schema() -> None:
    observation = base_observation()
    observation["observation"]["clear_text_entries"] = [
        {"text": "unknown", "category": "frequency", "confidence": "clear"}
    ]
    evaluation = audit.evaluate_observation(load_config(), observation)
    assert evaluation["pass"] is False
    assert evaluation["numeric_parent_values"] == []
