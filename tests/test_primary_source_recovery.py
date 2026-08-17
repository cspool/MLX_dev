from __future__ import annotations

from pathlib import Path

import yaml

from scripts import recover_mlx_lineage_sources as recovery

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    return yaml.safe_load(
        (PROJECT_ROOT / "configs/analysis/mlx_primary_source_recovery_v1.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_registered_route_set_and_exact_accept_headers() -> None:
    endpoints = recovery.build_endpoints(load_config())
    assert len(endpoints) == 17
    assert len({item.key for item in endpoints}) == 17
    assert sum(item.candidate_id == "dfu_e" for item in endpoints) == 3
    assert sum(item.candidate_id == "m2_dfu" for item in endpoints) == 0
    assert next(item for item in endpoints if item.key == "ucas_html_only").accept == (
        "text/html,application/xhtml+xml"
    )


def result_for(endpoint: recovery.RecoveryEndpoint, body: bytes) -> dict:
    return {
        "endpoint": endpoint,
        "body": body,
        "metadata": {
            "transport_success": True,
            "possible_truncation": False,
            "error": None,
        },
    }


def test_substantive_publisher_abstract_qualifies_exact_candidate() -> None:
    title = "DFU-E: A Dataflow Architecture for Edge DSP and AI Applications"
    body = f"<html><h1>{title}</h1><h2>Abstract</h2>{' evidence' * 200}</html>".encode()
    endpoint = recovery.RecoveryEndpoint(
        "dfu",
        "dfu_e",
        "https://example.test/dfu",
        "html",
        "text/html",
        "publisher_abstract_page",
    )
    config = {
        "candidates": [{"id": "dfu_e", "title": title}],
    }
    texts, summaries = recovery.parse_sources(config, {"dfu": result_for(endpoint, body)})
    assert summaries["dfu"]["candidate_qualifications"]["dfu_e"] == {
        "identity": True,
        "substantive": True,
        "feature_eligible": True,
    }
    assert texts[0]["feature_eligible"] is True


def test_institutional_bibliography_is_identity_only() -> None:
    title = (
        "M2-DFU: Multi-Mode Dataflow Architecture for Adaptive and High-Efficiency Data Processing"
    )
    body = f"<html><h1>{title}</h1><h2>Abstract</h2>{' evidence' * 200}</html>".encode()
    endpoint = recovery.RecoveryEndpoint(
        "ucas",
        "institutional_bibliography",
        "https://example.test/ucas",
        "html",
        "text/html",
        "primary_institutional_record",
    )
    config = {
        "candidates": [{"id": "m2_dfu", "title": title}],
    }
    texts, summaries = recovery.parse_sources(config, {"ucas": result_for(endpoint, body)})
    qualification = summaries["ucas"]["candidate_qualifications"]["m2_dfu"]
    assert qualification["identity"] is True
    assert qualification["feature_eligible"] is False
    assert texts[0]["feature_eligible"] is False


def test_h35_preflight_binds_h34_and_tracks_output_absence() -> None:
    report = recovery.preflight(load_config())
    output = PROJECT_ROOT / load_config()["run"]["output"]
    assert report["checks"]["output_absent"] is (not output.exists())
    assert all(value for name, value in report["checks"].items() if name != "output_absent")
    assert report["pass"] is (not output.exists())
