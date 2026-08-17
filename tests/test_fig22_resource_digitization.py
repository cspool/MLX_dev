from pathlib import Path

import pytest

from mlxsim.fig22_digitization import (
    audit_fig22_target_completion,
    derive_resource_targets,
    load_fig22_pixel_manifest,
    load_yaml,
)

ROOT = Path(__file__).resolve().parents[1]


def test_complete_resource_targets_have_expected_geometry() -> None:
    targets = derive_resource_targets(load_fig22_pixel_manifest())
    assert targets["sizes"] == [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    assert set(targets["panels"]) == {"bsmm", "chunk_fft"}
    assert targets["panels"]["bsmm"]["compute"][0] == pytest.approx(
        0.815181518
    )
    assert targets["panels"]["chunk_fft"]["xfer"][0] == pytest.approx(
        0.287128713
    )


def test_data_supply_segments_sum_to_registered_total() -> None:
    targets = derive_resource_targets(load_fig22_pixel_manifest())
    for points in targets["points"].values():
        for point in points:
            assert point["data_supply_sum"] == pytest.approx(
                point["data_supply_total"]
            )


def test_source_fill_axis_and_crosscheck_audit_pass() -> None:
    report = audit_fig22_target_completion(
        load_fig22_pixel_manifest(), verify_source=True
    )
    assert report["source_check"]["actual_dimensions"] == [621, 232]
    assert report["summary"]["numeric_value_count"] == 64
    assert report["summary"]["image_fill_evidence_pass"]
    assert report["summary"]["legacy_compute_crosscheck_pass"]
    assert report["summary"]["pass"]


def test_legacy_compute_binding_is_unchanged() -> None:
    manifest = load_fig22_pixel_manifest()
    canonical = load_yaml(ROOT / manifest["legacy_compute_crosscheck"]["source"])
    legacy = canonical["fig22_compute_utilization"]
    assert manifest["legacy_compute_crosscheck"]["bsmm"] == legacy["bsmm"]
    assert manifest["legacy_compute_crosscheck"]["chunk_fft"] == legacy["chunk_fft"]


def test_manifest_source_is_project_relative() -> None:
    assert not Path(load_fig22_pixel_manifest()["metadata"]["source"]).is_absolute()
