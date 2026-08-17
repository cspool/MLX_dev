from pathlib import Path

import pytest
import yaml

from mlxsim.fig17_digitization import (
    audit_fig17_digitization,
    derive_fig17_targets,
    load_pixel_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def test_fig17_derives_all_twenty_bars_in_visual_series_order() -> None:
    manifest = load_pixel_manifest()
    targets = derive_fig17_targets(manifest)
    assert targets["sequence_lengths"] == [512, 1024, 2048, 4096, 8192]
    assert targets["prefill_eager"] == pytest.approx(
        [1.140350877, 1.368421053, 2.105263158, 2.271929825, 2.728070175], abs=1e-9
    )
    assert targets["prefill_fa"] == pytest.approx(
        [0.991228070, 1.017543860, 1.087719298, 1.350877193, 1.649122807], abs=1e-9
    )
    assert targets["decode_eager"] == pytest.approx(
        [1.5, 1.675438596, 1.798245614, 1.921052632, 1.938596491], abs=1e-9
    )
    assert targets["decode_fa"] == pytest.approx(
        [1.447368421, 1.631578947, 1.780701754, 1.885964912, 1.921052632], abs=1e-9
    )


def test_fig17_visual_mapping_reconciles_prose() -> None:
    report = audit_fig17_digitization(load_pixel_manifest())
    checks = {item["name"]: item for item in report["reported_cross_checks"]}
    assert checks["prefill_fa_max"]["actual"] == pytest.approx(1.649122807, abs=1e-9)
    assert checks["decode_min"]["actual"] == pytest.approx(1.447368421, abs=1e-9)
    assert checks["decode_max"]["actual"] == pytest.approx(1.938596491, abs=1e-9)
    assert all(item["pass"] for item in checks.values())


def test_fig17_source_axis_and_canonical_targets_pass() -> None:
    report = audit_fig17_digitization(load_pixel_manifest(), verify_source=True)
    assert report["source_check"]["actual_dimensions"] == [562, 200]
    assert report["axis_check"]["intervals_pixels"] == [57.0, 57.0]
    assert report["summary"]["visible_bars"] == 20
    assert report["summary"]["reported_cross_check_count"] == 4
    assert report["summary"]["pass"] is True


def test_fig17_canonical_middle_series_are_not_swapped() -> None:
    with (ROOT / "artifacts/targets/paper_targets.yaml").open(encoding="utf-8") as handle:
        canonical = yaml.safe_load(handle)["fig17_h100_speedup"]
    assert canonical["prefill_fa"][-1] == pytest.approx(1.649122807, abs=1e-9)
    assert canonical["decode_eager"][-1] == pytest.approx(1.938596491, abs=1e-9)
