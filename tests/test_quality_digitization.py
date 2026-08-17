from pathlib import Path

import pytest
import yaml

from mlxsim.quality_digitization import (
    audit_quality_digitization,
    derive_quality_targets,
    load_pixel_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_quality_pixels_recover_all_visible_bars() -> None:
    report = audit_quality_digitization(load_pixel_manifest())
    assert report["summary"]["visible_quality_bars"] == 53
    assert report["summary"]["cross_check_count"] == 8
    assert report["summary"]["max_accuracy_absolute_error_pct"] < 0.1
    assert report["summary"]["max_perplexity_absolute_error"] < 0.05
    assert report["summary"]["pass"] is True


def test_fig15_series_identity_and_derived_values() -> None:
    targets = derive_quality_targets(load_pixel_manifest())["fig15_quality"]
    answering = targets["llm_answering"]
    assert answering["llama2_winogrande_512"]["compression_ratio"] == 0.75
    assert answering["llama2_winogrande_512"]["accuracy_pct"][1] == pytest.approx(89.6294, abs=1e-4)
    assert answering["internlm2_adaleval_4k"]["compression_ratio"] == 0.5
    assert answering["internlm2_adaleval_4k"]["accuracy_pct"][1] == pytest.approx(35.1941, abs=1e-4)
    generation = targets["llm_generation"]
    assert generation["internlm2_wikitext2_1k"]["perplexity"] == pytest.approx(
        [8.02, 6.7153, 6.3650], abs=1e-4
    )


def test_fig16_reported_and_digitized_targets_stay_distinct() -> None:
    targets = derive_quality_targets(load_pixel_manifest())["fig16_quality"]
    assert targets["vit"]["top1_accuracy_pct"] == pytest.approx([78.4, 77.7958, 77.3792], abs=1e-4)
    assert targets["vit"]["top1_provenance"] == [
        "reported_annotation",
        "digitized",
        "digitized",
    ]
    assert targets["llama2_winogrande_512"]["accuracy_pct"] == [89.7, 89.4, 88.3]


def test_canonical_quality_targets_match_frozen_derivation() -> None:
    derived = derive_quality_targets(load_pixel_manifest())
    with (ROOT / "artifacts/targets/paper_targets.yaml").open(encoding="utf-8") as handle:
        canonical = yaml.safe_load(handle)

    assert canonical["fig15_quality"]["vit"]["top1_accuracy_pct"] == pytest.approx(
        derived["fig15_quality"]["vit"]["top1_accuracy_pct"], abs=5e-5
    )
    assert canonical["fig15_quality"]["bert_squad11"]["f1_pct"] == pytest.approx(
        derived["fig15_quality"]["bert_squad11"]["f1_pct"], abs=5e-5
    )
    assert canonical["fig16_quality"]["vit"]["top5_accuracy_pct"] == pytest.approx(
        derived["fig16_quality"]["vit"]["top5_accuracy_pct"], abs=5e-5
    )
