"""Auditable derivation of Fig. 15/16 model-quality targets from frozen pixels."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIXEL_MANIFEST = PROJECT_ROOT / "artifacts/targets/quality_digitization_pixels.yaml"


def load_pixel_manifest(path: str | Path = PIXEL_MANIFEST) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _anchored_series(
    anchor_value: float,
    anchor_y: int,
    endpoint_y: list[int],
    pixels_per_unit: float,
) -> list[float]:
    return [anchor_value + (anchor_y - y) / pixels_per_unit for y in endpoint_y]


def _perplexity_from_y(y: int, axis: dict[str, int]) -> float:
    y_zero = axis["y_at_zero"]
    pixels_per_unit = (y_zero - axis["y_at_eight"]) / 8.0
    return (y_zero - y) / pixels_per_unit


def derive_quality_targets(manifest: dict[str, Any]) -> dict[str, Any]:
    fig15 = manifest["fig15"]
    pixels15 = fig15["accuracy_pixels_per_percentage_point"]

    vit = fig15["vit"]
    vit_quality: dict[str, Any] = {"methods": vit["methods"]}
    for metric in ("top1", "top5"):
        endpoints = vit["endpoint_y"][metric]
        vit_quality[f"{metric}_accuracy_pct"] = _anchored_series(
            vit["reported_original_pct"][metric], endpoints[0], endpoints, pixels15
        )
        vit_quality[f"{metric}_provenance"] = ["reported_annotation"] + [
            "digitized" for _ in endpoints[1:]
        ]

    bert = fig15["bert_squad11"]
    bert_quality: dict[str, Any] = {"modified_last_k_layers": bert["modified_last_k_layers"]}
    for metric in ("f1", "exact_match"):
        endpoints = bert["endpoint_y"][metric]
        values = _anchored_series(
            bert["reported_original_pct"][metric], endpoints[0], endpoints, pixels15
        )
        values[-1] = bert["reported_original_pct"][metric] - bert["reported_all12_loss_pct"][metric]
        bert_quality[f"{metric}_pct"] = values
        bert_quality[f"{metric}_provenance"] = [
            "reported_annotation",
            "digitized",
            "digitized",
            "digitized",
            "digitized",
            "reported_prose",
        ]

    answering_quality: dict[str, Any] = {}
    for name, case in fig15["llm_answering"].items():
        endpoints = case["endpoint_y"]
        compressed = (
            case["reported_original_pct"]
            + (endpoints["original"] - endpoints["compressed"]) / pixels15
        )
        answering_quality[name] = {
            "compression_ratio": case["compression_ratio"],
            "accuracy_pct": [case["reported_original_pct"], compressed],
            "provenance": ["reported_annotation", "digitized"],
        }

    generation_quality: dict[str, Any] = {}
    generation_axis = fig15["llm_generation"]["axis"]
    for name, case in fig15["llm_generation"].items():
        if name == "axis":
            continue
        generation_quality[name] = {
            "compression_ratios": [1.0, 0.75, 0.5],
            "perplexity": [
                case["reported_original_perplexity"],
                _perplexity_from_y(case["endpoint_y"]["s075"], generation_axis),
                _perplexity_from_y(case["endpoint_y"]["s05"], generation_axis),
            ],
            "provenance": ["reported_annotation", "digitized", "digitized"],
        }

    fig16 = manifest["fig16"]
    pixels16 = fig16["accuracy_pixels_per_percentage_point"]
    vit16 = fig16["vit"]
    vit16_quality: dict[str, Any] = {"block_sizes": fig16["block_sizes"]}
    for metric in ("top1", "top5"):
        endpoints = vit16["endpoint_y"][metric]
        vit16_quality[f"{metric}_accuracy_pct"] = _anchored_series(
            vit16["reported_b16_pct"][metric], endpoints[0], endpoints, pixels16
        )
        vit16_quality[f"{metric}_provenance"] = [
            "reported_annotation",
            "digitized",
            "digitized",
        ]

    return {
        "fig15_quality": {
            "vit": vit_quality,
            "bert_squad11": bert_quality,
            "llm_answering": answering_quality,
            "llm_generation": generation_quality,
        },
        "fig16_quality": {
            "vit": vit16_quality,
            "llama2_winogrande_512": {
                "block_sizes": fig16["block_sizes"],
                "accuracy_pct": fig16["llama2_winogrande_512"]["accuracy_pct"],
                "provenance": ["reported_annotation"] * 3,
            },
            "internlm2_wikitext103_2k": {
                "block_sizes": fig16["block_sizes"],
                "perplexity": fig16["internlm2_wikitext103_2k"]["perplexity"],
                "provenance": ["reported_annotation"] * 3,
            },
        },
    }


def _cross_check(
    name: str,
    actual: float,
    expected: float,
    tolerance: float,
    metric: str,
) -> dict[str, Any]:
    error = abs(actual - expected)
    return {
        "name": name,
        "metric": metric,
        "actual": actual,
        "expected": expected,
        "absolute_error": error,
        "tolerance": tolerance,
        "pass": error <= tolerance,
    }


def audit_quality_digitization(
    manifest: dict[str, Any], *, verify_sources: bool = False
) -> dict[str, Any]:
    source_checks: dict[str, Any] = {}
    if verify_sources:
        for figure in ("fig15", "fig16"):
            metadata = manifest["metadata"][figure]
            path = PROJECT_ROOT / metadata["source"]
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            source_checks[figure] = {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "expected_sha256": metadata["sha256"],
                "actual_sha256": actual_hash,
                "pass": actual_hash == metadata["sha256"],
            }

    fig15 = manifest["fig15"]
    pixels15 = fig15["accuracy_pixels_per_percentage_point"]
    bert = fig15["bert_squad11"]
    checks: list[dict[str, Any]] = []
    for metric in ("f1", "exact_match"):
        endpoints = bert["endpoint_y"][metric]
        raster_all12 = (
            bert["reported_original_pct"][metric] + (endpoints[0] - endpoints[-1]) / pixels15
        )
        prose_all12 = (
            bert["reported_original_pct"][metric] - bert["reported_all12_loss_pct"][metric]
        )
        checks.append(
            _cross_check(
                f"fig15_bert_all12_{metric}",
                raster_all12,
                prose_all12,
                0.15,
                "accuracy_pct",
            )
        )

    generation = fig15["llm_generation"]
    for name, case in generation.items():
        if name == "axis":
            continue
        checks.append(
            _cross_check(
                f"fig15_{name}_original",
                _perplexity_from_y(case["endpoint_y"]["original"], generation["axis"]),
                case["reported_original_perplexity"],
                0.12,
                "perplexity",
            )
        )

    fig16 = manifest["fig16"]
    llama = fig16["llama2_winogrande_512"]
    llama_from_pixels = _anchored_series(
        llama["accuracy_pct"][0],
        llama["endpoint_y"][0],
        llama["endpoint_y"],
        fig16["accuracy_pixels_per_percentage_point"],
    )
    for block_size, actual, expected in zip(
        fig16["block_sizes"], llama_from_pixels, llama["accuracy_pct"], strict=True
    ):
        checks.append(
            _cross_check(f"fig16_llama2_b{block_size}", actual, expected, 0.15, "accuracy_pct")
        )

    derived = derive_quality_targets(manifest)
    bar_count = (
        12  # Fig. 15(a): six methods times top-1/top-5.
        + 12  # Fig. 15(b): six layer counts times F1/EM.
        + 8  # Fig. 15(c): four original/compressed pairs.
        + 9  # Fig. 15(d): three settings for three tasks.
        + 6  # Fig. 16 ViT: three block sizes times top-1/top-5.
        + 3  # Fig. 16 Llama2.
        + 3  # Fig. 16 InternLM2.
    )
    source_pass = all(item["pass"] for item in source_checks.values()) if source_checks else True
    accuracy_errors = [
        item["absolute_error"] for item in checks if item["metric"] == "accuracy_pct"
    ]
    perplexity_errors = [
        item["absolute_error"] for item in checks if item["metric"] == "perplexity"
    ]
    return {
        "classification": "exploratory-raster-target-recovery",
        "validation_eligible": False,
        "source_checks": source_checks,
        "cross_checks": checks,
        "derived_targets": derived,
        "summary": {
            "visible_quality_bars": bar_count,
            "cross_check_count": len(checks),
            "max_accuracy_absolute_error_pct": max(accuracy_errors),
            "max_perplexity_absolute_error": max(perplexity_errors),
            "all_cross_checks_pass": all(item["pass"] for item in checks),
            "source_hashes_pass": source_pass,
            "pass": source_pass and all(item["pass"] for item in checks),
        },
    }
