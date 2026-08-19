#!/usr/bin/env python3
"""Repair Figure20 Attention holdout features without evaluation access."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from mlxsim.performance_service import (
    CrossFittedLogNContrastService,
    LogLinearFeatureService,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/analysis/fig20_attention_holdout_repair_v1.yaml"
)


def object_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def attention_records(document: dict[str, Any]) -> list[dict[str, Any]]:
    components = {"dense_flash_attention", "sparse_cuda_fft_attention"}
    records = document["cases"] if "cases" in document else document["trace_cases"]
    return [
        record
        for record in records
        if int(record["figure"]) == 20 and record["component"] in components
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    frozen = config["frozen_inputs"]
    base = json.loads(
        (PROJECT_ROOT / frozen["holdout_predictions"]["path"]).read_text()
    )
    endpoint = json.loads((PROJECT_ROOT / frozen["endpoint_trace"]["path"]).read_text())
    selected = json.loads((PROJECT_ROOT / frozen["selected_model"]["path"]).read_text())
    repair = config["repair"]
    sequences = [int(value) for value in repair["trace_sequences"]]
    holdouts = {int(value) for value in repair["holdout_sequences"]}

    records = [*attention_records(endpoint), *attention_records(base)]
    record_map = {record["key"]: record for record in records}
    dense_component = str(repair["dense_component"])
    sparse_component = str(repair["sparse_component"])
    contrasts: dict[int, float] = {}
    selected_records: list[dict[str, Any]] = []
    for sequence in sequences:
        dense = record_map[f"fig20-N{sequence}-{dense_component}"]
        sparse = record_map[f"fig20-N{sequence}-{sparse_component}"]
        dense_time = float(dense["timing"]["median_ms"])
        sparse_time = float(sparse["timing"]["median_ms"])
        contrasts[sequence] = 0.5 * math.log(dense_time / sparse_time)
        selected_records.extend((dense, sparse))

    contrast_service = CrossFittedLogNContrastService(
        values_by_sequence=contrasts,
        reference_sequence=int(repair["reference_sequence"]),
        model_name="fig20_attention_cross_fitted_log_n_contrast",
        target_informed=False,
        provenance="H182-endpoints+H193-target-free-traces",
    )
    parameters = {
        name: float(selected["figure20"]["parameters"][name])
        for name in repair["attention_parameters"]
    }
    attention_service = LogLinearFeatureService(
        feature_names=tuple(repair["attention_parameters"]),
        parameters=parameters,
        model_name="fig20_attention_frozen_h183_cross_fitted_feature",
        target_informed=True,
        provenance="H183-frozen-parameters+H195-cross-fit",
    )

    repaired = json.loads(json.dumps(base["predictions"]))
    repairs: list[dict[str, Any]] = []
    for item in repaired["figure20"]:
        sequence = int(item["sequence_length"])
        if item["operator"] != "attention" or sequence not in holdouts:
            continue
        fit = contrast_service.predict_excluding(sequence)
        features = {name: 0.0 for name in repair["attention_parameters"]}
        panel_index = list(repair["panels"]).index(item["panel"])
        features[
            "dense_attention_base" if panel_index == 0 else "sparse_attention_base"
        ] = 1.0
        fitted_contrast = float(fit["prediction"])
        features["attention_trace_contrast_slope"] = (
            fitted_contrast if panel_index == 0 else -fitted_contrast
        )
        old_prediction = float(item["prediction"])
        new_prediction = attention_service.predict(features)
        item["prediction"] = new_prediction
        item["raw_trace_contrast"] = contrasts[sequence]
        item["cross_fitted_trace_contrast"] = fitted_contrast
        item["cross_fit"] = fit
        item["features"] = features
        repairs.append(
            {
                "sequence_length": sequence,
                "panel": item["panel"],
                "old_prediction": old_prediction,
                "new_prediction": new_prediction,
                "raw_trace_contrast": contrasts[sequence],
                "cross_fitted_trace_contrast": fitted_contrast,
                "cross_fit": fit,
            }
        )

    parameter_object = base["parameter_object"]
    checks = {
        "trace_shapes": sorted(contrasts) == sequences,
        "trace_records": len(selected_records) == 2 * len(sequences),
        "cross_fit_count": len(repairs) == 2 * len(holdouts),
        "cross_fit_support": all(
            len(item["cross_fit"]["training_sequences"])
            == int(repair["required_cross_fit_support"])
            for item in repairs
        ),
        "cross_fit_excludes_shape": all(
            item["sequence_length"] not in item["cross_fit"]["training_sequences"]
            for item in repairs
        ),
        "positive_slopes": all(float(item["cross_fit"]["slope"]) > 0 for item in repairs),
        "parameter_object": parameter_object
        == {
            figure: selected[figure]["parameters"]
            for figure in ("figure23", "figure19", "figure20")
        },
        "parameter_hash": base["parameter_sha256"] == object_sha256(parameter_object),
        "prediction_counts": {key: len(value) for key, value in repaired.items()}
        == {"figure23": 9, "figure19": 15, "figure20": 24},
        "finite": all(
            math.isfinite(float(item["prediction"])) and float(item["prediction"]) > 0
            for values in repaired.values()
            for item in values
        ),
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "generated_without_evaluation_access": True,
        "paper_performance_targets_consumed": False,
        "parameter_object": parameter_object,
        "parameter_sha256": base["parameter_sha256"],
        "attention_parameters": parameters,
        "trace_records": selected_records,
        "raw_contrasts": {str(key): value for key, value in sorted(contrasts.items())},
        "contrast_service": contrast_service.to_dict(),
        "attention_service": attention_service.to_dict(),
        "repairs": repairs,
        "predictions": repaired,
        "checks": checks,
    }
    output = PROJECT_ROOT / config["prediction_manifest"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"repairs": len(repairs), "checks": checks}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
