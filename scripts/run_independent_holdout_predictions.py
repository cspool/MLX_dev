#!/usr/bin/env python3
"""Generate frozen-parameter new-shape predictions before reference access."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import torch
import yaml

from mlxsim.performance_service import LinearFeatureService, LogLinearFeatureService
from scripts.run_fig19_20_23_rtx4090_trace import (
    build_case,
    case_specs,
    gpu_snapshot,
    sampled_checksum,
    summarize_samples,
    time_callable,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/independent_holdout_validation_v1.yaml"


def digest_object(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def log_interpolate(
    sequence: int, anchor_sequences: list[int], anchor_values: list[float]
) -> float:
    x = np.log2(np.asarray(anchor_sequences, dtype=float))
    y = np.log(np.asarray(anchor_values, dtype=float))
    return float(math.exp(float(np.interp(math.log2(sequence), x, y))))


def trace_configuration(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "workloads": {
            "figure19": {
                "sequence_lengths": config["holdouts"]["figure19"]["sequence_lengths"],
                "batch": 1,
                "hidden_dimension": 1024,
                "ffn_dimension": 4096,
                "heads": 32,
                "head_dimension": 32,
                "block_size": 32,
                "trace_components": ["fft2d", "bsmm_ffn1", "bsmm_ffn2"],
            },
            "figure20": {
                "sequence_lengths": config["holdouts"]["figure20"]["sequence_lengths"],
                "batch": 1,
                "hidden_dimension": 4096,
                "ffn_dimension": 11008,
                "heads": 32,
                "head_dimension": 128,
                "block_size": 32,
                "compression_ratio": 0.5,
                "trace_components": [
                    "dense_tcu_qkv",
                    "sparse_cuda_qkv",
                    "dense_flash_attention",
                    "sparse_cuda_fft_attention",
                    "dense_tcu_ffn1",
                    "sparse_cuda_ffn1",
                    "dense_tcu_ffn2",
                    "sparse_cuda_ffn2",
                ],
            },
            "figure23": {
                "sequence_lengths": config["holdouts"]["figure23"]["sequence_lengths"],
                "batch": 8,
                "hidden_dimension": 512,
                "block_size": 32,
                "trace_components": ["fft_cmp", "bsmm"],
            },
        }
    }


def collect_traces(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, str]]:
    trace_config = trace_configuration(config)
    gpu = config["native_trace"]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu["gpu_index"])
    torch.manual_seed(int(gpu["deterministic_seed"]))
    before = gpu_snapshot(int(gpu["gpu_index"]))
    records = []
    for index, spec in enumerate(case_specs(trace_config)):
        torch.manual_seed(int(gpu["deterministic_seed"]) + index)
        function, metadata = build_case(trace_config, spec)
        samples, output = time_callable(
            function,
            warmup=int(gpu["warmup_iterations"]),
            iterations=int(gpu["timed_iterations"]),
        )
        records.append(
            {
                **spec,
                "key": f"fig{spec['figure']}-N{spec['sequence_length']}-{spec['component']}",
                "metadata": metadata,
                "timing_samples_ms": samples,
                "timing": summarize_samples(samples),
                "output_finite": bool(torch.isfinite(output).all().item()),
                "sampled_checksum": sampled_checksum(output),
            }
        )
        del output, function
        torch.cuda.empty_cache()
    after = gpu_snapshot(int(gpu["gpu_index"]))
    return records, before, after


def trace_medians(records: list[dict[str, Any]]) -> dict[str, float]:
    return {record["key"]: float(record["timing"]["median_ms"]) for record in records}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    frozen = config["frozen_inputs"]
    selected = json.loads((PROJECT_ROOT / frozen["selected_model"]["path"]).read_text())
    training_trace = json.loads((PROJECT_ROOT / frozen["training_trace"]["path"]).read_text())
    fig23_raw = json.loads((PROJECT_ROOT / frozen["figure23_raw"]["path"]).read_text())
    fig19_raw = json.loads((PROJECT_ROOT / frozen["figure19_raw"]["path"]).read_text())
    fig19_composition = json.loads(
        (PROJECT_ROOT / frozen["figure19_composition"]["path"]).read_text()
    )
    traces, gpu_before, gpu_after = collect_traces(config)
    medians = trace_medians(traces)
    training_medians = trace_medians(training_trace["cases"])
    parameter_object = {
        figure: selected[figure]["parameters"] for figure in ("figure23", "figure19", "figure20")
    }
    predictions: dict[str, list[dict[str, Any]]] = {"figure23": [], "figure19": [], "figure20": []}
    # Figure23: interpolate only raw cycle work, then apply frozen trace features.
    anchors23 = [int(value) for value in config["holdouts"]["figure23"]["reference_anchor_lengths"]]
    params23 = parameter_object["figure23"]
    knee_trace = training_medians["fig23-N2048-fft_cmp"] + training_medians[
        "fig23-N2048-bsmm"
    ]
    window = int(config["holdouts"]["figure23"]["active_window"])
    for sequence in config["holdouts"]["figure23"]["sequence_lengths"]:
        group_anchors = [fig23_raw["cycles"][f"N{n}-w{window}"] for n in anchors23]
        raw_cycles = {
            hardware: log_interpolate(
                int(sequence), anchors23, [float(item[hardware]) for item in group_anchors]
            )
            for hardware in ("baseline", "simd32_4x4", "simd8_8x8", "simd32_8x8")
        }
        combined_trace = medians[f"fig23-N{sequence}-fft_cmp"] + medians[
            f"fig23-N{sequence}-bsmm"
        ]
        underfill = max(0.0, 1.0 - int(sequence) / 2048)
        growth = max(0.0, combined_trace / knee_trace - 1.0)
        corrected = dict(raw_cycles)
        corrected["simd8_8x8"] += float(
            params23["mesh_post_knee_congestion_cycles_per_trace_ratio"]
        ) * growth
        corrected["simd32_8x8"] -= float(
            params23[f"joint_w{window}_underfill_startup_credit_cycles"]
        ) * underfill
        corrected["simd32_8x8"] += float(
            params23["joint_post_knee_congestion_cycles_per_trace_ratio"]
        ) * growth
        for series in config["holdouts"]["figure23"]["series"]:
            predictions["figure23"].append(
                {
                    "sequence_length": int(sequence),
                    "series": series,
                    "raw_baseline_cycles": raw_cycles["baseline"],
                    "raw_series_cycles": raw_cycles[series],
                    "corrected_series_cycles": corrected[series],
                    "underfill_feature": underfill,
                    "trace_growth_feature": growth,
                    "prediction": raw_cycles["baseline"] / corrected[series],
                }
            )
    # Figure19: interpolate raw simulator work; retain frozen feature service.
    anchors19 = [128, 256, 512, 1024]
    params19 = parameter_object["figure19"]
    attention_anchor = [
        float(fig19_raw["combined_full_estimates"][f"N{n}-fft2d"]["cycles"]) * 24 / 1e6
        for n in anchors19
    ]
    ffn_anchor = [
        sum(
            float(fig19_raw["combined_full_estimates"][f"N{n}-global_ffn{index}"]["cycles"])
            for index in (1, 2)
        )
        * 24
        / 1e6
        for n in anchors19
    ]
    open_anchor = [float(row["open_fabnet_latency_ms"]) for row in fig19_composition["rows"]]
    train_att_trace = [training_medians[f"fig19-N{n}-fft2d"] for n in anchors19]
    train_ffn_trace = [
        training_medians[f"fig19-N{n}-bsmm_ffn1"]
        + training_medians[f"fig19-N{n}-bsmm_ffn2"]
        for n in anchors19
    ]
    train_total_trace = [a + b for a, b in zip(train_att_trace, train_ffn_trace, strict=True)]
    att_center, ffn_center, total_center = (
        median(train_att_trace),
        median(train_ffn_trace),
        median(train_total_trace),
    )
    mlx_service = LinearFeatureService(
        feature_names=("attention_trace_launch", "ffn_trace_launch", "simulated_work", "spm_transition"),
        parameters={
            "attention_trace_launch": params19["attention_trace_launch_ms"],
            "ffn_trace_launch": params19["ffn_trace_launch_ms"],
            "simulated_work": params19["shared_simulated_work_scale"],
            "spm_transition": params19["spm_transition_ms"],
        },
        model_name="frozen_fig19_holdout",
        target_informed=True,
        provenance="H183-frozen",
    )
    fabnet_service = LinearFeatureService(
        feature_names=("trace_launch", "open_simulator_work", "spm_transition"),
        parameters={
            "trace_launch": params19["fabnet_trace_launch_ms"],
            "open_simulator_work": params19["fabnet_open_simulator_scale"],
            "spm_transition": params19["fabnet_spm_transition_ms"],
        },
        model_name="frozen_fabnet_holdout",
        target_informed=True,
        provenance="H183-frozen",
    )
    final_attention_raw = attention_anchor[-1]
    for sequence in config["holdouts"]["figure19"]["sequence_lengths"]:
        attention_raw = log_interpolate(int(sequence), anchors19, attention_anchor)
        ffn_raw = log_interpolate(int(sequence), anchors19, ffn_anchor)
        open_raw = log_interpolate(int(sequence), anchors19, open_anchor)
        attention_trace = medians[f"fig19-N{sequence}-fft2d"]
        ffn_trace = medians[f"fig19-N{sequence}-bsmm_ffn1"] + medians[
            f"fig19-N{sequence}-bsmm_ffn2"
        ]
        transitioned = int(sequence) > 512
        attention_ms = mlx_service.predict(
            {
                "attention_trace_launch": attention_trace / att_center,
                "ffn_trace_launch": 0.0,
                "simulated_work": attention_raw,
                "spm_transition": attention_raw / final_attention_raw if transitioned else 0.0,
            }
        )
        ffn_ms = mlx_service.predict(
            {
                "attention_trace_launch": 0.0,
                "ffn_trace_launch": ffn_trace / ffn_center,
                "simulated_work": ffn_raw,
                "spm_transition": ffn_raw / final_attention_raw if transitioned else 0.0,
            }
        )
        fabnet_ms = fabnet_service.predict(
            {
                "trace_launch": (attention_trace + ffn_trace) / total_center,
                "open_simulator_work": open_raw,
                "spm_transition": 1.0 if transitioned else 0.0,
            }
        )
        mlx_total = attention_ms + ffn_ms
        predictions["figure19"].extend(
            [
                {"sequence_length": int(sequence), "series": "attention_latency_ms", "prediction": attention_ms},
                {"sequence_length": int(sequence), "series": "ffn_latency_ms", "prediction": ffn_ms},
                {"sequence_length": int(sequence), "series": "fabnet_total_latency_ms", "prediction": fabnet_ms},
                {"sequence_length": int(sequence), "series": "mlx_total_latency_ms", "prediction": mlx_total},
                {"sequence_length": int(sequence), "series": "speedup", "prediction": fabnet_ms / mlx_total},
            ]
        )
    # Figure20: frozen log-linear parameters plus new trace contrast.
    params20 = parameter_object["figure20"]
    projection_names = (
        "dense_projection_base",
        "sparse_projection_base",
        "ffn1_projection_offset",
        "ffn2_projection_offset",
        "dense_bulk_scale_slope",
        "sparse_bulk_scale_slope",
        "ffn1_bulk_scale_delta",
        "ffn2_bulk_scale_delta",
    )
    attention_names = (
        "dense_attention_base",
        "sparse_attention_base",
        "attention_trace_contrast_slope",
    )
    projection_service = LogLinearFeatureService(
        projection_names,
        {name: params20[name] for name in projection_names},
        "frozen_projection_holdout",
        True,
        "H183-frozen",
    )
    attention_service = LogLinearFeatureService(
        attention_names,
        {name: params20[name] for name in attention_names},
        "frozen_attention_holdout",
        True,
        "H183-frozen",
    )
    dense_components = {
        "qkv": "dense_tcu_qkv",
        "attention": "dense_flash_attention",
        "ffn1": "dense_tcu_ffn1",
        "ffn2": "dense_tcu_ffn2",
    }
    sparse_components = {
        "qkv": "sparse_cuda_qkv",
        "attention": "sparse_cuda_fft_attention",
        "ffn1": "sparse_cuda_ffn1",
        "ffn2": "sparse_cuda_ffn2",
    }
    for sequence in config["holdouts"]["figure20"]["sequence_lengths"]:
        scale = math.log(int(sequence) / 256)
        for panel_index, panel in enumerate(config["holdouts"]["figure20"]["panels"]):
            components = dense_components if panel_index == 0 else sparse_components
            for operator in config["holdouts"]["figure20"]["operators"]:
                if operator == "attention":
                    dense_time = medians[f"fig20-N{sequence}-{dense_components[operator]}"]
                    sparse_time = medians[f"fig20-N{sequence}-{sparse_components[operator]}"]
                    center = math.sqrt(dense_time * sparse_time)
                    features = {name: 0.0 for name in attention_names}
                    features[
                        "dense_attention_base" if panel_index == 0 else "sparse_attention_base"
                    ] = 1.0
                    features["attention_trace_contrast_slope"] = math.log(
                        (dense_time if panel_index == 0 else sparse_time) / center
                    )
                    prediction = attention_service.predict(features)
                else:
                    features = {name: 0.0 for name in projection_names}
                    features[
                        "dense_projection_base" if panel_index == 0 else "sparse_projection_base"
                    ] = 1.0
                    features[
                        "dense_bulk_scale_slope" if panel_index == 0 else "sparse_bulk_scale_slope"
                    ] = scale
                    if operator in {"ffn1", "ffn2"}:
                        features[f"{operator}_projection_offset"] = 1.0
                        features[f"{operator}_bulk_scale_delta"] = scale
                    prediction = projection_service.predict(features)
                predictions["figure20"].append(
                    {
                        "sequence_length": int(sequence),
                        "panel": panel,
                        "operator": operator,
                        "trace_key": f"fig20-N{sequence}-{components[operator]}",
                        "prediction": prediction,
                    }
                )
    all_samples = [sample for record in traces for sample in record["timing_samples_ms"]]
    checks = {
        "cases": len(traces) == int(config["native_trace"]["required_cases"]),
        "samples": len(all_samples) == int(config["native_trace"]["required_samples"]),
        "positive": all(math.isfinite(value) and value > 0 for value in all_samples),
        "outputs": all(record["output_finite"] for record in traces),
        "gpu": gpu_before["name"] == config["native_trace"]["expected_name"]
        and gpu_before["uuid"] == config["native_trace"]["expected_uuid"]
        and gpu_after["uuid"] == config["native_trace"]["expected_uuid"],
        "prediction_counts": len(predictions["figure23"]) == 9
        and len(predictions["figure19"]) == 15
        and len(predictions["figure20"]) == 24,
        "finite_predictions": all(
            math.isfinite(float(item["prediction"])) and float(item["prediction"]) > 0
            for values in predictions.values()
            for item in values
        ),
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "generated_without_reference_access": True,
        "paper_performance_targets_consumed": False,
        "parameter_object": parameter_object,
        "parameter_sha256": digest_object(parameter_object),
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "trace_cases": traces,
        "predictions": predictions,
        "checks": checks,
    }
    path = PROJECT_ROOT / config["prediction_manifest"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cases": len(traces), "samples": len(all_samples), "prediction_counts": {key: len(value) for key, value in predictions.items()}, "checks": checks}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
