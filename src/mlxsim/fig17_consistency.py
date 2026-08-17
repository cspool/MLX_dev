"""Cross-figure H100 timing audit for MLX Figures 3 and 17."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import yaml

from .algorithm import AttentionShape, hierarchical_bsmm_density

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def interpolate_log2_clamped(
    sequence_length: float,
    endpoint_lengths: list[int],
    endpoint_values: list[float],
) -> float:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if len(endpoint_lengths) != 2 or len(endpoint_values) != 2:
        raise ValueError("exactly two interpolation endpoints are required")
    low_n, high_n = (float(value) for value in endpoint_lengths)
    low_value, high_value = (float(value) for value in endpoint_values)
    if not 0 < low_n < high_n:
        raise ValueError("endpoint lengths must be positive and increasing")
    log_n = min(max(math.log2(sequence_length), math.log2(low_n)), math.log2(high_n))
    fraction = (log_n - math.log2(low_n)) / (math.log2(high_n) - math.log2(low_n))
    return low_value + fraction * (high_value - low_value)


def qualify_fig3_throughputs(config: dict[str, Any]) -> dict[str, Any]:
    profile_cfg = config["fig3_h100_throughput_gflops"]
    source_path = PROJECT_ROOT / profile_cfg["provenance"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    measured = dict(zip(source["point_names"], source["performance_gflops"], strict=True))
    mappings = {
        "dense_qkv": ["to_qkv_512", "to_qkv_8K"],
        "dense_attention": ["softmax_qkv_512", "softmax_qkv_8K"],
        "bsmm": ["bsmm_512", "bsmm_8K"],
        "fft": ["fft_512", "fft_8K"],
    }
    checks: list[dict[str, Any]] = []
    for component, names in mappings.items():
        configured = [float(value) for value in profile_cfg[component]]
        actual = [float(measured[name]) for name in names]
        checks.append(
            {
                "component": component,
                "point_names": names,
                "configured_gflops": configured,
                "source_gflops": actual,
                "pass": configured == actual,
            }
        )
    return {
        "path": str(source_path.relative_to(PROJECT_ROOT)),
        "checks": checks,
        "pass": all(item["pass"] for item in checks),
    }


def _throughput(config: dict[str, Any], component: str, sequence_length: float) -> float:
    profile = config["fig3_h100_throughput_gflops"]
    return interpolate_log2_clamped(
        sequence_length,
        profile["endpoint_sequence_lengths"],
        profile[component],
    )


def predict_fig17_prefill(config: dict[str, Any]) -> list[dict[str, Any]]:
    model = config["model"]
    total_layers = int(model["total_layers"])
    modified_layers = int(model["modified_layers"])
    if not 0 <= modified_layers <= total_layers:
        raise ValueError("modified_layers must be between zero and total_layers")
    block_size = int(model["block_size"])
    compression_ratio = float(model["compression_ratio"])
    density = hierarchical_bsmm_density(block_size)
    targets = load_yaml(PROJECT_ROOT / config["targets"]["source"])["fig17_h100_speedup"]
    target_values = targets[config["targets"]["series"]]
    sequence_lengths = [int(value) for value in model["sequence_lengths"]]
    if len(target_values) != len(sequence_lengths):
        raise ValueError("target and sequence-length counts differ")

    predictions: list[dict[str, Any]] = []
    for sequence_length, target in zip(sequence_lengths, target_values, strict=True):
        shape = AttentionShape(
            name=f"llama2_n{sequence_length}",
            sequence_length=sequence_length,
            hidden_size=int(model["hidden_size"]),
            query_heads=int(model["query_heads"]),
            key_value_heads=int(model["key_value_heads"]),
        )
        p_qkv = _throughput(config, "dense_qkv", sequence_length)
        p_attention = _throughput(config, "dense_attention", sequence_length)
        p_bsmm = _throughput(config, "bsmm", sequence_length)
        compressed_length = sequence_length * compression_ratio
        p_compressed_attention = _throughput(config, "dense_attention", compressed_length)

        dense_qkv_ms = shape.dense_qkv_operations / (p_qkv * 1e9) * 1e3
        dense_attention_ms = shape.dense_attention_operations / (p_attention * 1e9) * 1e3
        structured_qkv_ms = density * shape.dense_qkv_operations / (p_bsmm * 1e9) * 1e3
        structured_attention_ms = (
            compression_ratio**2
            * shape.dense_attention_operations
            / (p_compressed_attention * 1e9)
            * 1e3
        )
        dense_layer_ms = dense_qkv_ms + dense_attention_ms
        structured_layer_ms = structured_qkv_ms + structured_attention_ms
        baseline_ms = total_layers * dense_layer_ms
        predicted_ms = (
            (total_layers - modified_layers) * dense_layer_ms
            + modified_layers * structured_layer_ms
        )
        speedup = baseline_ms / predicted_ms
        target_value = float(target)
        relative_error = abs(speedup - target_value) / target_value
        predictions.append(
            {
                "sequence_length": sequence_length,
                "throughput_gflops": {
                    "dense_qkv": p_qkv,
                    "dense_attention": p_attention,
                    "bsmm": p_bsmm,
                    "compressed_attention": p_compressed_attention,
                },
                "per_layer_time_ms": {
                    "dense_qkv": dense_qkv_ms,
                    "dense_attention": dense_attention_ms,
                    "structured_qkv_optimistic": structured_qkv_ms,
                    "structured_attention_optimistic": structured_attention_ms,
                    "dense_total": dense_layer_ms,
                    "structured_total_without_fft": structured_layer_ms,
                },
                "baseline_32_layer_phase_ms": baseline_ms,
                "mixed_32_layer_phase_ms_without_fft": predicted_ms,
                "predicted_speedup": speedup,
                "target_speedup": target_value,
                "relative_error": relative_error,
                "passes_10pct_gate": relative_error
                <= float(config["targets"]["all_point_relative_error_gate"]),
                "structured_phase_slower_than_dense": structured_layer_ms > dense_layer_ms,
            }
        )
    return predictions


def audit_fig17_cross_figure(config: dict[str, Any]) -> dict[str, Any]:
    source_qualification = qualify_fig3_throughputs(config)
    predictions = predict_fig17_prefill(config)
    errors = [item["relative_error"] for item in predictions]
    all_subunity = all(item["predicted_speedup"] < 1.0 for item in predictions)
    all_targets_above_unity = all(item["target_speedup"] > 1.0 for item in predictions)
    all_points_pass = all(item["passes_10pct_gate"] for item in predictions)
    return {
        "classification": config["classification"],
        "validation_eligible": bool(config["validation_eligible"]),
        "source_qualification": source_qualification,
        "predictions": predictions,
        "interpretation": {
            "fft_time_included": False,
            "unchanged_work_included": False,
            "all_predicted_speedups_below_one": all_subunity,
            "all_targets_above_one": all_targets_above_unity,
            "identifiable_from_public_profiles": source_qualification["pass"] and all_points_pass,
        },
        "summary": {
            "point_count": len(predictions),
            "mape": sum(errors) / len(errors),
            "max_relative_error": max(errors),
            "all_points_pass": all_points_pass,
            "source_qualification_pass": source_qualification["pass"],
            "pass": source_qualification["pass"] and all_points_pass,
        },
    }
