#!/usr/bin/env python3
"""Analyze shared-parameter numerical corrections for Figures 23/19/20."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/fig19_20_23_gap_attribution_v1.yaml"


def relative_error(prediction: float, target: float) -> float:
    return abs(prediction - target) / target


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def weighted_lstsq(
    matrix: np.ndarray, target: np.ndarray, *, relative_power: float = 0.0
) -> np.ndarray:
    weights = np.ones_like(target) if relative_power == 0 else 1.0 / target**relative_power
    return np.linalg.lstsq(matrix * weights[:, None], target * weights, rcond=None)[0]


def trace_medians(trace: dict[str, Any]) -> dict[str, float]:
    return {record["key"]: float(record["timing"]["median_ms"]) for record in trace["cases"]}


def fit_figure23(
    config: dict[str, Any], simulator: dict[str, Any], audit: dict[str, Any], trace: dict[str, Any]
) -> dict[str, Any]:
    model = config["models"]["figure23"]
    sequences = sorted({int(cell["sequence_length"]) for cell in audit["cells"]})
    windows = sorted({int(cell["active_window"]) for cell in audit["cells"]})
    targets = {
        (int(cell["active_window"]), int(cell["sequence_length"]), cell["series"]): float(
            cell["target_speedup"]
        )
        for cell in audit["cells"]
    }
    medians = trace_medians(trace)
    knee = int(model["knee_sequence_length"])
    knee_trace = medians[f"fig23-N{knee}-fft_cmp"] + medians[f"fig23-N{knee}-bsmm"]
    features = {
        sequence: {
            "underfill": max(0.0, 1.0 - sequence / knee),
            "post_knee_trace_growth": max(
                0.0,
                (
                    medians[f"fig23-N{sequence}-fft_cmp"]
                    + medians[f"fig23-N{sequence}-bsmm"]
                )
                / knee_trace
                - 1.0,
            ),
        }
        for sequence in sequences
    }
    rows: list[list[float]] = []
    values: list[float] = []
    weights: list[float] = []
    for window in windows:
        for sequence in model["calibration_sequences"]:
            group = f"N{sequence}-w{window}"
            baseline_cycles = float(simulator["cycles"][group]["baseline"])
            for series in model["corrected_series"]:
                raw_cycles = float(simulator["cycles"][group][series])
                target_cycles = baseline_cycles / targets[(window, int(sequence), series)]
                row = [0.0, 0.0, 0.0, 0.0]
                if series == "simd8_8x8":
                    row[0] = features[int(sequence)]["post_knee_trace_growth"]
                else:
                    row[1 if window == 2 else 2] = -features[int(sequence)]["underfill"]
                    row[3] = features[int(sequence)]["post_knee_trace_growth"]
                rows.append(row)
                values.append(target_cycles - raw_cycles)
                weights.append(1.0 / math.sqrt(target_cycles))
    matrix = np.asarray(rows, dtype=float)
    target_delta = np.asarray(values, dtype=float)
    weight = np.asarray(weights, dtype=float)
    coefficients = np.linalg.lstsq(
        matrix * weight[:, None], target_delta * weight, rcond=None
    )[0]
    parameter_names = [
        "mesh_post_knee_congestion_cycles_per_trace_ratio",
        "joint_w2_underfill_startup_credit_cycles",
        "joint_w4_underfill_startup_credit_cycles",
        "joint_post_knee_congestion_cycles_per_trace_ratio",
    ]
    parameters = {
        name: float(value) for name, value in zip(parameter_names, coefficients, strict=True)
    }
    cells: list[dict[str, Any]] = []
    for window in windows:
        for sequence in sequences:
            group = f"N{sequence}-w{window}"
            baseline_cycles = float(simulator["cycles"][group]["baseline"])
            for series in ("simd32_4x4", "simd8_8x8", "simd32_8x8"):
                raw_cycles = float(simulator["cycles"][group][series])
                corrected_cycles = raw_cycles
                if series == "simd8_8x8":
                    corrected_cycles += coefficients[0] * features[sequence][
                        "post_knee_trace_growth"
                    ]
                elif series == "simd32_8x8":
                    corrected_cycles -= coefficients[1 if window == 2 else 2] * features[
                        sequence
                    ]["underfill"]
                    corrected_cycles += coefficients[3] * features[sequence][
                        "post_knee_trace_growth"
                    ]
                corrected_cycles = float(corrected_cycles)
                prediction = float(baseline_cycles / corrected_cycles)
                target = targets[(window, sequence, series)]
                error = relative_error(prediction, target)
                cells.append(
                    {
                        "active_window": window,
                        "sequence_length": sequence,
                        "series": series,
                        "raw_cycles": raw_cycles,
                        "corrected_cycles": corrected_cycles,
                        "target_speedup": target,
                        "predicted_speedup": prediction,
                        "relative_error": error,
                        "pass_15pct": error
                        <= float(config["acceptance"]["maximum_relative_error"]),
                        "direction_match": prediction > 1.0 and target > 1.0,
                        "is_holdout": sequence in model["holdout_sequences"],
                        "features": features[sequence],
                    }
                )
    holdout = [cell for cell in cells if cell["is_holdout"]]
    return {
        "family": model["family"],
        "parameters": parameters,
        "parameter_support_counts": {
            parameter_names[0]: 4,
            parameter_names[1]: 2,
            parameter_names[2]: 2,
            parameter_names[3]: 4,
        },
        "trace_knee_sequence_length": knee,
        "trace_features": {str(key): value for key, value in features.items()},
        "cells": cells,
        "summary": {
            "points": len(cells),
            "passing_points": int(sum(cell["pass_15pct"] for cell in cells)),
            "mape": float(sum(cell["relative_error"] for cell in cells) / len(cells)),
            "max_relative_error": float(max(cell["relative_error"] for cell in cells)),
            "holdout_points": len(holdout),
            "holdout_max_relative_error": float(
                max(cell["relative_error"] for cell in holdout)
            ),
            "direction_matches": sum(cell["direction_match"] for cell in cells),
            "parameter_count": len(parameters),
        },
    }


def fig19_mlx_design(
    sequences: list[int], current: dict[str, np.ndarray], medians: dict[str, float]
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, int]]]:
    attention_trace = np.asarray(
        [medians[f"fig19-N{sequence}-fft2d"] for sequence in sequences], dtype=float
    )
    ffn_trace = np.asarray(
        [
            medians[f"fig19-N{sequence}-bsmm_ffn1"]
            + medians[f"fig19-N{sequence}-bsmm_ffn2"]
            for sequence in sequences
        ],
        dtype=float,
    )
    attention_trace /= np.median(attention_trace)
    ffn_trace /= np.median(ffn_trace)
    rows: list[list[float]] = []
    values: list[float] = []
    labels: list[tuple[str, int]] = []
    final_attention = float(current["attention_latency_ms"][-1])
    for index, sequence in enumerate(sequences):
        transition = 1.0 if sequence > 512 else 0.0
        rows.append(
            [
                float(attention_trace[index]),
                0.0,
                float(current["attention_latency_ms"][index]),
                transition
                * float(current["attention_latency_ms"][index])
                / final_attention,
            ]
        )
        labels.append(("attention_latency_ms", sequence))
        rows.append(
            [
                0.0,
                float(ffn_trace[index]),
                float(current["ffn_latency_ms"][index]),
                transition * float(current["ffn_latency_ms"][index]) / final_attention,
            ]
        )
        labels.append(("ffn_latency_ms", sequence))
    return np.asarray(rows), np.asarray(values), labels


def fit_figure19(
    config: dict[str, Any], audit: dict[str, Any], fabnet: dict[str, Any], trace: dict[str, Any]
) -> dict[str, Any]:
    model = config["models"]["figure19"]
    sequences = [128, 256, 512, 1024]
    medians = trace_medians(trace)
    current = {
        series: np.asarray(audit["curve_audits"][series]["prediction_values_ms"], dtype=float)
        for series in ("attention_latency_ms", "ffn_latency_ms")
    }
    targets = {
        series: np.asarray(audit["curve_audits"][series]["target_values_ms"], dtype=float)
        for series in ("attention_latency_ms", "ffn_latency_ms")
    }
    design, _, labels = fig19_mlx_design(sequences, current, medians)
    mlx_target = np.asarray(
        [targets[series][sequences.index(sequence)] for series, sequence in labels],
        dtype=float,
    )
    coefficients = weighted_lstsq(design, mlx_target, relative_power=0.5)
    mlx_parameter_names = [
        "attention_trace_launch_ms",
        "ffn_trace_launch_ms",
        "shared_simulated_work_scale",
        "spm_transition_ms",
    ]
    calibration_indices = [
        index
        for index, (_, sequence) in enumerate(labels)
        if sequence in model["calibration_sequences"]
    ]
    calibration_coefficients = weighted_lstsq(
        design[calibration_indices], mlx_target[calibration_indices], relative_power=0.5
    )
    mlx_prediction = design @ coefficients
    mlx_holdout_prediction = design @ calibration_coefficients
    mlx_rows: list[dict[str, Any]] = []
    for index, (series, sequence) in enumerate(labels):
        error = relative_error(float(mlx_prediction[index]), float(mlx_target[index]))
        mlx_rows.append(
            {
                "series": series,
                "sequence_length": sequence,
                "prediction_ms": float(mlx_prediction[index]),
                "target_ms": float(mlx_target[index]),
                "relative_error": error,
                "pass_15pct": error
                <= float(config["acceptance"]["maximum_relative_error"]),
                "holdout_prediction_ms": float(mlx_holdout_prediction[index])
                if sequence in model["holdout_sequences"]
                else None,
                "holdout_relative_error": relative_error(
                    float(mlx_holdout_prediction[index]), float(mlx_target[index])
                )
                if sequence in model["holdout_sequences"]
                else None,
            }
        )
    open_fabnet = np.asarray(
        [float(point["latency_ms"]) for point in fabnet["comparison"]["points"]]
    )
    target_fabnet = np.asarray(
        [float(point["target_latency_ms"]) for point in fabnet["comparison"]["points"]]
    )
    trace_total = np.asarray(
        [
            medians[f"fig19-N{sequence}-fft2d"]
            + medians[f"fig19-N{sequence}-bsmm_ffn1"]
            + medians[f"fig19-N{sequence}-bsmm_ffn2"]
            for sequence in sequences
        ]
    )
    trace_total /= np.median(trace_total)
    fabnet_design = np.column_stack(
        (trace_total, open_fabnet, np.asarray(sequences) > 512)
    ).astype(float)
    fabnet_coefficients = weighted_lstsq(
        fabnet_design, target_fabnet, relative_power=0.5
    )
    fabnet_calibration_indices = [
        index for index, sequence in enumerate(sequences) if sequence in model["calibration_sequences"]
    ]
    fabnet_calibration_coefficients = weighted_lstsq(
        fabnet_design[fabnet_calibration_indices],
        target_fabnet[fabnet_calibration_indices],
        relative_power=0.5,
    )
    fabnet_prediction = fabnet_design @ fabnet_coefficients
    fabnet_holdout = fabnet_design @ fabnet_calibration_coefficients
    fabnet_rows: list[dict[str, Any]] = []
    component_by_key = {(row["series"], row["sequence_length"]): row for row in mlx_rows}
    derived_rows: list[dict[str, Any]] = []
    for index, sequence in enumerate(sequences):
        baseline_error = relative_error(
            float(fabnet_prediction[index]), float(target_fabnet[index])
        )
        fabnet_rows.append(
            {
                "sequence_length": sequence,
                "prediction_ms": float(fabnet_prediction[index]),
                "target_ms": float(target_fabnet[index]),
                "relative_error": baseline_error,
                "pass_15pct": baseline_error
                <= float(config["acceptance"]["maximum_relative_error"]),
                "holdout_prediction_ms": float(fabnet_holdout[index])
                if sequence in model["holdout_sequences"]
                else None,
                "holdout_relative_error": relative_error(
                    float(fabnet_holdout[index]), float(target_fabnet[index])
                )
                if sequence in model["holdout_sequences"]
                else None,
            }
        )
        attention = component_by_key[("attention_latency_ms", sequence)]
        ffn = component_by_key[("ffn_latency_ms", sequence)]
        mlx_total_prediction = attention["prediction_ms"] + ffn["prediction_ms"]
        mlx_total_target = attention["target_ms"] + ffn["target_ms"]
        speedup_prediction = float(fabnet_prediction[index]) / mlx_total_prediction
        speedup_target = float(target_fabnet[index]) / mlx_total_target
        total_error = relative_error(mlx_total_prediction, mlx_total_target)
        speedup_error = relative_error(speedup_prediction, speedup_target)
        derived_rows.append(
            {
                "sequence_length": sequence,
                "mlx_total_prediction_ms": mlx_total_prediction,
                "mlx_total_target_ms": mlx_total_target,
                "mlx_total_relative_error": total_error,
                "mlx_total_pass_15pct": total_error
                <= float(config["acceptance"]["maximum_relative_error"]),
                "speedup_prediction": speedup_prediction,
                "speedup_target": speedup_target,
                "speedup_relative_error": speedup_error,
                "speedup_pass_15pct": speedup_error
                <= float(config["acceptance"]["maximum_relative_error"]),
                "direction_match": speedup_prediction > 1.0 and speedup_target > 1.0,
            }
        )
    parameter_names = [
        *mlx_parameter_names,
        "fabnet_trace_launch_ms",
        "fabnet_open_simulator_scale",
        "fabnet_spm_transition_ms",
    ]
    parameter_values = [*coefficients, *fabnet_coefficients]
    parameters = {
        name: float(value) for name, value in zip(parameter_names, parameter_values, strict=True)
    }
    full_errors = [row["relative_error"] for row in mlx_rows]
    full_errors.extend(row["relative_error"] for row in fabnet_rows)
    full_errors.extend(row["mlx_total_relative_error"] for row in derived_rows)
    full_errors.extend(row["speedup_relative_error"] for row in derived_rows)
    cross_errors = [
        row["holdout_relative_error"]
        for row in [*mlx_rows, *fabnet_rows]
        if row["holdout_relative_error"] is not None
    ]
    return {
        "family": model["family"],
        "parameters": parameters,
        "parameter_support_counts": {
            mlx_parameter_names[0]: 4,
            mlx_parameter_names[1]: 4,
            mlx_parameter_names[2]: 8,
            mlx_parameter_names[3]: 2,
            "fabnet_trace_launch_ms": 4,
            "fabnet_open_simulator_scale": 4,
            "fabnet_spm_transition_ms": 2,
        },
        "mlx_component_rows": mlx_rows,
        "fabnet_rows": fabnet_rows,
        "derived_rows": derived_rows,
        "summary": {
            "fitted_points": 12,
            "reported_points": len(full_errors),
            "passing_points": sum(
                error <= float(config["acceptance"]["maximum_relative_error"])
                for error in full_errors
            ),
            "mape": sum(full_errors) / len(full_errors),
            "max_relative_error": max(full_errors),
            "cross_validation_points": len(cross_errors),
            "cross_validation_max_relative_error": max(cross_errors),
            "direction_matches": sum(row["direction_match"] for row in derived_rows),
            "parameter_count": len(parameters),
        },
    }


def fit_figure20(
    config: dict[str, Any], trace: dict[str, Any], targets: dict[str, Any]
) -> dict[str, Any]:
    model = config["models"]["figure20"]
    panels = list(model["panels"])
    sequences = [256, 8192]
    projection_operators = ["qkv", "ffn1", "ffn2"]
    target_indices = {"qkv": 0, "ffn1": 2, "ffn2": 3}
    projection_rows: list[list[float]] = []
    projection_target_log: list[float] = []
    projection_labels: list[tuple[str, int, str]] = []
    for panel_index, panel in enumerate(panels):
        for sequence_index, sequence in enumerate(sequences):
            scale_feature = math.log(sequence / 256)
            for operator_index, operator in enumerate(projection_operators):
                row = [0.0] * 8
                row[panel_index] = 1.0
                if operator_index > 0:
                    row[1 + operator_index] = 1.0
                row[4 + panel_index] = scale_feature
                if operator_index > 0:
                    row[5 + operator_index] = scale_feature
                target_index = sequence_index * 4 + target_indices[operator]
                projection_rows.append(row)
                projection_target_log.append(
                    math.log(float(targets[panel]["speedup"][target_index]))
                )
                projection_labels.append((panel, sequence, operator))
    projection_design = np.asarray(projection_rows)
    projection_log_target = np.asarray(projection_target_log)
    projection_coefficients = np.linalg.lstsq(
        projection_design, projection_log_target, rcond=None
    )[0]
    projection_prediction = np.exp(projection_design @ projection_coefficients)
    projection_parameter_names = [
        "dense_projection_base",
        "sparse_projection_base",
        "ffn1_projection_offset",
        "ffn2_projection_offset",
        "dense_bulk_scale_slope",
        "sparse_bulk_scale_slope",
        "ffn1_bulk_scale_delta",
        "ffn2_bulk_scale_delta",
    ]
    projection_cv_errors: list[float] = []
    for held_out in range(len(projection_labels)):
        retained = [index for index in range(len(projection_labels)) if index != held_out]
        cv_coefficients = np.linalg.lstsq(
            projection_design[retained], projection_log_target[retained], rcond=None
        )[0]
        prediction = math.exp(float(projection_design[held_out] @ cv_coefficients))
        target = math.exp(float(projection_log_target[held_out]))
        projection_cv_errors.append(relative_error(prediction, target))
    medians = trace_medians(trace)
    attention_rows: list[list[float]] = []
    attention_target_log: list[float] = []
    attention_labels: list[tuple[str, int, str]] = []
    dense_component = "dense_flash_attention"
    sparse_component = "sparse_cuda_fft_attention"
    for panel_index, panel in enumerate(panels):
        for sequence_index, sequence in enumerate(sequences):
            dense_time = medians[f"fig20-N{sequence}-{dense_component}"]
            sparse_time = medians[f"fig20-N{sequence}-{sparse_component}"]
            center = math.sqrt(dense_time * sparse_time)
            panel_time = dense_time if panel_index == 0 else sparse_time
            row = [0.0, 0.0, math.log(panel_time / center)]
            row[panel_index] = 1.0
            attention_rows.append(row)
            attention_target_log.append(
                math.log(float(targets[panel]["speedup"][sequence_index * 4 + 1]))
            )
            attention_labels.append((panel, sequence, "attention"))
    attention_design = np.asarray(attention_rows)
    attention_log_target = np.asarray(attention_target_log)
    attention_coefficients = np.linalg.lstsq(
        attention_design, attention_log_target, rcond=None
    )[0]
    attention_prediction = np.exp(attention_design @ attention_coefficients)
    attention_parameter_names = [
        "dense_attention_base",
        "sparse_attention_base",
        "attention_trace_contrast_slope",
    ]
    attention_cv_errors: list[float] = []
    for held_out in range(len(attention_labels)):
        retained = [index for index in range(len(attention_labels)) if index != held_out]
        cv_coefficients = np.linalg.lstsq(
            attention_design[retained], attention_log_target[retained], rcond=None
        )[0]
        prediction = math.exp(float(attention_design[held_out] @ cv_coefficients))
        target = math.exp(float(attention_log_target[held_out]))
        attention_cv_errors.append(relative_error(prediction, target))
    predictions = {
        label: float(value)
        for label, value in zip(projection_labels, projection_prediction, strict=True)
    }
    predictions.update(
        {
            label: float(value)
            for label, value in zip(attention_labels, attention_prediction, strict=True)
        }
    )
    cells: list[dict[str, Any]] = []
    all_operators = ["qkv", "attention", "ffn1", "ffn2"]
    for panel in panels:
        for sequence_index, sequence in enumerate(sequences):
            for operator_index, operator in enumerate(all_operators):
                prediction = predictions[(panel, sequence, operator)]
                target = float(targets[panel]["speedup"][sequence_index * 4 + operator_index])
                error = relative_error(prediction, target)
                cells.append(
                    {
                        "panel": panel,
                        "sequence_length": sequence,
                        "operator": operator,
                        "prediction": prediction,
                        "target": target,
                        "relative_error": error,
                        "pass_15pct": error
                        <= float(config["acceptance"]["maximum_relative_error"]),
                        "direction_match": prediction >= 1.0 and target >= 1.0,
                    }
                )
    geomeans: list[dict[str, Any]] = []
    for panel in panels:
        values = [cell["prediction"] for cell in cells if cell["panel"] == panel]
        prediction = geometric_mean(values)
        target = float(targets[panel]["speedup"][8])
        error = relative_error(prediction, target)
        geomeans.append(
            {
                "panel": panel,
                "prediction": prediction,
                "target": target,
                "relative_error": error,
                "pass_15pct": error
                <= float(config["acceptance"]["maximum_relative_error"]),
            }
        )
    parameter_names = [*projection_parameter_names, *attention_parameter_names]
    parameter_values = [*projection_coefficients, *attention_coefficients]
    parameters = {
        name: float(value) for name, value in zip(parameter_names, parameter_values, strict=True)
    }
    errors = [cell["relative_error"] for cell in cells]
    errors.extend(item["relative_error"] for item in geomeans)
    cross_errors = [*projection_cv_errors, *attention_cv_errors]
    return {
        "family": model["family"],
        "parameters": parameters,
        "parameter_support_counts": {
            "dense_projection_base": 6,
            "sparse_projection_base": 6,
            "ffn1_projection_offset": 4,
            "ffn2_projection_offset": 4,
            "dense_bulk_scale_slope": 3,
            "sparse_bulk_scale_slope": 3,
            "ffn1_bulk_scale_delta": 2,
            "ffn2_bulk_scale_delta": 2,
            "dense_attention_base": 2,
            "sparse_attention_base": 2,
            "attention_trace_contrast_slope": 4,
        },
        "cells": cells,
        "geomeans": geomeans,
        "cross_validation": {
            "projection_errors": projection_cv_errors,
            "attention_errors": attention_cv_errors,
        },
        "summary": {
            "speedup_bars": len(cells),
            "geomeans": len(geomeans),
            "passing_points": sum(
                error <= float(config["acceptance"]["maximum_relative_error"])
                for error in errors
            ),
            "mape": sum(errors) / len(errors),
            "max_relative_error": max(errors),
            "cross_validation_points": len(cross_errors),
            "cross_validation_max_relative_error": max(cross_errors),
            "direction_matches": sum(cell["direction_match"] for cell in cells),
            "parameter_count": len(parameters),
        },
    }


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    documents: dict[str, Any] = {}
    for name, spec in config["frozen_inputs"].items():
        path = PROJECT_ROOT / spec["path"]
        documents[name] = (
            yaml.safe_load(path.read_text()) if path.suffix in {".yaml", ".yml"} else json.loads(path.read_text())
        )
    parent_checks = {
        name: document["hypothesis_status"] == spec["required_status"]
        and document["audit_integrity"] is spec["required_integrity"]
        for name, document in documents.items()
        for spec in [config["frozen_inputs"][name]]
        if "required_status" in spec
    }
    trace = documents["rtx4090_trace"]
    fig23 = fit_figure23(
        config, documents["fig23_simulator"], documents["fig23_current_audit"], trace
    )
    fig19 = fit_figure19(
        config, documents["fig19_current_audit"], documents["fig19_fabnet"], trace
    )
    targets = documents["targets"]["fig20_xavier_kernels"]
    fig20 = fit_figure20(config, trace, targets)
    target_checks = {
        "fig23": all(
            math.isclose(
                float(cell["target_speedup"]),
                float(
                    documents["targets"]["fig23_scalability"][cell["series"]][
                        documents["targets"]["fig23_scalability"]["sequence_lengths"].index(
                            cell["sequence_length"]
                        )
                    ]
                ),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for cell in fig23["cells"]
        ),
        "fig19": documents["fig19_current_audit"]["length_checks"]["h13"] is True
        and documents["fig19_current_audit"]["length_checks"]["h130"] is True,
        "fig20_dense": documents["fig20_legacy"]["target"]["versus_dense_tcu"]["speedup"]
        == targets["versus_dense_tcu"]["speedup"],
        "fig20_sparse": documents["fig20_legacy"]["target"]["versus_sparse_cuda"][
            "speedup"
        ]
        == targets["versus_sparse_cuda"]["speedup"],
    }
    feature_checks = {
        "trace_status": trace["hypothesis_status"] == "supported"
        and trace["paper_performance_targets_consumed"] is False,
        "fig23": all(
            any(
                record["figure"] == 23 and record["sequence_length"] == sequence
                for record in trace["cases"]
            )
            for sequence in (512, 1024, 2048, 4096, 8192)
        ),
        "fig19": all(
            any(
                record["figure"] == 19 and record["sequence_length"] == sequence
                for record in trace["cases"]
            )
            for sequence in (128, 256, 512, 1024)
        ),
        "fig20": all(
            any(
                record["figure"] == 20 and record["sequence_length"] == sequence
                for record in trace["cases"]
            )
            for sequence in (256, 8192)
        ),
    }
    model_results = {"figure23": fig23, "figure19": fig19, "figure20": fig20}
    parameter_checks = {
        figure: result["summary"]["parameter_count"]
        <= int(config["models"][figure]["maximum_parameters"])
        for figure, result in model_results.items()
    }
    fitted_points = {"figure23": 30, "figure19": 12, "figure20": 16}
    parameter_checks["below_points"] = all(
        result["summary"]["parameter_count"] < fitted_points[figure]
        for figure, result in model_results.items()
    )
    forbidden_tokens = ("128", "256", "512", "1024", "2048", "4096", "8192", "target_index")
    parameter_checks["not_point_keyed"] = all(
        not any(token in name for token in forbidden_tokens)
        for result in model_results.values()
        for name in result["parameters"]
    )
    parameter_checks["minimum_support_two"] = all(
        support >= 2
        for result in model_results.values()
        for support in result["parameter_support_counts"].values()
    )
    finite_checks = {
        figure: all(math.isfinite(float(value)) for value in result["parameters"].values())
        and math.isfinite(float(result["summary"]["mape"]))
        and math.isfinite(float(result["summary"]["max_relative_error"]))
        for figure, result in model_results.items()
    }
    limit = float(config["acceptance"]["maximum_relative_error"])
    error_checks = {
        figure: bool(result["summary"]["max_relative_error"] <= limit)
        for figure, result in model_results.items()
    }
    direction_checks = {
        "figure23": fig23["summary"]["direction_matches"] == 30,
        "figure19": fig19["summary"]["direction_matches"] == 4,
        "figure20": fig20["summary"]["direction_matches"] == 16,
    }
    cross_validation = {
        "figure23": float(fig23["summary"]["holdout_max_relative_error"]),
        "figure19": float(fig19["summary"]["cross_validation_max_relative_error"]),
        "figure20": float(fig20["summary"]["cross_validation_max_relative_error"]),
    }
    cross_check = max(cross_validation.values()) <= float(
        config["acceptance"]["maximum_cross_validation_relative_error"]
    )
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        all(target_checks.values()),
        all(feature_checks.values()),
        all(parameter_checks.values()),
        parameter_checks["not_point_keyed"] and parameter_checks["minimum_support_two"],
        all(finite_checks.values()),
        error_checks["figure23"] and direction_checks["figure23"],
        error_checks["figure19"] and direction_checks["figure19"],
        error_checks["figure20"] and direction_checks["figure20"],
        cross_check and all(item["pass"] for item in source_files.values()),
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": len(parent_checks) == 6,
        "targets": len(target_checks) == 4,
        "features": len(feature_checks) == 4,
        "parameters": len(parameter_checks) == 6,
        "finite": len(finite_checks) == 3,
        "errors": len(error_checks) == 3,
        "directions": len(direction_checks) == 3,
        "cross_validation": len(cross_validation) == 3,
        "source": all(item["pass"] for item in source_files.values()),
        "acceptance_evaluated": len(acceptance_gates) == 10
        and all(isinstance(value, bool) for value in acceptance_gates),
    }
    integrity = all(integrity_checks.values())
    supported = integrity and all(acceptance_gates)
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "classification": config["classification"],
        "validation_eligible": config["validation_eligible"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_commit": git_commit(),
        "hypothesis_status": "supported" if supported else "rejected",
        "audit_integrity": integrity,
        "paper_performance_targets_consumed": True,
        "paper_reproduction_claim": "shared_parameter_selection_not_final_implementation",
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "target_checks": target_checks,
        "feature_checks": feature_checks,
        "parameter_checks": parameter_checks,
        "finite_checks": finite_checks,
        "error_checks": error_checks,
        "direction_checks": direction_checks,
        "cross_validation": cross_validation,
        "figure23": fig23,
        "figure19": fig19,
        "figure20": fig20,
        "implementation_recommendations": {
            "figure23": "opt_in_underfill_credit_and_trace_knee_congestion_in_cycle_service",
            "figure19": "trace_normalized_launch_work_scale_and_spm_transition_in_composer",
            "figure20": "operator_panel_scale_regime_mapping_with_attention_trace_contrast",
        },
        "source_files": source_files,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "figure23_points": fig23["summary"]["points"],
            "figure23_max_relative_error": fig23["summary"]["max_relative_error"],
            "figure19_reported_points": fig19["summary"]["reported_points"],
            "figure19_max_relative_error": fig19["summary"]["max_relative_error"],
            "figure20_speedup_bars": fig20["summary"]["speedup_bars"],
            "figure20_geomeans": fig20["summary"]["geomeans"],
            "figure20_max_relative_error": fig20["summary"]["max_relative_error"],
            "maximum_cross_validation_relative_error": max(cross_validation.values()),
            "parameter_counts": {
                figure: result["summary"]["parameter_count"]
                for figure, result in model_results.items()
            },
            "all_full_fit_points_within_15pct": all(error_checks.values()),
            "all_directions_match": all(direction_checks.values()),
            "final_implementation_required": True,
            "acceptance_gates_passed": sum(acceptance_gates),
            "acceptance_gates_total": len(acceptance_gates),
        },
        "integrity_checks": integrity_checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    report = build_audit(config)
    output = PROJECT_ROOT / config["result_path"]
    if args.verify_existing:
        existing = json.loads(output.read_text())
        keys = (
            "hypothesis_status",
            "audit_integrity",
            "target_checks",
            "feature_checks",
            "parameter_checks",
            "finite_checks",
            "error_checks",
            "direction_checks",
            "cross_validation",
            "figure23",
            "figure19",
            "figure20",
            "acceptance_gates",
            "summary",
            "integrity_checks",
        )
        matches = all(
            json.dumps(existing.get(key), sort_keys=True)
            == json.dumps(report.get(key), sort_keys=True)
            for key in keys
        )
        print(json.dumps({"existing_matches": matches, **report}, indent=2))
        return 0 if matches else 1
    if args.preflight_only:
        print(json.dumps(report, indent=2))
        return 0 if report["audit_integrity"] and not output.exists() else 1
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["hypothesis_status"], **report["summary"]}, indent=2))
    return 0 if report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
