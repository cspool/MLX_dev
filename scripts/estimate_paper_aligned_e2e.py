#!/usr/bin/env python3
"""Fit H174's transparent three-parameter Figure-21 estimate."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.optimize import least_squares

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/paper_aligned_e2e_estimate_v1.yaml"


def fit_parameters(
    xavier_base: np.ndarray,
    linear_work: np.ndarray,
    attention_work: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
    initial: list[float],
) -> np.ndarray:
    def residual(parameters: np.ndarray) -> np.ndarray:
        xavier_seconds = xavier_base[indices] + parameters[0]
        mlx_seconds = (
            parameters[1] * linear_work[indices]
            + parameters[2] * attention_work[indices]
        )
        return xavier_seconds / mlx_seconds - targets[indices]

    result = least_squares(
        residual,
        np.asarray(initial, dtype=np.float64),
        bounds=(0.0, np.inf),
        ftol=1e-14,
        xtol=1e-14,
        gtol=1e-14,
        max_nfev=100000,
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    parents = {
        name: json.loads((PROJECT_ROOT / spec["path"]).read_text())
        for name, spec in config["frozen_inputs"].items()
    }
    status_inputs = {
        name for name, spec in config["frozen_inputs"].items() if "required_status" in spec
    }
    parent_checks = {
        name: parents[name]["hypothesis_status"]
        == config["frozen_inputs"][name]["required_status"]
        and parents[name]["audit_integrity"]
        is config["frozen_inputs"][name]["required_integrity"]
        for name in status_inputs
    }
    parent_checks["targets"] = parents["figure21_targets"]["summary"]["pass"]
    contract = parents["layer_contract"]["contracts"]
    projections = parents["xavier_projection"]["projection_estimates"]
    scalar = parents["xavier_scalar"]["family_estimates"]
    target_artifact = parents["figure21_targets"]["derived_targets"]
    sequences = [int(value) for value in config["workload"]["sequence_lengths"]]
    targets = np.asarray(target_artifact["speedup_over_xavier"], dtype=np.float64)
    xavier_base_values: list[float] = []
    linear_values: list[float] = []
    attention_values: list[float] = []
    feature_rows: list[dict[str, Any]] = []
    for sequence in sequences:
        shape = f"N{sequence}"
        row = contract[shape]
        xavier_projection = float(projections[shape]["xavier_seconds"])
        xavier_attention = float(scalar[shape]["dense_attention_seconds"])
        xavier_elementwise = float(scalar[shape]["elementwise_seconds"])
        xavier_base = xavier_projection + xavier_attention + xavier_elementwise
        structured_projection = sum(
            float(component["operations"])
            for name, component in row["structured_components"].items()
            if name != "attention"
        )
        dense_projection = sum(
            float(component["operations"])
            for name, component in row["dense_components"].items()
            if name != "attention"
        )
        linear_work = (
            int(config["workload"]["structured_layers"]) * structured_projection
            + int(config["workload"]["dense_layers"]) * dense_projection
            + int(config["workload"]["layers"])
            * float(row["elementwise"]["operation_count"])
        ) / 1e12
        attention_work = (
            int(config["workload"]["structured_layers"])
            * float(row["structured_components"]["attention"]["operations"])
            + int(config["workload"]["dense_layers"])
            * float(row["dense_components"]["attention"]["operations"])
        ) / 1e12
        xavier_base_values.append(xavier_base)
        linear_values.append(linear_work)
        attention_values.append(attention_work)
        feature_rows.append(
            {
                "sequence_length": sequence,
                "xavier_projection_seconds": xavier_projection,
                "xavier_attention_seconds": xavier_attention,
                "xavier_elementwise_seconds": xavier_elementwise,
                "xavier_base_seconds": xavier_base,
                "mlx_linear_work_TOP": linear_work,
                "mlx_attention_work_TOP": attention_work,
            }
        )
    xavier_base = np.asarray(xavier_base_values, dtype=np.float64)
    linear_work = np.asarray(linear_values, dtype=np.float64)
    attention_work = np.asarray(attention_values, dtype=np.float64)
    indices = np.arange(len(sequences))
    parameters = fit_parameters(
        xavier_base,
        linear_work,
        attention_work,
        targets,
        indices,
        [float(value) for value in config["model"]["initial_parameters"]],
    )
    xavier_seconds = xavier_base + parameters[0]
    mlx_seconds = parameters[1] * linear_work + parameters[2] * attention_work
    predictions = xavier_seconds / mlx_seconds
    errors = np.abs(predictions - targets) / np.abs(targets)
    rows: list[dict[str, Any]] = []
    for index, feature in enumerate(feature_rows):
        rows.append(
            {
                **feature,
                "xavier_fixed_framework_seconds": float(parameters[0]),
                "xavier_estimated_seconds": float(xavier_seconds[index]),
                "mlx_linear_estimated_seconds": float(
                    parameters[1] * linear_work[index]
                ),
                "mlx_attention_estimated_seconds": float(
                    parameters[2] * attention_work[index]
                ),
                "mlx_estimated_seconds": float(mlx_seconds[index]),
                "estimated_speedup": float(predictions[index]),
                "paper_speedup": float(targets[index]),
                "relative_error": float(errors[index]),
                "within_10pct": bool(
                    errors[index]
                    <= float(config["acceptance"]["maximum_fit_relative_error"])
                ),
                "xavier_capacity_status": (
                    "native_capacity_eligible"
                    if feature["sequence_length"] <= 512
                    else "projected_beyond_16GB_Xavier_capacity"
                ),
                "mlx_fusion_status": (
                    "two_kernel_cost_absorbed_global_model"
                    if feature["sequence_length"] == 2048
                    else "one_kernel_capacity_eligible"
                ),
            }
        )
    leave_one_out: list[dict[str, Any]] = []
    for held_index, sequence in enumerate(sequences):
        fit_indices = np.asarray(
            [index for index in range(len(sequences)) if index != held_index]
        )
        held_parameters = fit_parameters(
            xavier_base,
            linear_work,
            attention_work,
            targets,
            fit_indices,
            [float(value) for value in config["model"]["initial_parameters"]],
        )
        held_xavier = xavier_base[held_index] + held_parameters[0]
        held_mlx = (
            held_parameters[1] * linear_work[held_index]
            + held_parameters[2] * attention_work[held_index]
        )
        held_prediction = held_xavier / held_mlx
        held_error = abs(held_prediction - targets[held_index]) / abs(
            targets[held_index]
        )
        leave_one_out.append(
            {
                "sequence_length": sequence,
                "fit_indices": [sequences[index] for index in fit_indices],
                "parameters": [float(value) for value in held_parameters],
                "estimated_speedup": float(held_prediction),
                "paper_speedup": float(targets[held_index]),
                "relative_error": float(held_error),
            }
        )
    parameter_record = {
        "xavier_fixed_framework_seconds": float(parameters[0]),
        "mlx_linear_seconds_per_TOP": float(parameters[1]),
        "mlx_attention_seconds_per_TOP": float(parameters[2]),
        "mlx_linear_effective_TOP_per_second": float(1.0 / parameters[1]),
        "mlx_attention_effective_TOP_per_second": float(1.0 / parameters[2]),
        "parameter_count": len(parameters),
        "point_count": len(sequences),
        "degrees_of_freedom": len(sequences) - len(parameters),
    }
    row_checks = {
        f"N{row['sequence_length']}": all(
            math.isfinite(float(row[field])) and float(row[field]) > 0
            for field in (
                "xavier_base_seconds",
                "xavier_estimated_seconds",
                "mlx_estimated_seconds",
                "estimated_speedup",
                "paper_speedup",
            )
        )
        for row in rows
    }
    fit_mape = float(np.mean(errors))
    fit_max = float(np.max(errors))
    loo_max = max(float(item["relative_error"]) for item in leave_one_out)
    model_checks = {
        "parameters": len(parameters) == int(config["model"]["parameter_count"])
        and all(math.isfinite(float(value)) and value >= 0 for value in parameters),
        "degrees": parameter_record["degrees_of_freedom"] == 2,
        "rows": len(rows) == int(config["acceptance"]["required_rows"])
        and all(row_checks.values()),
        "decreasing": all(
            predictions[index] > predictions[index + 1]
            for index in range(len(predictions) - 1)
        ),
        "mlx_faster": all(value > 1.0 for value in predictions),
        "fit_points": all(row["within_10pct"] for row in rows),
        "fit_mape": fit_mape
        <= float(config["acceptance"]["maximum_fit_mape"]),
        "loo": len(leave_one_out) == len(sequences)
        and loo_max
        <= float(config["acceptance"]["maximum_leave_one_out_relative_error"]),
    }
    functional_checks = {
        "xavier": parents["xavier_functional"]["summary"][
            "xavier_e2e_functional_complete"
        ]
        is True,
        "mlx": parents["mlx_functional"]["summary"]["goal_complete"] is True,
    }
    fusion_checks = {
        "rows": parents["fusion_contract"]["summary"]["one_kernel_shapes"] == 4
        and parents["fusion_contract"]["summary"]["two_kernel_shapes"] == 1,
        "blocked_disclosed": parents["fusion_contract"]["summary"][
            "blocked_sequence_lengths"
        ]
        == [2048],
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    claim_checks = {
        "paper_informed": config["classification"].startswith("paper_informed"),
        "not_validation": config["validation_eligible"] is False
        and config["acceptance"]["strict_independent_validation_claimed"] is False,
        "not_exact": config["acceptance"]["exact_paper_numbers_claimed"] is False,
        "targets_consumed": True,
    }
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        sequences == [128, 256, 512, 1024, 2048]
        and int(config["workload"]["structured_layers"]) == 24
        and int(config["workload"]["dense_layers"]) == 8,
        all(value > 0 for value in xavier_base),
        model_checks["parameters"] and model_checks["degrees"],
        model_checks["rows"],
        model_checks["decreasing"] and model_checks["mlx_faster"],
        model_checks["fit_points"] and model_checks["fit_mape"],
        model_checks["loo"],
        all(functional_checks.values()) and all(fusion_checks.values()),
        all(claim_checks.values()) and all(item["pass"] for item in source_files.values()),
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "identity": sequences == [128, 256, 512, 1024, 2048],
        "features": len(feature_rows) == 5,
        "model": model_checks["parameters"] and model_checks["rows"],
        "fit_evaluated": math.isfinite(fit_mape) and math.isfinite(fit_max),
        "loo_evaluated": len(leave_one_out) == 5 and math.isfinite(loo_max),
        "functional": all(functional_checks.values()),
        "fusion": all(fusion_checks.values()),
        "claims": all(claim_checks.values()),
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
        "paper_reproduction_claim": "paper_informed_estimate_not_independent_validation",
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "parameters": parameter_record,
        "rows": rows,
        "row_checks": row_checks,
        "leave_one_out": leave_one_out,
        "model_checks": model_checks,
        "functional_checks": functional_checks,
        "fusion_checks": fusion_checks,
        "source_files": source_files,
        "claim_checks": claim_checks,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "rows": len(rows),
            "parameters": len(parameters),
            "degrees_of_freedom": parameter_record["degrees_of_freedom"],
            "fit_mape": fit_mape,
            "fit_max_relative_error": fit_max,
            "leave_one_out_max_relative_error": loo_max,
            "estimated_speedups": [float(value) for value in predictions],
            "paper_speedups": [float(value) for value in targets],
            "strictly_decreasing": model_checks["decreasing"],
            "mlx_faster_rows": int(sum(value > 1.0 for value in predictions)),
            "xavier_functional_complete": functional_checks["xavier"],
            "mlx_functional_complete": functional_checks["mlx"],
            "paper_informed_estimate_complete": supported,
            "independent_validation_claimed": False,
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
            "parameters",
            "rows",
            "leave_one_out",
            "model_checks",
            "functional_checks",
            "claim_checks",
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
