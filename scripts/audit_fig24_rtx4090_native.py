#!/usr/bin/env python3
"""Audit H177 native RTX4090 Figure-24 replacement measurements."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig24_rtx4090_native_v1.yaml"


def service_key(comparison: dict[str, Any]) -> str:
    operator = comparison["operator"]["name"]
    if operator == "fft_cmp":
        return f"fft-s{int(comparison['actual']['stage_count'])}"
    if operator == "qkv_bsmm":
        return "bsmm-s4"
    if operator == "qkv_bsmm_b32":
        return "bsmm-s5"
    if operator == "qkv_bsmm_b64":
        return "bsmm-s6"
    if operator == "swa_w128_q32":
        return "swa-w128"
    if operator == "swa_w256_q64":
        return "swa-w256"
    raise ValueError(operator)


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    identity = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["work_identity"]["path"]).read_text()
    )
    mlx = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["mlx_full_work"]["path"]).read_text()
    )
    parent_checks = {
        "identity": identity["hypothesis_status"]
        == config["frozen_inputs"]["work_identity"]["required_status"]
        and identity["audit_integrity"]
        is config["frozen_inputs"]["work_identity"]["required_integrity"],
        "mlx": mlx["hypothesis_status"]
        == config["frozen_inputs"]["mlx_full_work"]["required_status"]
        and mlx["audit_integrity"]
        is config["frozen_inputs"]["mlx_full_work"]["required_integrity"],
        "full_work": mlx["summary"]["full_work_passing_paths"]
        == mlx["summary"]["path_count"]
        == 48,
        "target_free": identity["paper_performance_targets_consumed"] is False
        and mlx["paper_performance_targets_consumed"] is False,
    }
    output_root = PROJECT_ROOT / config["output_root"]
    manifest_path = output_root / "fig24-rtx4090-native-run-manifest.json"
    manifest_file = qualify(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    gpu_checks = {
        "name": manifest["gpu_before"]["name"] == config["gpu"]["expected_name"],
        "compute": manifest["gpu_before"]["compute_cap"]
        == config["gpu"]["expected_compute_capability"],
        "index": int(manifest["gpu_before"]["index"]) == int(config["gpu"]["index"]),
        "identity_stable": manifest["gpu_before"]["uuid"]
        == manifest["gpu_after"]["uuid"],
    }
    correctness_checks = {
        key: record["pass"]
        and record["summary"]["verify"] is True
        and float(record["summary"]["maximum_absolute_error"])
        <= float(config["acceptance"]["maximum_correctness_absolute_error"])
        and qualify(PROJECT_ROOT / record["log"]["path"], record["log"])["pass"]
        for key, record in manifest["correctness"].items()
    }
    fit_counts = [int(value) for value in config["service_grid"]["fit_counts"]]
    holdout = int(config["service_grid"]["holdout_counts"][0])
    models: dict[str, Any] = {}
    holdout_checks: dict[str, bool] = {}
    timing_artifact_checks: dict[str, bool] = {}
    for key, records in manifest["timings"].items():
        first = records[str(fit_counts[0])]
        second = records[str(fit_counts[1])]
        held = records[str(holdout)]
        x1 = float(first["summary"]["fma_count"])
        x2 = float(second["summary"]["fma_count"])
        y1 = float(first["summary"]["average_ms"])
        y2 = float(second["summary"]["average_ms"])
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        held_work = float(held["summary"]["fma_count"])
        prediction = intercept + slope * held_work
        actual = float(held["summary"]["average_ms"])
        error = abs(prediction - actual) / actual
        models[key] = {
            "intercept_ms": intercept,
            "slope_ms_per_fma": slope,
            "fit_counts": fit_counts,
            "holdout_count": holdout,
            "holdout_actual_ms": actual,
            "holdout_predicted_ms": prediction,
            "holdout_relative_error": error,
            "holdout_pass": error
            <= float(config["acceptance"]["maximum_holdout_relative_error"]),
        }
        holdout_checks[key] = models[key]["holdout_pass"]
        timing_artifact_checks[key] = all(
            record["pass"]
            and qualify(PROJECT_ROOT / record["log"]["path"], record["log"])[
                "pass"
            ]
            for record in records.values()
        )
    allowed_cases = set(config["figure24"]["cases"])
    allowed_operators = set(config["figure24"]["operators"])
    comparisons = [
        item
        for item in identity["comparisons"]
        if item["surface"] == "fig24"
        and item["case"]["name"] in allowed_cases
        and item["operator"]["name"] in allowed_operators
    ]
    rows: list[dict[str, Any]] = []
    row_checks: dict[str, bool] = {}
    for comparison in comparisons:
        key = comparison["key"]
        service = service_key(comparison)
        model = models[service]
        full_fma = int(comparison["actual"]["fu"]["fma"])
        rtx_ms = model["intercept_ms"] + model["slope_ms_per_fma"] * full_fma
        rtx_seconds = rtx_ms / 1000.0
        mlx_cycles = float(mlx["full_estimates"][key]["cycles"])
        mlx_seconds = mlx_cycles / float(config["figure24"]["mlx_clock_hz"])
        ratio = rtx_seconds / mlx_seconds
        row = {
            "key": key,
            "case": comparison["case"]["name"],
            "operator": comparison["operator"]["name"],
            "service_key": service,
            "full_fma": full_fma,
            "rtx4090_estimated_seconds": rtx_seconds,
            "mlx_simulated_seconds": mlx_seconds,
            "rtx4090_seconds_div_mlx_seconds": ratio,
            "mlx_faster_than_rtx4090": ratio > 1.0,
            "service_holdout_pass": model["holdout_pass"],
        }
        rows.append(row)
        row_checks[key] = all(
            math.isfinite(float(row[field])) and float(row[field]) > 0
            for field in (
                "full_fma",
                "rtx4090_estimated_seconds",
                "mlx_simulated_seconds",
                "rtx4090_seconds_div_mlx_seconds",
            )
        )
    matrix_checks = {
        "rows": len(rows) == int(config["figure24"]["required_rows"]),
        "keys": len({row["key"] for row in rows}) == len(rows),
        "cases": {row["case"] for row in rows} == allowed_cases,
        "operators": {row["operator"] for row in rows} == allowed_operators,
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    target_free_check = (
        manifest["paper_performance_targets_consumed"] is False
        and config["acceptance"]["paper_targets_consumed"] is False
    )
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        all(gpu_checks.values()),
        len(models) == int(config["acceptance"]["required_service_configs"]),
        all(correctness_checks.values()),
        sum(len(items) for items in manifest["timings"].values())
        == int(config["acceptance"]["required_timing_runs"]),
        all(holdout_checks.values()),
        all(matrix_checks.values()),
        all(row_checks.values()),
        all(timing_artifact_checks.values())
        and qualify(PROJECT_ROOT / manifest["binary"]["path"], manifest["binary"])[
            "pass"
        ],
        target_free_check and all(item["pass"] for item in source_files.values()),
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "manifest": manifest_file["pass"] and all(manifest["checks"].values()),
        "gpu": all(gpu_checks.values()),
        "correctness": all(correctness_checks.values()),
        "timings": all(timing_artifact_checks.values()),
        "models": len(models) == 10,
        "holdouts_evaluated": len(holdout_checks) == 10,
        "matrix": all(matrix_checks.values()),
        "rows": all(row_checks.values()),
        "source": all(item["pass"] for item in source_files.values()),
        "target_free": target_free_check,
        "acceptance_evaluated": len(acceptance_gates) == 10
        and all(isinstance(value, bool) for value in acceptance_gates),
    }
    integrity = all(integrity_checks.values())
    supported = integrity and all(acceptance_gates)
    failed_services = sorted(key for key, passed in holdout_checks.items() if not passed)
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
        "paper_performance_targets_consumed": False,
        "paper_reproduction_claim": "none_native_4090_replacement_exploration",
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "generated_input": manifest_file,
        "gpu_checks": gpu_checks,
        "correctness_checks": correctness_checks,
        "timing_artifact_checks": timing_artifact_checks,
        "models": models,
        "holdout_checks": holdout_checks,
        "matrix_checks": matrix_checks,
        "rows": rows,
        "row_checks": row_checks,
        "source_files": source_files,
        "target_free_check": target_free_check,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "native_gpu": manifest["gpu_before"]["name"],
            "service_configs": len(models),
            "correctness_runs": len(correctness_checks),
            "timing_runs": sum(len(items) for items in manifest["timings"].values()),
            "holdout_passes": sum(holdout_checks.values()),
            "holdout_total": len(holdout_checks),
            "failed_services": failed_services,
            "maximum_holdout_relative_error": max(
                model["holdout_relative_error"] for model in models.values()
            ),
            "figure24_rows": len(rows),
            "mlx_faster_rows": sum(row["mlx_faster_than_rtx4090"] for row in rows),
            "rtx4090_faster_rows": sum(
                not row["mlx_faster_than_rtx4090"] for row in rows
            ),
            "figure24_rtx4090_complete": supported,
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
            "gpu_checks",
            "models",
            "holdout_checks",
            "rows",
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
