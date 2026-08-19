#!/usr/bin/env python3
"""Audit H178's SWA-W256 post-regime Figure-24 completion."""

from __future__ import annotations

import argparse
import copy
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/simulators/fig24_rtx4090_postcache_v1.yaml"
)


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    h177 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h177"]["path"]).read_text()
    )
    parent_checks = {
        "status": h177["hypothesis_status"]
        == config["frozen_inputs"]["h177"]["required_status"],
        "integrity": h177["audit_integrity"]
        is config["frozen_inputs"]["h177"]["required_integrity"],
        "one_failure": h177["summary"]["failed_services"] == ["swa-w256"]
        and h177["summary"]["holdout_passes"] == 9,
        "matrix": h177["summary"]["figure24_rows"] == 42,
        "target_free": h177["paper_performance_targets_consumed"] is False,
    }
    output_root = PROJECT_ROOT / config["output_root"]
    manifest_path = output_root / "fig24-rtx4090-postcache-run-manifest.json"
    manifest_file = qualify(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    gpu_checks = {
        "name": manifest["gpu_before"]["name"] == config["gpu"]["expected_name"],
        "compute": manifest["gpu_before"]["compute_cap"]
        == config["gpu"]["expected_compute_capability"],
        "stable": manifest["gpu_before"]["uuid"] == manifest["gpu_after"]["uuid"],
    }
    fit_counts = [int(value) for value in config["post_regime"]["fit_counts"]]
    holdout = int(config["post_regime"]["holdout_counts"][0])
    first = manifest["records"][str(fit_counts[0])]
    second = manifest["records"][str(fit_counts[1])]
    held = manifest["records"][str(holdout)]
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
    post_model = {
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
    record_checks = {
        count: record["pass"]
        and record["summary"]["operation"] == "swa"
        and record["summary"]["parameter"] == 256
        and record["summary"]["repeat"] == 1
        and qualify(PROJECT_ROOT / record["log"]["path"], record["log"])["pass"]
        for count, record in manifest["records"].items()
    }
    inherited_models = {
        key: value for key, value in h177["models"].items() if key != "swa-w256"
    }
    combined_models = {**copy.deepcopy(inherited_models), "swa-w256": post_model}
    models_checks = {
        "inherited_count": len(inherited_models)
        == int(config["acceptance"]["required_inherited_models"]),
        "total": len(combined_models)
        == int(config["acceptance"]["required_total_models"]),
        "inherited_exact": all(
            inherited_models[key] == h177["models"][key] for key in inherited_models
        ),
        "all_holdouts": all(model["holdout_pass"] for model in combined_models.values()),
    }
    rows: list[dict[str, Any]] = []
    row_checks: dict[str, bool] = {}
    changed_rows = 0
    unchanged_rows = 0
    for old in h177["rows"]:
        row = copy.deepcopy(old)
        if row["service_key"] == "swa-w256":
            rtx_ms = intercept + slope * int(row["full_fma"])
            row["rtx4090_estimated_seconds"] = rtx_ms / 1000.0
            row["rtx4090_seconds_div_mlx_seconds"] = (
                row["rtx4090_estimated_seconds"] / row["mlx_simulated_seconds"]
            )
            row["mlx_faster_than_rtx4090"] = (
                row["rtx4090_seconds_div_mlx_seconds"] > 1.0
            )
            row["service_holdout_pass"] = post_model["holdout_pass"]
            row["service_regime"] = "post_262K"
            changed_rows += 1
        else:
            unchanged_rows += 1
        rows.append(row)
        row_checks[row["key"]] = all(
            math.isfinite(float(row[field])) and float(row[field]) > 0
            for field in (
                "full_fma",
                "rtx4090_estimated_seconds",
                "mlx_simulated_seconds",
                "rtx4090_seconds_div_mlx_seconds",
            )
        ) and row["service_holdout_pass"] is True
    matrix_checks = {
        "rows": len(rows) == int(config["acceptance"]["required_figure24_rows"]),
        "unique": len({row["key"] for row in rows}) == len(rows),
        "changed": changed_rows == 7,
        "unchanged": unchanged_rows == 35,
        "unchanged_exact": all(
            row == old
            for row, old in zip(rows, h177["rows"], strict=True)
            if row["service_key"] != "swa-w256"
        ),
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
        len(manifest["records"]) == int(config["acceptance"]["required_new_timings"])
        and all(record_checks.values()),
        math.isfinite(intercept) and math.isfinite(slope) and slope > 0,
        post_model["holdout_pass"],
        models_checks["inherited_count"] and models_checks["inherited_exact"],
        matrix_checks["changed"] and matrix_checks["unchanged_exact"],
        matrix_checks["rows"] and matrix_checks["unique"],
        all(row_checks.values()) and models_checks["all_holdouts"],
        target_free_check and all(item["pass"] for item in source_files.values()),
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parent": all(parent_checks.values()),
        "manifest": manifest_file["pass"] and all(manifest["checks"].values()),
        "gpu": all(gpu_checks.values()),
        "records": all(record_checks.values()),
        "model": all(models_checks.values()),
        "matrix": all(matrix_checks.values()),
        "rows": all(row_checks.values()),
        "source": all(item["pass"] for item in source_files.values()),
        "target_free": target_free_check,
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
        "paper_performance_targets_consumed": False,
        "paper_reproduction_claim": "none_native_4090_replacement_exploration",
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "generated_input": manifest_file,
        "gpu_checks": gpu_checks,
        "record_checks": record_checks,
        "post_model": post_model,
        "combined_models": combined_models,
        "models_checks": models_checks,
        "rows": rows,
        "row_checks": row_checks,
        "matrix_checks": matrix_checks,
        "source_files": source_files,
        "target_free_check": target_free_check,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "native_gpu": manifest["gpu_before"]["name"],
            "new_timings": len(manifest["records"]),
            "post_holdout_relative_error": error,
            "service_models": len(combined_models),
            "service_holdout_passes": sum(
                model["holdout_pass"] for model in combined_models.values()
            ),
            "figure24_rows": len(rows),
            "changed_rows": changed_rows,
            "unchanged_rows": unchanged_rows,
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
            "post_model",
            "combined_models",
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
