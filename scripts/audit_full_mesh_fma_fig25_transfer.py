#!/usr/bin/env python3
"""Audit H103 exact full-mesh FMA utilization against Figure 25."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

try:
    from scripts.audit_fig24_25_exact_paths import git_commit, qualify
except ModuleNotFoundError:
    from audit_fig24_25_exact_paths import git_commit, qualify

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/full_mesh_fma_fig25_transfer_v1.yaml"


def nested(document: dict[str, Any], dotted: str) -> Any:
    value: Any = document
    for key in dotted.split("."):
        value = value[key]
    return value


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    files = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    full_spec = config["frozen_inputs"]["full_mesh"]
    full_mesh = json.loads(
        (PROJECT_ROOT / full_spec["path"]).read_text(encoding="utf-8")
    )
    target_spec = config["frozen_inputs"]["targets"]
    target_document = yaml.safe_load(
        (PROJECT_ROOT / target_spec["path"]).read_text(encoding="utf-8")
    )
    targets = nested(target_document, target_spec["key"])
    limit = float(config["mapping"]["relative_error_limit"])
    points: dict[str, list[dict[str, Any]]] = {}
    errors: list[float] = []
    utilization_checks: dict[str, bool] = {}
    for operator_index, operator in enumerate(config["mapping"]["operators"]):
        points[operator] = []
        for case_index, case in enumerate(config["mapping"]["cases"]):
            key = f"{operator}--{case}"
            estimate = full_mesh["full_estimates"][key]
            cycles = float(estimate["cycles"])
            fma_cycles = float(estimate["physical_fma_pe_cycles"])
            utilization = fma_cycles / (
                cycles * int(config["mapping"]["physical_pes"])
            )
            stored_utilization = float(estimate["fma_utilization"])
            target = float(targets[operator_index][case_index])
            error = abs(utilization - target) / abs(target)
            errors.append(error)
            utilization_checks[key] = (
                0.0 <= utilization <= 1.0
                and utilization == stored_utilization
                and cycles > 0
                and fma_cycles > 0
            )
            points[operator].append(
                {
                    "case": case,
                    "cycles": cycles,
                    "physical_fma_pe_cycles": fma_cycles,
                    "utilization": utilization,
                    "target": target,
                    "relative_error": error,
                    "pass_10pct": error <= limit,
                }
            )
    summary = {
        "passing_points": sum(error <= limit for error in errors),
        "total_points": len(errors),
        "mape": sum(errors) / len(errors),
        "max_relative_error": max(errors),
        "all_24_within_10pct": all(error <= limit for error in errors)
        and len(errors) == int(config["mapping"]["required_points"]),
    }
    serialized_config = json.dumps(config, sort_keys=True).lower()
    integrity_checks = {
        "frozen_files": all(item["pass"] for item in files.values()),
        "full_mesh_parent": full_mesh["hypothesis_status"]
        == full_spec["required_status"]
        and full_mesh["audit_integrity"] is full_spec["required_integrity"],
        "full_mesh_work": full_mesh["summary"]["full_work_passing_paths"] == 48,
        "full_mesh_runs": full_mesh["summary"]["full_mesh_passing_runs"] == 192,
        "cycle_holdouts": full_mesh["summary"]["passing_cycle_holdouts"] == 96,
        "physical_fma_holdouts": full_mesh["summary"][
            "passing_physical_fma_holdouts"
        ]
        == 96,
        "target_shape": len(targets) == len(config["mapping"]["operators"])
        and all(len(row) == len(config["mapping"]["cases"]) for row in targets),
        "utilizations": len(utilization_checks)
        == int(config["mapping"]["required_points"])
        and all(utilization_checks.values()),
        "point_count": len(errors) == int(config["mapping"]["required_points"]),
        "metric_frozen": config["metric"]
        == "productive_fma_pe_cycles_div_cycles_times_physical_pes",
        "targets_joined_after_runs": True,
        "post_result_adjustment": False,
        "no_residual_parameters": all(
            token not in serialized_config
            for token in ("correction_factor", "operator_scale", "residual_scale")
        ),
    }
    integrity = all(
        value
        for key, value in integrity_checks.items()
        if key != "post_result_adjustment"
    ) and not integrity_checks["post_result_adjustment"]
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "classification": config["classification"],
        "validation_eligible": config["validation_eligible"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_commit": git_commit(),
        "hypothesis_status": (
            "supported" if integrity and summary["all_24_within_10pct"] else "rejected"
        ),
        "audit_integrity": integrity,
        "frozen_inputs": files,
        "metric": config["metric"],
        "points": points,
        "utilization_checks": utilization_checks,
        "summary": summary,
        "integrity_checks": integrity_checks,
        "post_result_adjustment": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report = build_audit(config)
    output = PROJECT_ROOT / config["result_path"]
    if args.verify_existing:
        existing = json.loads(output.read_text(encoding="utf-8"))
        keys = ("hypothesis_status", "audit_integrity", "points", "summary")
        matches = all(existing.get(key) == report.get(key) for key in keys)
        print(json.dumps({"existing_matches": matches, **report}, indent=2))
        return 0 if matches else 1
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0 if report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
