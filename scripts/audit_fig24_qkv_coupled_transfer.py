#!/usr/bin/env python3
"""Audit H127's frozen direct-time Figure 24 QKV subset transfer."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_fig22_coupled_transfer import (
    PROJECT_ROOT,
    git_commit,
    qualify,
    summarize,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig24_qkv_coupled_transfer_v1.yaml"


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    h126 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h126"]["path"]).read_text()
    )
    h114 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h114"]["path"]).read_text()
    )
    targets = yaml.safe_load(
        (PROJECT_ROOT / config["frozen_inputs"]["targets"]["path"]).read_text()
    )["fig24_structured_sweep"]["mlx_over_orin"]
    parent_checks = {
        "h126": h126["hypothesis_status"] == "supported"
        and h126["audit_integrity"] is True
        and h126["summary"]["full_estimates"] == 21,
        "h114": h114["hypothesis_status"] == "supported"
        and h114["audit_integrity"] is True,
    }
    points: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    mapping_checks: dict[str, bool] = {}
    limit = float(config["acceptance"]["relative_error_limit"])
    for operator in config["mapping"]["operators"]:
        target_row = targets[operator]
        for index, case in enumerate(config["mapping"]["cases"]):
            key = f"{operator}--{case}"
            identity = (operator, case)
            if identity in identities:
                raise ValueError(f"duplicate H127 identity: {identity}")
            identities.add(identity)
            orin = h126["full_estimates"][key]
            mlx = h114["full_estimates"][key]
            orin_seconds = float(orin["seconds"])
            mlx_cycles = float(mlx["cycles"])
            mlx_seconds = mlx_cycles / int(config["mapping"]["mlx_clock_hz"])
            prediction = orin_seconds / mlx_seconds
            target = float(target_row[index])
            error = abs(prediction - target) / abs(target)
            mapping_checks[key] = (
                orin["template"] == operator
                and orin["case"]["name"] == case
                and mlx["eligible"] is True
            )
            points.append(
                {
                    "key": key,
                    "operator": operator,
                    "case": case,
                    "orin_seconds": orin_seconds,
                    "mlx_cycles": mlx_cycles,
                    "mlx_seconds": mlx_seconds,
                    "prediction": prediction,
                    "target": target,
                    "relative_error": error,
                    "pass_10pct": error <= limit,
                    "ratio": config["mapping"]["ratio"],
                }
            )
    global_summary = summarize(points)
    by_operator = {
        operator: summarize(
            [point for point in points if point["operator"] == operator]
        )
        for operator in config["mapping"]["operators"]
    }
    finite_checks = {
        point["key"]: all(
            math.isfinite(float(point[name])) and float(point[name]) > 0
            for name in (
                "orin_seconds",
                "mlx_cycles",
                "mlx_seconds",
                "prediction",
                "target",
            )
        )
        and math.isfinite(float(point["relative_error"]))
        and float(point["relative_error"]) >= 0
        for point in points
    }
    coverage_checks = {
        "points": len(points) == int(config["mapping"]["required_points"]),
        "identities": len(identities) == int(config["mapping"]["required_points"]),
        "mapping": len(mapping_checks) == 21 and all(mapping_checks.values()),
        "targets": all(
            len(targets[operator]) == len(config["mapping"]["cases"])
            for operator in config["mapping"]["operators"]
        ),
        "summaries": all(item["points"] == 7 for item in by_operator.values()),
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    source_text = "\n".join(
        (PROJECT_ROOT / path).read_text(errors="replace")
        for path in config["source_layout"].values()
    )
    forbidden = (
        "fit" + "_affine",
        "correction" + "_factor",
        "pointwise" + "_oracle",
        "seconds_per" + "_fma",
        "prediction" + " *",
        "prediction" + " +",
    )
    source_checks = {
        "no_fit_or_selection": not any(token in source_text for token in forbidden),
        "direct_ratio": config["mapping"]["ratio"]
        == "orin_total_seconds_div_mlx_total_seconds",
    }
    all_points_pass = global_summary["passing_points"] == int(
        config["acceptance"]["required_passing_points"]
    )
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        all(coverage_checks.values()),
        all(mapping_checks.values()),
        all(point["ratio"] == config["mapping"]["ratio"] for point in points),
        coverage_checks["targets"],
        all(finite_checks.values()),
        all_points_pass,
        global_summary["points"] == 21
        and all(item["points"] == 7 for item in by_operator.values()),
        all(source_checks.values()) and all(item["pass"] for item in source_files.values()),
        config["acceptance"]["full_figure_completion"] is False,
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "coverage": all(coverage_checks.values()),
        "mapping": all(mapping_checks.values()),
        "finite": all(finite_checks.values()),
        "summaries": global_summary["points"] == 21,
        "source": all(source_checks.values())
        and all(item["pass"] for item in source_files.values()),
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
        "paper_reproduction_claim": "figure24_qkv_subset_only_not_full_figure",
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "mapping_checks": mapping_checks,
        "coverage_checks": coverage_checks,
        "finite_checks": finite_checks,
        "points": points,
        "summaries": {"global": global_summary, "by_operator": by_operator},
        "source_checks": source_checks,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "points": global_summary["points"],
            "passing_points": global_summary["passing_points"],
            "mape": global_summary["mape"],
            "max_relative_error": global_summary["max_relative_error"],
            "qkv_subset_supported": supported,
            "figure24_reproduced": False,
            "acceptance_gates_passed": sum(acceptance_gates),
            "acceptance_gates_total": len(acceptance_gates),
            "active_simulator_figures_reproduced": 0,
            "active_simulator_figures_total": 8,
        },
        "source_files": source_files,
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
            "points",
            "summaries",
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
