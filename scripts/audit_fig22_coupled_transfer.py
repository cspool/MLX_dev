#!/usr/bin/env python3
"""Join frozen H118 primary utilizations to all 64 Figure 22 targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig22_coupled_transfer_v1.yaml"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qualify(path: Path, expected: dict[str, Any] | None = None) -> dict[str, Any]:
    path = path.resolve()
    exists = path.is_file()
    size = path.stat().st_size if exists else None
    digest = sha256_file(path) if exists else None
    checks = {"is_file": exists}
    if expected and "bytes" in expected:
        checks["bytes"] = size == int(expected["bytes"])
    if expected and "sha256" in expected:
        checks["sha256"] = digest == expected["sha256"]
    try:
        display = path.relative_to(PROJECT_ROOT)
    except ValueError:
        display = path
    return {
        "path": str(display),
        "bytes": size,
        "sha256": digest,
        "checks": checks,
        "pass": all(checks.values()),
    }


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={PROJECT_ROOT}", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def summarize(points: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [float(point["relative_error"]) for point in points]
    return {
        "points": len(points),
        "passing_points": sum(point["pass_10pct"] for point in points),
        "mape": sum(errors) / len(errors),
        "max_relative_error": max(errors),
        "all_within_10pct": all(point["pass_10pct"] for point in points),
    }


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    h118 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h118"]["path"]).read_text()
    )
    h60 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h60"]["path"]).read_text()
    )
    parent_checks = {
        "h118": h118["hypothesis_status"]
        == config["frozen_inputs"]["h118"]["required_status"]
        and h118["audit_integrity"]
        is config["frozen_inputs"]["h118"]["required_integrity"],
        "h60": h60["verdict"]
        == config["frozen_inputs"]["h60"]["required_verdict"]
        and h60["summary"]["pass"]
        is config["frozen_inputs"]["h60"]["required_summary_pass"],
        "h60_count": h60["summary"]["numeric_value_count"] == 64,
    }
    h118_run_path = PROJECT_ROOT / h118["run_manifest"]["path"]
    h118_run_file = qualify(h118_run_path, h118["run_manifest"])
    h118_run = json.loads(h118_run_path.read_text())
    parent_checks["h118_target_free"] = all(h118["target_free_checks"].values())
    parent_checks["h118_regressions"] = (
        h118_run_file["pass"] and h118_run["checks"]["regressions"] is True
    )

    sizes = [int(value) for value in config["mapping"]["sizes"]]
    resources = list(config["mapping"]["resources"])
    prediction_field = config["mapping"]["prediction_field"]
    limit = float(config["acceptance"]["relative_error_limit"])
    points: list[dict[str, Any]] = []
    identities: set[tuple[str, int, str]] = set()
    mapping_checks: dict[str, bool] = {}
    for operator, panel in config["mapping"].items():
        if operator not in {"bsmm", "fft"}:
            continue
        target_panel = h60["derived_targets"]["panels"][panel]
        for index, size in enumerate(sizes):
            key = f"{operator}-{size}"
            measurement = h118["measurements"][key]
            mapping_checks[key] = (
                measurement["operator"] == operator
                and int(measurement["size"]) == size
                and measurement["launch_cycles"] is None
            )
            for resource in resources:
                identity = (operator, size, resource)
                if identity in identities:
                    raise ValueError(f"duplicate Figure 22 identity: {identity}")
                identities.add(identity)
                prediction = float(measurement[prediction_field][resource])
                target = float(target_panel[resource][index])
                error = abs(prediction - target) / abs(target)
                points.append(
                    {
                        "operator": operator,
                        "target_panel": panel,
                        "size": size,
                        "resource": resource,
                        "prediction": prediction,
                        "target": target,
                        "relative_error": error,
                        "pass_10pct": error <= limit,
                        "prediction_provenance": f"H118.measurements.{key}.{prediction_field}.{resource}",
                        "target_provenance": f"H60.derived_targets.panels.{panel}.{resource}[{index}]",
                    }
                )

    global_summary = summarize(points)
    by_resource = {
        resource: summarize(
            [point for point in points if point["resource"] == resource]
        )
        for resource in resources
    }
    by_operator = {
        operator: summarize(
            [point for point in points if point["operator"] == operator]
        )
        for operator in config["mapping"]
        if operator in {"bsmm", "fft"}
    }
    finite_checks = {
        f"{point['operator']}-{point['size']}-{point['resource']}": (
            math.isfinite(point["prediction"])
            and math.isfinite(point["target"])
            and math.isfinite(point["relative_error"])
            and 0 <= point["prediction"] <= 1
            and 0 < point["target"] <= 1
        )
        for point in points
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    source_text = "\n".join(
        (PROJECT_ROOT / path).read_text(errors="replace")
        for path in config["source_layout"].values()
    )
    source_checks = {
        "no_fit": "fit" + "_affine" not in source_text,
        "no_oracle": "pointwise" + "_oracle" not in source_text,
        "no_correction": "correction" + "_factor" not in source_text,
        "no_alternative_field": config["mapping"]["diagnostic_field_forbidden"]
        not in source_text,
        "no_prediction_arithmetic": "prediction" + " *" not in source_text
        and "prediction" + " +" not in source_text,
    }
    coverage_checks = {
        "points": len(points) == int(config["mapping"]["required_points"]),
        "identities": len(identities) == int(config["mapping"]["required_points"]),
        "mapping": len(mapping_checks) == 16 and all(mapping_checks.values()),
        "sizes": sizes == [int(value) for value in h60["derived_targets"]["sizes"]],
        "resource_summaries": all(item["points"] == 16 for item in by_resource.values()),
        "operator_summaries": all(item["points"] == 32 for item in by_operator.values()),
    }
    all_points_pass = (
        global_summary["passing_points"]
        == int(config["acceptance"]["required_passing_points"])
    )
    acceptance_gates = [
        all(item["pass"] for item in frozen.values())
        and all(parent_checks.values()),
        all(coverage_checks.values()),
        all(mapping_checks.values())
        and all(point["prediction_provenance"].startswith("H118.") for point in points),
        all(point["target_provenance"].startswith("H60.") for point in points),
        all(finite_checks.values()),
        all_points_pass,
        global_summary["points"] == 64
        and all(item["points"] > 0 for item in by_resource.values())
        and all(item["points"] > 0 for item in by_operator.values()),
        parent_checks["h118_target_free"] and parent_checks["h118_regressions"],
        all(source_checks.values()) and all(item["pass"] for item in source_files.values()),
        config["acceptance"]["figure_complete_only_if_all_points_pass"] is True,
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "h118_run": h118_run_file["pass"],
        "coverage": all(coverage_checks.values()),
        "mapping": all(mapping_checks.values()),
        "finite": all(finite_checks.values()),
        "summaries": global_summary["points"] == 64,
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
        "paper_reproduction_claim": (
            "figure22_complete" if supported else "figure22_rejected"
        ),
        "frozen_inputs": frozen,
        "h118_run_manifest": h118_run_file,
        "parent_checks": parent_checks,
        "mapping_checks": mapping_checks,
        "coverage_checks": coverage_checks,
        "finite_checks": finite_checks,
        "points": points,
        "summaries": {
            "global": global_summary,
            "by_resource": by_resource,
            "by_operator": by_operator,
        },
        "source_checks": source_checks,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "points": global_summary["points"],
            "passing_points": global_summary["passing_points"],
            "mape": global_summary["mape"],
            "max_relative_error": global_summary["max_relative_error"],
            "acceptance_gates_passed": sum(acceptance_gates),
            "acceptance_gates_total": len(acceptance_gates),
            "figure22_reproduced": supported,
            "active_simulator_figures_reproduced": 1 if supported else 0,
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
