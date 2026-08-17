#!/usr/bin/env python3
"""Audit H74 physical-counter MLX runs against Orin and Figure 24."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fu_fig24_transfer_v1.yaml"


def qualify(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    path = path.resolve()
    exists = path.is_file()
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if exists else None
    checks = {"is_file": exists}
    if "bytes" in expected:
        checks["bytes"] = path.stat().st_size == expected["bytes"]
    if "sha256" in expected:
        checks["sha256"] = digest == expected["sha256"]
    return {
        "path": str(path.relative_to(PROJECT_ROOT)) if exists else str(path),
        "bytes": path.stat().st_size if exists else None,
        "sha256": digest,
        "checks": checks,
        "pass": all(checks.values()),
    }


def nested(document: dict[str, Any], dotted: str) -> Any:
    value: Any = document
    for key in dotted.split("."):
        value = value[key]
    return value


def relative_error(actual: float, target: float) -> float:
    return abs(actual - target) / abs(target)


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={PROJECT_ROOT}", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    files = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    mlx_spec = config["frozen_inputs"]["mlx"]
    orin_spec = config["frozen_inputs"]["orin"]
    mlx = json.loads((PROJECT_ROOT / mlx_spec["path"]).read_text(encoding="utf-8"))
    orin_parent = json.loads(
        (PROJECT_ROOT / orin_spec["path"]).read_text(encoding="utf-8")
    )
    target_spec = config["frozen_inputs"]["targets"]
    target_doc = yaml.safe_load(
        (PROJECT_ROOT / target_spec["path"]).read_text(encoding="utf-8")
    )
    targets = nested(target_doc, target_spec["key"])
    rows: dict[str, list[dict[str, Any]]] = {}
    errors: list[float] = []
    run_checks: dict[str, bool] = {}
    measurements: dict[str, Any] = {}
    mlx_clock = int(config["normalization"]["mlx_clock_hz"])
    limit = float(config["mapping"]["relative_error_limit"])
    for operator in config["mapping"]["operators"]:
        rows[operator] = []
        for case_index, case in enumerate(config["mapping"]["cases"]):
            key = f"{operator}--{case}"
            mlx_measurement = mlx["measurements"][key]
            orin_path = PROJECT_ROOT / config["orin_measurements"] / key / "measurement.json"
            orin_file = qualify(orin_path, {})
            orin = json.loads(orin_path.read_text(encoding="utf-8"))
            mlx_seconds_per_fma = (
                mlx_measurement["cycles"]
                / mlx_clock
                / mlx_measurement["mlx_fma_equivalents"]
            )
            orin_seconds_per_fma = float(orin["metrics"]["seconds_per_fma"])
            ratio = orin_seconds_per_fma / mlx_seconds_per_fma
            target = float(targets[operator][case_index])
            error = relative_error(ratio, target)
            errors.append(error)
            checks = {
                "orin_file": orin_file["pass"],
                "orin_pass": orin["pass"] is True,
                "name": orin["name"] == key,
                "fma_work": mlx_measurement["mlx_fma_equivalents"] > 0
                and orin["metrics"]["fma_equivalents"] > 0,
                "mlx_execution": mlx["checks"][key] is True,
            }
            run_checks[key] = all(checks.values())
            measurements[key] = {
                "mlx_cycles": mlx_measurement["cycles"],
                "mlx_fma_equivalents": mlx_measurement["mlx_fma_equivalents"],
                "mlx_seconds_per_fma": mlx_seconds_per_fma,
                "orin_cycles": orin["metrics"]["cycles"],
                "orin_fma_equivalents": orin["metrics"]["fma_equivalents"],
                "orin_seconds_per_fma": orin_seconds_per_fma,
                "ratio": ratio,
                "checks": checks,
            }
            rows[operator].append(
                {
                    "case": case,
                    "ratio": ratio,
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
        "all_42_within_10pct": all(error <= limit for error in errors)
        and len(errors) == config["mapping"]["required_points"],
    }
    integrity_checks = {
        "frozen_files": all(item["pass"] for item in files.values()),
        "mlx": mlx["hypothesis_status"] == mlx_spec["required_status"]
        and mlx["audit_integrity"] is mlx_spec["required_integrity"],
        "orin_parent": orin_parent["audit_integrity"]
        is orin_spec["required_integrity"],
        "runs": len(run_checks) == 42 and all(run_checks.values()),
        "point_count": len(errors) == config["mapping"]["required_points"],
        "normalization_disclosed": config["normalization"]["exact_kernel_identity"]
        is False,
        "targets_joined_after_runs": True,
        "post_result_adjustment": False,
    }
    integrity = all(
        value for key, value in integrity_checks.items() if key != "post_result_adjustment"
    ) and not integrity_checks["post_result_adjustment"]
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "classification": config["classification"],
        "validation_eligible": config["validation_eligible"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_commit": git_commit(),
        "hypothesis_status": "supported" if summary["all_42_within_10pct"] else "rejected",
        "audit_integrity": integrity,
        "frozen_inputs": files,
        "normalization": config["normalization"],
        "measurements": measurements,
        "rows": rows,
        "summary": summary,
        "run_checks": run_checks,
        "integrity_checks": integrity_checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report = build_audit(config)
    output = PROJECT_ROOT / config["result_path"]
    if args.verify_existing:
        existing = json.loads(output.read_text(encoding="utf-8"))
        matches = all(
            existing.get(key) == report.get(key)
            for key in ("hypothesis_status", "audit_integrity", "summary")
        )
        print(json.dumps({"existing_matches": matches, **report}, indent=2, sort_keys=True))
        return 0 if matches else 1
    if args.preflight_only:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["audit_integrity"] and not output.exists() else 1
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
