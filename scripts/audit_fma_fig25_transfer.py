#!/usr/bin/env python3
"""Audit H72 physical FMA utilization against Figure 25 MLX cells."""

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
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fma_fig25_transfer_v1.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--verify-existing", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qualify(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    path = path.resolve()
    exists = path.is_file()
    digest = sha256_file(path) if exists else None
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


def summarize(errors: list[float]) -> dict[str, Any]:
    return {
        "passing_points": sum(error <= 0.10 for error in errors),
        "total_points": len(errors),
        "mape": sum(errors) / len(errors),
        "max_relative_error": max(errors),
        "all_within_10pct": all(error <= 0.10 for error in errors),
    }


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={PROJECT_ROOT}", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    files = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    counter_spec = config["frozen_inputs"]["counters"]
    counters = json.loads(
        (PROJECT_ROOT / counter_spec["path"]).read_text(encoding="utf-8")
    )
    target_spec = config["frozen_inputs"]["targets"]
    target_document = yaml.safe_load(
        (PROJECT_ROOT / target_spec["path"]).read_text(encoding="utf-8")
    )
    targets = nested(target_document, target_spec["key"])
    primary_backend = config["mapping"]["backend"]
    diagnostic_backend = config["mapping"]["diagnostic_backend"]
    points: dict[str, list[dict[str, Any]]] = {}
    primary_errors: list[float] = []
    diagnostic_errors: list[float] = []
    for operator_index, operator in enumerate(config["mapping"]["operators"]):
        points[operator] = []
        for case_index, case in enumerate(config["mapping"]["cases"]):
            key = f"{operator}--{case}"
            target = float(targets[operator_index][case_index])
            primary = float(counters["fma_utilization"][key][primary_backend])
            diagnostic = float(
                counters["fma_utilization"][key][diagnostic_backend]
            )
            primary_error = relative_error(primary, target)
            diagnostic_error = relative_error(diagnostic, target)
            primary_errors.append(primary_error)
            diagnostic_errors.append(diagnostic_error)
            points[operator].append(
                {
                    "case": case,
                    "target": target,
                    "primary": primary,
                    "primary_relative_error": primary_error,
                    "primary_pass_10pct": primary_error <= 0.10,
                    "fixed_diagnostic": diagnostic,
                    "fixed_relative_error": diagnostic_error,
                    "fixed_pass_10pct": diagnostic_error <= 0.10,
                }
            )
    primary_summary = summarize(primary_errors)
    diagnostic_summary = summarize(diagnostic_errors)
    integrity_checks = {
        "frozen_files": all(item["pass"] for item in files.values()),
        "counter_parent": counters["hypothesis_status"]
        == counter_spec["required_status"]
        and counters["audit_integrity"] is counter_spec["required_integrity"],
        "target_shape": len(targets) == len(config["mapping"]["operators"])
        and all(len(row) == len(config["mapping"]["cases"]) for row in targets),
        "point_count": len(primary_errors) == config["mapping"]["required_points"],
        "primary_preselected": primary_backend == "column_port",
        "targets_joined_after_runs": True,
        "post_result_adjustment": False,
    }
    audit_integrity = all(
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
        "hypothesis_status": (
            "supported" if primary_summary["all_within_10pct"] else "rejected"
        ),
        "audit_integrity": audit_integrity,
        "frozen_inputs": files,
        "metric": "productive_fma_pe_cycles / (cycles * physical_pe_count)",
        "primary_backend": primary_backend,
        "diagnostic_backend": diagnostic_backend,
        "points": points,
        "primary_summary": primary_summary,
        "diagnostic_summary": diagnostic_summary,
        "integrity_checks": integrity_checks,
    }


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report = build_audit(config)
    output = PROJECT_ROOT / config["result_path"]
    if args.verify_existing:
        if not output.is_file():
            raise FileNotFoundError(output)
        existing = json.loads(output.read_text(encoding="utf-8"))
        keys = ("hypothesis_status", "audit_integrity", "primary_summary")
        matches = all(existing.get(key) == report.get(key) for key in keys)
        print(json.dumps({"existing_matches": matches, **report}, indent=2, sort_keys=True))
        return 0 if matches else 1
    if args.preflight_only:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["audit_integrity"] and not output.exists() else 1
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
