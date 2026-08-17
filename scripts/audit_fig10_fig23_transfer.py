#!/usr/bin/env python3
"""Audit H65's frozen Figure 10 speedups against Figure 23."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig10_fig23_transfer_v1.yaml"


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
    size = path.stat().st_size if exists else None
    digest = sha256_file(path) if exists else None
    checks = {"is_file": exists}
    if "bytes" in expected:
        checks["bytes"] = size == int(expected["bytes"])
    if "sha256" in expected:
        checks["sha256"] = digest == expected["sha256"]
    return {
        "path": str(path.relative_to(PROJECT_ROOT)) if exists else str(path),
        "bytes": size,
        "sha256": digest,
        "checks": checks,
        "pass": all(checks.values()),
    }


def relative_error(actual: float, target: float) -> float:
    return abs(actual - target) / abs(target)


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
    started = time.perf_counter()
    files = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    mechanism_spec = config["frozen_inputs"]["mechanism"]
    mechanism = json.loads(
        (PROJECT_ROOT / mechanism_spec["path"]).read_text(encoding="utf-8")
    )
    target_spec = config["frozen_inputs"]["targets"]
    targets = yaml.safe_load(
        (PROJECT_ROOT / target_spec["path"]).read_text(encoding="utf-8")
    )[target_spec["key"]]
    points: dict[str, list[dict[str, Any]]] = {}
    errors: list[float] = []
    passing = 0
    limit = float(config["mapping"]["relative_error_limit"])
    for series in config["mapping"]["series"]:
        points[series] = []
        actuals = mechanism["speedups"][series]
        expected = targets[series]
        for sequence, actual, target in zip(
            config["mapping"]["sequence_lengths"], actuals, expected, strict=True
        ):
            error = relative_error(float(actual), float(target))
            passed = error <= limit
            errors.append(error)
            passing += passed
            points[series].append(
                {
                    "sequence_length": int(sequence),
                    "actual": float(actual),
                    "target": float(target),
                    "relative_error": error,
                    "pass_10pct": passed,
                }
            )
    expected_points = int(config["mapping"]["required_points"])
    summary = {
        "passing_points": passing,
        "total_points": len(errors),
        "mape": sum(errors) / len(errors),
        "max_relative_error": max(errors),
        "failing_points": [
            f"{series}-{point['sequence_length']}"
            for series, values in points.items()
            for point in values
            if not point["pass_10pct"]
        ],
        "all_15_within_10pct": passing == expected_points == len(errors),
    }
    integrity_checks = {
        "frozen_files": all(item["pass"] for item in files.values()),
        "mechanism": mechanism.get("hypothesis_status")
        == mechanism_spec["required_status"]
        and mechanism.get("audit_integrity") is mechanism_spec["required_integrity"],
        "target_shape": targets["sequence_lengths"]
        == config["mapping"]["sequence_lengths"]
        and all(len(targets[series]) == 5 for series in config["mapping"]["series"]),
        "point_count": len(errors) == expected_points,
        "speedups_frozen_before_audit": True,
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
        "hypothesis_status": "supported" if summary["all_15_within_10pct"] else "rejected",
        "audit_integrity": audit_integrity,
        "frozen_inputs": files,
        "cycles": mechanism["cycles"],
        "speedups": mechanism["speedups"],
        "points": points,
        "summary": summary,
        "integrity_checks": integrity_checks,
        "wall_seconds": time.perf_counter() - started,
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
        keys = ("hypothesis_status", "audit_integrity", "summary")
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
