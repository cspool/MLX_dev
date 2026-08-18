#!/usr/bin/env python3
"""Audit H95 GEMM/memory targets and close unavailable Figure 21 speedups."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/fig21_evidence_closure_v1.yaml"


def qualify(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    exists = path.is_file()
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if exists else None
    checks = {"is_file": exists, "sha256": digest == expected["sha256"]}
    return {
        "path": str(path.relative_to(PROJECT_ROOT)) if exists else str(path),
        "bytes": path.stat().st_size if exists else None,
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
    mlx = json.loads((PROJECT_ROOT / mlx_spec["path"]).read_text(encoding="utf-8"))
    target = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["targets"]["path"]).read_text(
            encoding="utf-8"
        )
    )["derived_targets"]
    parent_check = (
        mlx["hypothesis_status"] == mlx_spec["required_status"]
        and mlx["audit_integrity"] is mlx_spec["required_integrity"]
    )
    limit = float(config["relative_error_limit"])
    rows = []
    for index, mlx_row in enumerate(mlx["rows"]):
        n = int(mlx_row["sequence_length"])
        series = {
            "gemm_time_pct": 100.0 * float(mlx_row["gemm_time_share"]),
            "dense_memory_gb": float(mlx_row["memory"]["dense"]),
            "sparse_memory_gb": float(mlx_row["memory"]["sparse"]),
        }
        for name, actual in series.items():
            target_value = float(target[name][index])
            error = relative_error(actual, target_value)
            passed = error <= limit
            rows.append(
                {
                    "sequence_length": n,
                    "series": name,
                    "actual": actual,
                    "target": target_value,
                    "relative_error": error,
                    "pass_10pct": passed,
                    "status": "reproduced" if passed else "numerical_failure",
                }
            )
        rows.append(
            {
                "sequence_length": n,
                "series": "speedup_over_xavier",
                "actual": None,
                "target": float(target["speedup_over_xavier"][index]),
                "relative_error": None,
                "pass_10pct": None,
                "status": "execution_incomplete",
                "reason": "matched_dense_xavier_tensor_cycles_unavailable",
            }
        )
    counts = Counter(row["status"] for row in rows)
    series_summary = {}
    for name in (
        "gemm_time_pct",
        "dense_memory_gb",
        "sparse_memory_gb",
        "speedup_over_xavier",
    ):
        points = [row for row in rows if row["series"] == name]
        numeric = [row for row in points if row["relative_error"] is not None]
        series_summary[name] = {
            "point_count": len(points),
            "passing_points": sum(row["pass_10pct"] is True for row in points),
            "mape": (
                sum(row["relative_error"] for row in numeric) / len(numeric)
                if numeric
                else None
            ),
            "max_error": max((row["relative_error"] for row in numeric), default=None),
            "available": bool(numeric),
        }
    summary = {
        "total_target_values": len(rows),
        "status_counts": {
            status: counts.get(status, 0)
            for status in ("reproduced", "numerical_failure", "execution_incomplete")
        },
        "series": series_summary,
        "figure21_reproduced_within_10pct": False,
    }
    integrity_checks = {
        "frozen_files": all(item["pass"] for item in files.values()),
        "parent": parent_check,
        "twenty_values": len(rows) == 20,
        "five_each": all(item["point_count"] == 5 for item in series_summary.values()),
        "speedup_null": all(
            row["actual"] is None and row["status"] == "execution_incomplete"
            for row in rows
            if row["series"] == "speedup_over_xavier"
        ),
        "global_false": summary["figure21_reproduced_within_10pct"] is False,
    }
    integrity = all(integrity_checks.values())
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "classification": config["classification"],
        "validation_eligible": config["validation_eligible"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_commit": git_commit(),
        "hypothesis_status": "supported" if integrity else "rejected",
        "audit_integrity": integrity,
        "frozen_inputs": files,
        "parent_check": parent_check,
        "rows": rows,
        "summary": summary,
        "integrity_checks": integrity_checks,
        "paper_target_values_consumed_during_audit": True,
        "stopping_rule_applied": True,
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
        keys = ("hypothesis_status", "audit_integrity", "rows", "summary", "integrity_checks")
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
