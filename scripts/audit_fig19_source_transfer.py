#!/usr/bin/env python3
"""Compare frozen H98 source cycles with Figure 19 MLX components."""

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
DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/fig19_source_transfer_v1.yaml"


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
    source_spec = config["frozen_inputs"]["source_paths"]
    source = json.loads(
        (PROJECT_ROOT / source_spec["path"]).read_text(encoding="utf-8")
    )
    target = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["targets"]["path"]).read_text(
            encoding="utf-8"
        )
    )["digitization"]["derived_targets"]
    parent_check = (
        source["hypothesis_status"] == source_spec["required_status"]
        and source["audit_integrity"] is source_spec["required_integrity"]
    )
    layers = int(config["layers"])
    clock = int(config["clock_hz"])
    limit = float(config["relative_error_limit"])
    points = []
    simulated = []
    errors = []
    for index, n_value in enumerate(target["sequence_lengths"]):
        n = int(n_value)
        attention = float(source["full_estimates"][f"N{n}-fft2d"]) * layers / clock * 1000
        ffn = (
            float(source["full_estimates"][f"N{n}-global_ffn1"])
            + float(source["full_estimates"][f"N{n}-global_ffn2"])
        ) * layers / clock * 1000
        values = {"attention_latency_ms": attention, "ffn_latency_ms": ffn, "total_latency_ms": attention + ffn}
        simulated.append({"sequence_length": n, **values})
        for series, actual in values.items():
            target_value = float(target["mlx"][series][index])
            error = abs(actual - target_value) / target_value
            errors.append(error)
            points.append(
                {
                    "sequence_length": n,
                    "series": series,
                    "actual_ms": actual,
                    "target_ms": target_value,
                    "relative_error": error,
                    "pass_10pct": error <= limit,
                }
            )
    summary = {
        "point_count": len(points),
        "passing_points": sum(point["pass_10pct"] for point in points),
        "mape": sum(errors) / len(errors),
        "max_error": max(errors),
        "all_points_pass": all(point["pass_10pct"] for point in points),
    }
    integrity_checks = {
        "frozen_files": all(item["pass"] for item in files.values()),
        "parent": parent_check,
        "twelve_points": len(points) == 12,
        "component_sums": all(
            abs(item["attention_latency_ms"] + item["ffn_latency_ms"] - item["total_latency_ms"])
            < 1e-12
            for item in simulated
        ),
        "stopping_rule": True,
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
        "hypothesis_status": (
            "supported" if integrity and summary["all_points_pass"] else "rejected"
        ),
        "audit_integrity": integrity,
        "frozen_inputs": files,
        "parent_check": parent_check,
        "simulated": simulated,
        "points": points,
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
        keys = ("hypothesis_status", "audit_integrity", "simulated", "points", "summary", "integrity_checks")
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
