#!/usr/bin/env python3
"""Audit H66 standalone scratchpad against real H63 dsa-gem5 runs."""

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
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/standalone_dsagen_spad_v1.yaml"
RESOURCES = ("load", "store", "compute", "xfer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--verify-existing", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qualify(path: Path, expected: dict[str, Any] | None = None) -> dict[str, Any]:
    path = path.resolve()
    exists = path.is_file()
    digest = sha256_file(path) if exists else None
    checks = {"is_file": exists}
    if expected and "sha256" in expected:
        checks["sha256"] = digest == expected["sha256"]
    return {
        "path": str(path.relative_to(PROJECT_ROOT)) if exists else str(path),
        "bytes": path.stat().st_size if exists else None,
        "sha256": digest,
        "checks": checks,
        "pass": all(checks.values()),
    }


def relative_error(actual: float, target: float) -> float:
    return abs(actual - target) / abs(target) if target else float(actual != target)


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
    reference_spec = config["frozen_reference"]
    reference_file = qualify(PROJECT_ROOT / reference_spec["path"], reference_spec)
    reference = json.loads(
        (PROJECT_ROOT / reference_spec["path"]).read_text(encoding="utf-8")
    )
    run_path = PROJECT_ROOT / config["output_root"] / "standalone-spad-run-manifest.json"
    run_file = qualify(run_path)
    run = json.loads(run_path.read_text(encoding="utf-8"))
    points: dict[str, Any] = {}
    cycle_errors: list[float] = []
    utilization_errors: list[float] = []
    run_checks: dict[str, bool] = {}
    for panel, values in reference["points"].items():
        kernel = "fft" if panel == "chunk_fft" else panel
        points[panel] = []
        for reference_point in values:
            size = int(reference_point["size"])
            key = f"{kernel}-{size}"
            record = run["runs"][key]["first"]
            summary_path = PROJECT_ROOT / record["summary_path"]
            adapter_path = PROJECT_ROOT / record["adapter_path"]
            summary_file = qualify(summary_path, {"sha256": record["summary_sha256"]})
            adapter_file = qualify(adapter_path, {"sha256": record["adapter_sha256"]})
            summary = record["summary"]
            adapter = record["adapter"]
            expected = reference_point["primary_summary"]
            common_exact = all(
                summary.get(field) == value
                for field, value in expected.items()
                if field != "scenario"
            )
            cycle_error = relative_error(summary["cycles"], expected["cycles"])
            cycle_errors.append(cycle_error)
            resource_errors: dict[str, float] = {}
            for resource in RESOURCES:
                actual = summary["productive_pe_cycles_by_pipeline"][resource] / (
                    summary["cycles"] * summary["physical_pe_count"]
                )
                target = expected["productive_pe_cycles_by_pipeline"][resource] / (
                    expected["cycles"] * expected["physical_pe_count"]
                )
                error = abs(actual - target)
                utilization_errors.append(error)
                resource_errors[resource] = error
            checks = {
                "summary_file": summary_file["pass"],
                "adapter_file": adapter_file["pass"],
                "common_fields_exact": common_exact,
                "cycle": cycle_error <= config["acceptance"]["cycle_relative_error"],
                "utilization": all(
                    error
                    <= config["acceptance"]["productive_utilization_absolute_error"]
                    for error in resource_errors.values()
                ),
                "adapter_counts": adapter["requests"]
                == adapter["responses"]
                == summary["external_memory_requests"]
                == summary["external_memory_completions"],
            }
            run_checks[key] = all(checks.values())
            points[panel].append(
                {
                    "size": size,
                    "cycles": summary["cycles"],
                    "reference_cycles": expected["cycles"],
                    "cycle_relative_error": cycle_error,
                    "productive_utilization_absolute_errors": resource_errors,
                    "adapter": adapter,
                    "checks": checks,
                }
            )
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    accel_source = (
        PROJECT_ROOT / "third_party/dsa-framework/dsa-gem5/src/cpu/minor/ssim/accel.cc"
    ).read_text(encoding="utf-8", errors="replace")
    source_checks = {
        "shape": "spads.emplace_back(8, 8, SCRATCH_SIZE, 1" in accel_source,
        "buffer": "new dsa::sim::InputBuffer(4, 16, 1)" in accel_source,
    }
    summary = {
        "passing_runs": sum(run_checks.values()),
        "total_runs": len(run_checks),
        "max_cycle_relative_error": max(cycle_errors),
        "max_productive_utilization_absolute_error": max(utilization_errors),
        "all_16_pass": all(run_checks.values())
        and len(run_checks) == config["acceptance"]["required_runs"],
    }
    integrity_checks = {
        "reference": reference_file["pass"]
        and reference["audit_integrity"] is reference_spec["required_integrity"],
        "run_manifest": run_file["pass"],
        "replays": all(run["checks"].values()),
        "source_files": all(item["pass"] for item in source_files.values()),
        "dsagen_source": all(source_checks.values()),
        "paper_targets_consumed": run["paper_performance_targets_consumed"] is False,
    }
    audit_integrity = all(integrity_checks.values())
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "classification": config["classification"],
        "validation_eligible": config["validation_eligible"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_commit": git_commit(),
        "hypothesis_status": "supported" if summary["all_16_pass"] else "rejected",
        "audit_integrity": audit_integrity,
        "reference": reference_file,
        "run_manifest": run_file,
        "source_files": source_files,
        "source_checks": source_checks,
        "points": points,
        "run_checks": run_checks,
        "summary": summary,
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
