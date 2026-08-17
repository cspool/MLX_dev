#!/usr/bin/env python3
"""Validate one H55 Orin run and emit normalized timing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.audit_gpgpusim_rtx3090_proxy import parse_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def build_measurement(manifest_path: Path, name: str, log_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    job = next(item for item in manifest["orin_jobs"] if item["name"] == name)
    run = parse_run(log_path)
    summary = run["summary"] or {}
    fmas = int(job["gpu_fma_equivalents"])
    cycles = int(run["cycles"] or 0)
    clock = int(manifest["normalization"]["orin_clock_hz"])
    checks = {
        "operator": summary.get("operator") == job["gpu_operation"],
        "count": summary.get("count") == job["gpu_count"],
        "parameter": summary.get("parameter") == job["gpu_parameter"],
        "checksum": summary.get("relative_error", 1.0) <= 1e-6,
        "cycles": cycles > 0,
        "instructions": int(run["instructions"] or 0) > 0,
        "ctas": int(run["ctas"] or 0) > 0,
        "detailed": run["detailed_mode"],
        "exit": run["normal_exit"],
        "fmas": fmas > 0,
        "no_targets": manifest.get("paper_target_values_consumed") is False,
    }
    return {
        "schema_version": 1,
        "experiment_id": "H55",
        "name": name,
        "job": job,
        "run": run,
        "metrics": {
            "cycles": cycles,
            "clock_hz": clock,
            "fma_equivalents": fmas,
            "seconds": cycles / clock,
            "seconds_per_fma": cycles / clock / fmas,
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def main() -> int:
    args = parse_args()
    report = build_measurement(args.manifest, args.name, args.log)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"name": args.name, "cycles": report["metrics"]["cycles"], "pass": report["pass"]}))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
