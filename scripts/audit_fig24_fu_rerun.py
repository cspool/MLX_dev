#!/usr/bin/env python3
"""Audit H73 target-free Figure 24 MLX reruns."""

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
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig24_fu_rerun_v1.yaml"


def qualify(path: Path, expected: dict[str, Any] | None = None) -> dict[str, Any]:
    path = path.resolve()
    exists = path.is_file()
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if exists else None
    checks = {"is_file": exists}
    if expected and "bytes" in expected:
        checks["bytes"] = path.stat().st_size == expected["bytes"]
    if expected and "sha256" in expected:
        checks["sha256"] = digest == expected["sha256"]
    try:
        display = path.relative_to(PROJECT_ROOT)
    except ValueError:
        display = path
    return {
        "path": str(display),
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
    frozen_files = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    frozen = {
        name: json.loads((PROJECT_ROOT / spec["path"]).read_text(encoding="utf-8"))
        for name, spec in config["frozen_inputs"].items()
    }
    output_root = PROJECT_ROOT / config["output_root"]
    compile_path = output_root / "fig24-fu-compile-manifest.json"
    run_path = output_root / "fig24-fu-run-manifest.json"
    compile_file = qualify(compile_path)
    run_file = qualify(run_path)
    compiler = json.loads(compile_path.read_text(encoding="utf-8"))
    run = json.loads(run_path.read_text(encoding="utf-8"))
    measurements: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    for key, record in compiler["records"].items():
        first = run["runs"][key]["first"]
        second = run["runs"][key]["second"]
        summary = first["summary"]
        adapter = first["adapter"]
        metadata = record["metadata"]
        parent_document = json.loads(
            (PROJECT_ROOT / record["parent"]["path"]).read_text(encoding="utf-8")
        )
        expected_events = sum(
            int(block["trip_count"])
            for block in parent_document["blocks"]
            for instruction in block["instructions"]
            if instruction.get("emit_event")
        )
        capacity = summary["cycles"] * summary["physical_pe_count"]
        fma_cycles = summary["productive_pe_cycles_by_fu_class"].get("fma", 0)
        key_checks = {
            "parent": qualify(
                PROJECT_ROOT / record["parent"]["path"], record["parent"]
            )["pass"],
            "output": qualify(
                PROJECT_ROOT / record["output"]["path"], record["output"]
            )["pass"],
            "backend": record["only_backend_changed"] is True,
            "replay": first["summary_sha256"] == second["summary_sha256"]
            and first["adapter_sha256"] == second["adapter_sha256"],
            "done": summary["done"] is True,
            "instructions": summary["instructions_issued"]
            == summary["instructions_completed"]
            == sum(metadata["pipeline_counts"].values()),
            "events": summary["boundary_events_emitted"] == expected_events,
            "memory": summary["external_memory_requests"]
            == summary["external_memory_completions"]
            == adapter["requests"]
            == adapter["responses"]
            == metadata["memory_requests"],
            "fma": 0 < fma_cycles <= capacity,
        }
        checks[key] = all(key_checks.values())
        measurements[key] = {
            "cycles": summary["cycles"],
            "fma_productive_pe_cycles": fma_cycles,
            "fma_utilization": fma_cycles / capacity,
            "mlx_fma_equivalents": record["mlx_fma_equivalents"],
            "summary": summary,
            "adapter": adapter,
            "checks": key_checks,
        }
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    source_text = "\n".join(
        (PROJECT_ROOT / config["source_layout"][name]).read_text(encoding="utf-8")
        for name in ("compiler", "runner")
    )
    h55_spec = config["frozen_inputs"]["h55"]
    counter_spec = config["frozen_inputs"]["counters"]
    integrity_checks = {
        "frozen_files": all(item["pass"] for item in frozen_files.values()),
        "h55": frozen["h55"]["audit_integrity"] is h55_spec["required_integrity"],
        "counters": frozen["counters"]["hypothesis_status"]
        == counter_spec["required_status"]
        and frozen["counters"]["audit_integrity"]
        is counter_spec["required_integrity"],
        "compile": compile_file["pass"]
        and compiler["record_count"] == 42
        and compiler["all_only_backend_changed"] is True,
        "run": run_file["pass"] and all(run["checks"].values()),
        "measurements": len(checks) == 42 and all(checks.values()),
        "source": all(item["pass"] for item in source_files.values()),
        "target_paths_absent": "paper_targets" not in source_text
        and "fig24_structured" not in source_text,
        "targets_consumed": compiler["paper_performance_targets_consumed"] is False
        and run["paper_performance_targets_consumed"] is False,
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
        "frozen_inputs": frozen_files,
        "compile_manifest": compile_file,
        "run_manifest": run_file,
        "measurements": measurements,
        "checks": checks,
        "source_files": source_files,
        "integrity_checks": integrity_checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report = build_audit(config)
    output = PROJECT_ROOT / config["result_path"]
    if args.verify_existing:
        existing = json.loads(output.read_text(encoding="utf-8"))
        matches = all(
            existing.get(key) == report.get(key)
            for key in ("hypothesis_status", "audit_integrity", "integrity_checks")
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
