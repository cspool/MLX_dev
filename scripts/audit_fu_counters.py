#!/usr/bin/env python3
"""Audit H71 physical FU-class counters without paper targets."""

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
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fu_counters_v1.yaml"


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
        check=False,
        capture_output=True,
        text=True,
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
    compile_path = output_root / "fu-counter-compile-manifest.json"
    run_path = output_root / "fu-counter-run-manifest.json"
    compile_file = qualify(compile_path)
    run_file = qualify(run_path)
    compiler = json.loads(compile_path.read_text(encoding="utf-8"))
    run = json.loads(run_path.read_text(encoding="utf-8"))
    outputs: dict[str, Any] = {}
    execution_checks: dict[str, bool] = {}
    fma_utilization: dict[str, dict[str, float]] = {}
    for key, record in compiler["records"].items():
        parent_path = PROJECT_ROOT / record["parent"]["path"]
        parent_file = qualify(parent_path, record["parent"])
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        metadata = parent["metadata"]
        expected_classes = {
            parent["functional_units"][operation]["class"]
            for operation, count in metadata["operation_counts"].items()
            if count > 0
        }
        outputs[key] = {"parent": parent_file, "backends": {}}
        fma_utilization[key] = {}
        for backend, backend_record in record["backends"].items():
            output_file = qualify(
                PROJECT_ROOT / backend_record["output"]["path"],
                backend_record["output"],
            )
            first = run["runs"][key][backend]["first"]
            second = run["runs"][key][backend]["second"]
            first_file = qualify(
                PROJECT_ROOT / first["summary_path"],
                {"sha256": first["summary_sha256"]},
            )
            second_file = qualify(
                PROJECT_ROOT / second["summary_path"],
                {"sha256": second["summary_sha256"]},
            )
            summary = first["summary"]
            capacity = summary["cycles"] * summary["physical_pe_count"]
            fu_counts = summary["productive_pe_cycles_by_fu_class"]
            fma_utilization[key][backend] = fu_counts.get("fma", 0) / capacity
            checks = {
                "output": output_file["pass"],
                "first": first_file["pass"],
                "second": second_file["pass"],
                "replay": first["summary_sha256"] == second["summary_sha256"],
                "done": summary["done"] is True,
                "counts": summary["instructions_issued"]
                == summary["instructions_completed"]
                == sum(metadata["pipeline_counts"].values()),
                "pipelines": summary["issued_by_pipeline"]
                == metadata["pipeline_counts"],
                "fu_classes": expected_classes == set(fu_counts),
                "fu_bounds": all(0 < count <= capacity for count in fu_counts.values()),
                "fma_present": fu_counts.get("fma", 0) > 0,
            }
            if backend == "fixed":
                checks["memory"] = summary["external_memory_requests"] == 0
            else:
                adapter_file = qualify(
                    PROJECT_ROOT / first["adapter_path"],
                    {"sha256": first["adapter_sha256"]},
                )
                second_adapter_file = qualify(
                    PROJECT_ROOT / second["adapter_path"],
                    {"sha256": second["adapter_sha256"]},
                )
                adapter = first["adapter"]
                checks["adapter_files"] = adapter_file["pass"] and second_adapter_file[
                    "pass"
                ]
                checks["adapter_replay"] = first["adapter_sha256"] == second[
                    "adapter_sha256"
                ]
                checks["memory"] = (
                    summary["external_memory_requests"]
                    == summary["external_memory_completions"]
                    == adapter["requests"]
                    == adapter["responses"]
                    == metadata["memory_requests"]
                )
            execution_checks[f"{key}-{backend}"] = all(checks.values())
            outputs[key]["backends"][backend] = {
                "output": output_file,
                "first": first_file,
                "second": second_file,
                "summary": summary,
                "checks": checks,
            }
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    source_text = "\n".join(
        (PROJECT_ROOT / config["source_layout"][name]).read_text(encoding="utf-8")
        for name in ("compiler", "runner")
    )
    operator_spec = config["frozen_inputs"]["operators"]
    memory_spec = config["frozen_inputs"]["memory"]
    integrity_checks = {
        "frozen_files": all(item["pass"] for item in frozen_files.values()),
        "operator_integrity": frozen["operators"]["audit_integrity"]
        is operator_spec["required_integrity"],
        "memory_parent": frozen["memory"]["hypothesis_status"]
        == memory_spec["required_status"]
        and frozen["memory"]["audit_integrity"] is memory_spec["required_integrity"],
        "compile_manifest": compile_file["pass"]
        and compiler["record_count"] == 24
        and compiler["all_only_backend_changed"] is True,
        "run_manifest": run_file["pass"],
        "replays": all(run["checks"].values()),
        "executions": all(execution_checks.values()),
        "source_files": all(item["pass"] for item in source_files.values()),
        "target_paths_absent": "paper_targets" not in source_text
        and "fig25_roofline" not in source_text,
        "targets_consumed": compiler["paper_performance_targets_consumed"] is False
        and run["paper_performance_targets_consumed"] is False,
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
        "hypothesis_status": "supported" if audit_integrity else "rejected",
        "audit_integrity": audit_integrity,
        "frozen_inputs": frozen_files,
        "compile_manifest": compile_file,
        "run_manifest": run_file,
        "outputs": outputs,
        "execution_checks": execution_checks,
        "fma_utilization": fma_utilization,
        "source_files": source_files,
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
        keys = ("hypothesis_status", "audit_integrity", "integrity_checks")
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
