#!/usr/bin/env python3
"""Audit H67's target-free DSAGEN-memory scalability mechanism."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/spad_scalability_v1.yaml"


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
    h64 = frozen["scaling"]
    output_root = PROJECT_ROOT / config["output_root"]
    compile_path = output_root / "spad-scalability-compile-manifest.json"
    run_path = output_root / "spad-scalability-run-manifest.json"
    compile_file = qualify(compile_path)
    run_file = qualify(run_path)
    compiler = json.loads(compile_path.read_text(encoding="utf-8"))
    run = json.loads(run_path.read_text(encoding="utf-8"))
    parent_manifest = json.loads(
        (PROJECT_ROOT / config["compile_manifest"]).read_text(encoding="utf-8")
    )
    outputs: dict[str, Any] = {}
    execution_checks: dict[str, bool] = {}
    slowdowns: dict[str, dict[str, float]] = {}
    for key, record in compiler["records"].items():
        parent_file = qualify(PROJECT_ROOT / record["parent"]["path"], record["parent"])
        output_file = qualify(PROJECT_ROOT / record["output"]["path"], record["output"])
        parent = json.loads(
            (PROJECT_ROOT / record["parent"]["path"]).read_text(encoding="utf-8")
        )
        output = json.loads(
            (PROJECT_ROOT / record["output"]["path"]).read_text(encoding="utf-8")
        )
        restored = {**output, "memory_backend": parent["memory_backend"]}
        first = run["runs"][key]["first"]
        second = run["runs"][key]["second"]
        first_file = qualify(
            PROJECT_ROOT / first["summary_path"], {"sha256": first["summary_sha256"]}
        )
        second_file = qualify(
            PROJECT_ROOT / second["summary_path"],
            {"sha256": second["summary_sha256"]},
        )
        first_adapter = qualify(
            PROJECT_ROOT / first["adapter_path"], {"sha256": first["adapter_sha256"]}
        )
        second_adapter = qualify(
            PROJECT_ROOT / second["adapter_path"],
            {"sha256": second["adapter_sha256"]},
        )
        summary = first["summary"]
        adapter = first["adapter"]
        metadata = parent_manifest["outputs"][key]["metadata"]
        fixed_cycles = h64["cycles"][key.split("-", 1)[0]][key.split("-", 1)[1]]
        sequence, hardware = key.split("-", 1)
        slowdowns.setdefault(sequence, {})[hardware] = summary["cycles"] / fixed_cycles
        checks = {
            "parent": parent_file["pass"],
            "output": output_file["pass"],
            "only_backend": restored == parent and record["only_backend_changed"],
            "first": first_file["pass"] and first_adapter["pass"],
            "second": second_file["pass"] and second_adapter["pass"],
            "replay": first["summary_sha256"] == second["summary_sha256"]
            and first["adapter_sha256"] == second["adapter_sha256"],
            "done": summary["done"] is True,
            "counts": summary["instructions_issued"]
            == summary["instructions_completed"]
            == metadata["instruction_count"],
            "events": summary["boundary_events_emitted"]
            == metadata["boundary_events"],
            "routes": summary["route_hops"] == metadata["route_hops"],
            "memory": summary["external_memory_requests"]
            == summary["external_memory_completions"]
            == adapter["requests"]
            == adapter["responses"]
            == metadata["memory_requests"],
            "slower_than_fixed": summary["cycles"] >= fixed_cycles,
        }
        execution_checks[key] = all(checks.values())
        outputs[key] = {
            "parent": parent_file,
            "output": output_file,
            "first": first_file,
            "second": second_file,
            "first_adapter": first_adapter,
            "second_adapter": second_adapter,
            "checks": checks,
        }
    sequences = [str(value) for value in (512, 1024, 2048, 4096, 8192)]
    monotonic = {
        hardware: all(
            run["cycles"][right][hardware] > run["cycles"][left][hardware]
            for left, right in pairwise(sequences)
        )
        for hardware in ("baseline", "simd32_4x4", "simd8_8x8", "simd32_8x8")
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    source_text = "\n".join(
        (PROJECT_ROOT / config["source_layout"][name]).read_text(encoding="utf-8")
        for name in ("transformer", "runner")
    )
    integrity_checks = {
        "frozen_files": all(item["pass"] for item in frozen_files.values()),
        "frozen_results": all(
            frozen[name]["hypothesis_status"]
            == config["frozen_inputs"][name]["required_status"]
            and frozen[name]["audit_integrity"]
            is config["frozen_inputs"][name]["required_integrity"]
            for name in frozen
        ),
        "compile_manifest": compile_file["pass"]
        and compiler["record_count"] == 20
        and compiler["all_only_backend_changed"] is True,
        "run_manifest": run_file["pass"],
        "replays": all(run["checks"].values()),
        "executions": all(execution_checks.values()),
        "monotonic": all(monotonic.values()),
        "source_files": all(item["pass"] for item in source_files.values()),
        "target_paths_absent": "paper_targets" not in source_text
        and "fig23_scalability" not in source_text,
        "targets_consumed": compiler["paper_performance_targets_consumed"] is False
        and run["paper_performance_targets_consumed"] is False,
        "numerical_target_comparison_performed": False,
    }
    audit_integrity = all(
        value
        for key, value in integrity_checks.items()
        if key != "numerical_target_comparison_performed"
    ) and not integrity_checks["numerical_target_comparison_performed"]
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
        "cycles": run["cycles"],
        "speedups": run["speedups"],
        "fixed_to_spad_slowdowns": slowdowns,
        "monotonic": monotonic,
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
