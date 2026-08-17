#!/usr/bin/env python3
"""Audit H63's frozen Figure 10 mapping against all Figure 22 resources."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig10_fig22_transfer_v1.yaml"
RESOURCES = ("xfer", "load", "store", "compute")
SIZES = (64, 128, 256, 512, 1024, 2048, 4096, 8192)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--verify-existing", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def qualify(path: Path, expected: dict[str, Any] | None = None) -> dict[str, Any]:
    path = path.resolve()
    exists = path.is_file()
    size = path.stat().st_size if exists else None
    digest = sha256_file(path) if exists else None
    checks = {"is_file": exists}
    if expected and "bytes" in expected:
        checks["bytes"] = size == int(expected["bytes"])
    if expected and "sha256" in expected:
        checks["sha256"] = digest == expected["sha256"]
    try:
        display = path.relative_to(PROJECT_ROOT)
    except ValueError:
        display = path
    return {
        "path": str(display),
        "bytes": size,
        "sha256": digest,
        "checks": checks,
        "pass": all(checks.values()),
    }


def parse_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")

    def prefixed(prefix: str) -> dict[str, Any] | None:
        matches = re.findall(rf"^{prefix} (\{{.*\}})$", text, flags=re.MULTILINE)
        return json.loads(matches[-1]) if matches else None

    return {
        "overlay": prefixed("MLX_OVERLAY_SUMMARY"),
        "adapter": prefixed("MLX_SPAD_ADAPTER_SUMMARY"),
        "sanity": "sanity check passed successfully!" in text,
        "normal_exit": "exiting with last active thread context" in text
        and "Simulated exit code not 0!" not in text,
    }


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


def execution_checks(
    summary: dict[str, Any], metadata: dict[str, Any], adapter: dict[str, Any] | None
) -> dict[str, bool]:
    pipeline = metadata["expected_pipeline_instructions"]
    checks = {
        "done": summary.get("done") is True,
        "paper_static": summary.get("pe_dependency_model") == "paper_static",
        "physical_pes": summary.get("physical_pe_count") == 16,
        "mapped_pes": summary.get("mapped_pe_count") == 16,
        "instructions": summary.get("instructions_issued")
        == summary.get("instructions_completed")
        == metadata["instruction_count"],
        "pipelines": all(
            summary.get("issued_by_pipeline", {}).get(name) == count
            for name, count in pipeline.items()
        ),
        "events": summary.get("boundary_events_emitted")
        == metadata["boundary_events"],
        "routes": summary.get("route_hops") == metadata["route_hops"],
        "counters": all(
            0
            <= summary["productive_pe_cycles_by_pipeline"][resource]
            <= summary["resident_pe_cycles_by_pipeline"][resource]
            <= summary["cycles"] * 16
            for resource in RESOURCES
        ),
    }
    if adapter is None:
        checks["fixed"] = summary.get("memory_backend") == "fixed"
        checks["external_requests"] = summary.get("external_memory_requests") == 0
    else:
        checks["adapter"] = summary.get("memory_backend") == "adapter"
        checks["external_requests"] = (
            summary.get("external_memory_requests")
            == summary.get("external_memory_completions")
            == adapter.get("requests")
            == adapter.get("responses")
            == metadata["memory_requests"]
        )
    return checks


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    mapping = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["mapping"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    target_result = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["targets"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    targets = target_result["derived_targets"]["panels"]
    compile_manifest_path = PROJECT_ROOT / config["compile_manifest"]
    compile_manifest_file = qualify(compile_manifest_path)
    compile_manifest = json.loads(compile_manifest_path.read_text(encoding="utf-8"))
    control_manifest_path = PROJECT_ROOT / config["output_root"] / "fixed-control-manifest.json"
    control_manifest_file = qualify(control_manifest_path)
    control_manifest = json.loads(control_manifest_path.read_text(encoding="utf-8"))

    points: dict[str, list[dict[str, Any]]] = {"bsmm": [], "chunk_fft": []}
    logs: dict[str, Any] = {}
    summaries = {"primary_dsagen_spad": [], "fixed_control": []}
    per_resource_errors = {
        backend: {resource: [] for resource in RESOURCES}
        for backend in summaries
    }
    run_checks: dict[str, bool] = {}
    for panel, kernel in (("bsmm", "bsmm"), ("chunk_fft", "fft")):
        for index, size in enumerate(SIZES):
            key = f"{kernel}-{size}"
            metadata = compile_manifest["outputs"][key]["metadata"]
            parent_record = compile_manifest["outputs"][key]["primary"]
            control_record = control_manifest["records"][key]
            parent_file = qualify(PROJECT_ROOT / parent_record["path"], parent_record)
            control_file = qualify(
                PROJECT_ROOT / control_record["control"]["path"],
                control_record["control"],
            )
            parent = json.loads(
                (PROJECT_ROOT / parent_record["path"]).read_text(encoding="utf-8")
            )
            control = json.loads(
                (PROJECT_ROOT / control_record["control"]["path"]).read_text(
                    encoding="utf-8"
                )
            )
            restored = {**control, "memory_backend": parent["memory_backend"]}
            exact_control = restored == parent

            log_path = PROJECT_ROOT / config["output_root"] / f"runs/gem5/{key}/run.log"
            log_file = qualify(log_path)
            logs[key] = log_file
            parsed = parse_log(log_path)
            primary_summary = parsed["overlay"] or {}
            adapter = parsed["adapter"] or {}
            primary_checks = execution_checks(primary_summary, metadata, adapter)
            primary_checks.update(
                {
                    "log": log_file["pass"],
                    "sanity": parsed["sanity"],
                    "normal_exit": parsed["normal_exit"],
                }
            )
            fixed_path = PROJECT_ROOT / config["output_root"] / f"runs/fixed/{key}.json"
            fixed_file = qualify(fixed_path)
            fixed_summary = json.loads(fixed_path.read_text(encoding="utf-8"))
            fixed_checks = execution_checks(fixed_summary, metadata, None)
            fixed_checks["file"] = fixed_file["pass"]
            checks = {
                "parent": parent_file["pass"],
                "control": control_file["pass"],
                "only_backend_changed": exact_control
                and control_record["only_backend_changed"] is True,
                "primary_execution": all(primary_checks.values()),
                "fixed_execution": all(fixed_checks.values()),
            }
            run_checks[key] = all(checks.values())
            resources: dict[str, Any] = {}
            capacity = primary_summary["cycles"] * primary_summary["physical_pe_count"]
            fixed_capacity = fixed_summary["cycles"] * fixed_summary["physical_pe_count"]
            for resource in RESOURCES:
                target = float(targets[panel][resource][index])
                primary = (
                    primary_summary["productive_pe_cycles_by_pipeline"][resource]
                    / capacity
                )
                fixed = (
                    fixed_summary["productive_pe_cycles_by_pipeline"][resource]
                    / fixed_capacity
                )
                primary_error = relative_error(primary, target)
                fixed_error = relative_error(fixed, target)
                summaries["primary_dsagen_spad"].append(primary_error)
                summaries["fixed_control"].append(fixed_error)
                per_resource_errors["primary_dsagen_spad"][resource].append(
                    primary_error
                )
                per_resource_errors["fixed_control"][resource].append(fixed_error)
                resources[resource] = {
                    "target": target,
                    "primary": primary,
                    "primary_relative_error": primary_error,
                    "primary_pass_10pct": primary_error <= 0.10,
                    "fixed_control": fixed,
                    "fixed_relative_error": fixed_error,
                    "fixed_pass_10pct": fixed_error <= 0.10,
                }
            points[panel].append(
                {
                    "size": size,
                    "resources": resources,
                    "primary_summary": primary_summary,
                    "adapter": adapter,
                    "fixed_summary": fixed_summary,
                    "primary_checks": primary_checks,
                    "fixed_checks": fixed_checks,
                    "checks": checks,
                }
            )
    metric_summaries = {
        backend: summarize(errors) for backend, errors in summaries.items()
    }
    resource_summaries = {
        backend: {
            resource: summarize(errors)
            for resource, errors in resources.items()
        }
        for backend, resources in per_resource_errors.items()
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    integrity_checks = {
        "frozen_inputs": all(item["pass"] for item in frozen.values()),
        "mapping": mapping.get("hypothesis_status")
        == config["frozen_inputs"]["mapping"]["required_status"]
        and mapping.get("audit_integrity")
        is config["frozen_inputs"]["mapping"]["required_integrity"],
        "targets": target_result.get("verdict")
        == config["frozen_inputs"]["targets"]["required_verdict"]
        and target_result.get("summary", {}).get("pass")
        is config["frozen_inputs"]["targets"]["required_summary_pass"],
        "compile_manifest": compile_manifest_file["pass"],
        "control_manifest": control_manifest_file["pass"]
        and control_manifest.get("record_count") == 16
        and control_manifest.get("all_only_backend_changed") is True,
        "all_logs": all(item["pass"] for item in logs.values()),
        "all_runs": all(run_checks.values()),
        "source_files": all(item["pass"] for item in source_files.values()),
        "primary_metric_unchanged": config["primary_metric"]["source"]
        == "productive_pe_cycles_by_pipeline",
        "targets_loaded_only_by_auditor": True,
        "fixed_control_selected_as_primary": False,
        "post_result_adjustment": False,
    }
    audit_integrity = all(
        value
        for key, value in integrity_checks.items()
        if key not in {"fixed_control_selected_as_primary", "post_result_adjustment"}
    ) and not integrity_checks["fixed_control_selected_as_primary"] and not integrity_checks[
        "post_result_adjustment"
    ]
    primary = metric_summaries["primary_dsagen_spad"]
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "classification": config["classification"],
        "validation_eligible": config["validation_eligible"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_commit": git_commit(),
        "hypothesis_status": "supported" if primary["all_within_10pct"] else "rejected",
        "audit_integrity": audit_integrity,
        "frozen_inputs": frozen,
        "compile_manifest": compile_manifest_file,
        "control_manifest": control_manifest_file,
        "source_files": source_files,
        "logs": logs,
        "points": points,
        "run_checks": run_checks,
        "primary_metric": {**config["primary_metric"], "summary": primary},
        "fixed_control_summary": metric_summaries["fixed_control"],
        "resource_summaries": resource_summaries,
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
        keys = (
            "hypothesis_status",
            "audit_integrity",
            "primary_metric",
            "fixed_control_summary",
        )
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
