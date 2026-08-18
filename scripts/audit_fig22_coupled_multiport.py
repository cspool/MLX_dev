#!/usr/bin/env python3
"""Audit H120's target-free ported live-memory mechanism."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from mlxsim.dsagen_overlay import canonical_json
from mlxsim.fig22_coupled_multiport import compile_fig22_coupled_multiport

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig22_coupled_multiport_v1.yaml"
RESOURCES = ("compute", "load", "store", "xfer")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={PROJECT_ROOT}", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def patch_audit(config: dict[str, Any]) -> dict[str, Any]:
    patch = PROJECT_ROOT / config["source_layout"]["patch"]
    header = PROJECT_ROOT / config["source_layout"]["memory_header"]
    source = PROJECT_ROOT / config["source_layout"]["memory_source"]
    control = config["patch_control"]
    with tempfile.TemporaryDirectory(prefix="h120-patch-") as directory:
        root = Path(directory)
        target = root / "simulator_ext/dsagen"
        target.mkdir(parents=True)
        temporary_header = target / header.name
        temporary_source = target / source.name
        shutil.copy2(header, temporary_header)
        shutil.copy2(source, temporary_source)
        reverse = subprocess.run(
            ["git", "apply", "--unidiff-zero", "--reverse", str(patch)],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        before_header = (
            sha256_file(temporary_header) if reverse.returncode == 0 else None
        )
        before_source = (
            sha256_file(temporary_source) if reverse.returncode == 0 else None
        )
        forward = subprocess.run(
            ["git", "apply", "--unidiff-zero", "--check", str(patch)],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    checks = {
        "patch": patch.is_file(),
        "after_header": sha256_file(header) == control["header_after_sha256"],
        "after_source": sha256_file(source) == control["source_after_sha256"],
        "reverse": reverse.returncode == 0,
        "before_header": before_header == control["header_before_sha256"],
        "before_source": before_source == control["source_before_sha256"],
        "forward": forward.returncode == 0,
    }
    return {
        "patch": qualify(patch),
        "before_header_sha256": before_header,
        "before_source_sha256": before_source,
        "reverse_stderr": reverse.stderr.strip(),
        "forward_stderr": forward.stderr.strip(),
        "checks": checks,
        "pass": all(checks.values()),
    }


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    parents = {
        name: json.loads((PROJECT_ROOT / spec["path"]).read_text())
        for name, spec in config["frozen_inputs"].items()
        if name != "h118_config"
    }
    parent_checks = {
        name: parent["hypothesis_status"] == spec["required_status"]
        and parent["audit_integrity"] is spec["required_integrity"]
        for name, spec in config["frozen_inputs"].items()
        if name != "h118_config"
        for parent in [parents[name]]
    }
    h118 = parents["h118"]
    h118_config = yaml.safe_load(
        (PROJECT_ROOT / config["frozen_inputs"]["h118_config"]["path"]).read_text()
    )
    h118_run_path = PROJECT_ROOT / h118["run_manifest"]["path"]
    h118_run_file = qualify(h118_run_path, h118["run_manifest"])
    h118_run = json.loads(h118_run_path.read_text())
    h118_compile = json.loads(
        (PROJECT_ROOT / h118["compile_manifest"]["path"]).read_text()
    )
    baseline_records = {
        item["key"]: item
        for item in h118_run["records"]
        if item["mode"] == "optimized" and int(item["replay"]) == 1
    }

    output_root = PROJECT_ROOT / config["output_root"]
    compile_path = output_root / "fig22-coupled-multiport-compile-manifest.json"
    run_path = output_root / "fig22-coupled-multiport-run-manifest.json"
    compile_file = qualify(compile_path)
    run_file = qualify(run_path)
    compiled = json.loads(compile_path.read_text())
    run = json.loads(run_path.read_text())
    records = {
        (item["key"], item["mode"], int(item["replay"])): item
        for item in run["records"]
    }

    compile_checks: dict[str, bool] = {}
    partition_checks: dict[str, bool] = {}
    execution_checks: dict[str, bool] = {}
    work_checks: dict[str, bool] = {}
    port_checks: dict[str, bool] = {}
    counter_checks: dict[str, bool] = {}
    comparisons: dict[str, Any] = {}
    measurements: dict[str, Any] = {}
    record_checks: dict[str, bool] = {}
    for key, item in compiled["outputs"].items():
        operator, size_text = key.split("-", 1)
        overlay, memory_document, metadata = compile_fig22_coupled_multiport(
            operator, int(size_text), config, h118_config
        )
        overlay_path = PROJECT_ROOT / item["overlay"]["path"]
        memory_path = PROJECT_ROOT / item["memory"]["path"]
        compile_checks[key] = (
            qualify(overlay_path, item["overlay"])["pass"]
            and qualify(memory_path, item["memory"])["pass"]
            and overlay_path.read_text() == canonical_json(overlay)
            and memory_path.read_text() == canonical_json(memory_document)
            and item["metadata"] == metadata
            and all(metadata["checks"].values())
        )
        parent = metadata["parent"]
        h118_overlay_path = (
            PROJECT_ROOT / h118_compile["outputs"][key]["overlay"]["path"]
        )
        partition_checks[key] = (
            metadata["ports"] == 4
            and metadata["total_banks"] == config["candidate"]["total_banks"]
            and metadata["total_issue_width"]
            == config["candidate"]["total_issue_width"]
            and memory_document["spad"]["request_buffer_entries"]
            == h118_config["hardware"]["spm"]["request_buffer_entries"]
            and memory_document["spad"]["bank_width_bytes"]
            == h118_config["hardware"]["spm"]["bank_width_bytes"]
            and overlay_path.read_bytes() == h118_overlay_path.read_bytes()
        )

        record = records[(key, "optimized", 1)]
        summary = record["summary"]
        overlay_summary = summary["overlay"]
        memory = summary["memory"]
        baseline = baseline_records[key]["summary"]
        record_checks[key] = qualify(
            PROJECT_ROOT / record["summary_path"],
            {"sha256": record["summary_sha256"]},
        )["pass"]
        execution_checks[key] = (
            overlay_summary["done"] is True
            and memory["idle"] is True
            and overlay_summary["pe_dependency_model"] == "dpu_pipelined"
            and overlay_summary["memory_backend"] == "dpu_memory"
            and summary["end_to_end_cycles"] > 0
        )
        work_checks[key] = (
            overlay_summary["instructions_issued"]
            == overlay_summary["instructions_completed"]
            == parent["dynamic_instruction_count"]
            and overlay_summary["issued_by_pipeline"]
            == parent["expected_pipeline_instructions"]
            and overlay_summary["boundary_events_emitted"]
            == parent["boundary_events"]
            and overlay_summary["route_hops"] == parent["route_hops"]
            and overlay_summary["external_memory_requests"]
            == overlay_summary["external_memory_completions"]
            == memory["requests"]
            == memory["responses"]
            == parent["memory_requests"]
            and memory["offchip_read_bytes"] == parent["input_bytes"]
            and memory["offchip_write_bytes"] == parent["output_bytes"]
            and memory["released_tiles"] == memory["drained_tiles"] == 1
            and memory["ownership_violations"] == 0
        )
        spad = memory["spad"]
        per_port = spad["per_port"]
        port_checks[key] = (
            spad["ports"] == 4
            and spad["axis"] == metadata["axis"]
            and len(per_port) == 4
            and spad["requests"] == sum(port["requests"] for port in per_port)
            and spad["responses"] == sum(port["responses"] for port in per_port)
            and spad["requests"] == spad["responses"] == memory["requests"]
            and all(port["requests"] > 0 for port in per_port)
        )
        productive = overlay_summary["productive_pe_cycles_by_pipeline"]
        resident = overlay_summary["resident_pe_cycles_by_pipeline"]
        capacity = overlay_summary["cycles"] * 16
        counter_checks[key] = all(
            0 <= productive[name] <= resident[name] <= capacity
            for name in RESOURCES
        )
        queue_unavailable = sum(port["unavailable_checks"] for port in per_port)
        baseline_unavailable = baseline["memory"]["spad"]["unavailable_checks"]
        comparisons[key] = {
            "baseline_end_to_end_cycles": baseline["end_to_end_cycles"],
            "ported_end_to_end_cycles": summary["end_to_end_cycles"],
            "cycle_speedup": baseline["end_to_end_cycles"]
            / summary["end_to_end_cycles"],
            "baseline_overlay_cycles": baseline["overlay_cycles"],
            "ported_overlay_cycles": summary["overlay_cycles"],
            "baseline_queue_unavailable_checks": baseline_unavailable,
            "ported_queue_unavailable_checks": queue_unavailable,
            "end_to_end_non_regression": summary["end_to_end_cycles"]
            <= baseline["end_to_end_cycles"],
            "overlay_non_regression": summary["overlay_cycles"]
            <= baseline["overlay_cycles"],
            "queue_non_regression": queue_unavailable <= baseline_unavailable,
            "strict_cycle_improvement": summary["end_to_end_cycles"]
            < baseline["end_to_end_cycles"],
        }
        primary_capacity = summary["end_to_end_cycles"] * 16
        measurements[key] = {
            "operator": operator,
            "size": int(size_text),
            "end_to_end_cycles": summary["end_to_end_cycles"],
            "overlay_cycles": summary["overlay_cycles"],
            "primary_end_to_end_utilization": {
                name: productive[name] / primary_capacity for name in RESOURCES
            },
            "diagnostic_overlay_utilization": {
                name: productive[name] / capacity for name in RESOURCES
            },
        }

    utilization_checks = {
        key: all(
            math.isfinite(value) and 0 <= value <= 1
            for group in (
                item["primary_end_to_end_utilization"],
                item["diagnostic_overlay_utilization"],
            )
            for value in group.values()
        )
        for key, item in measurements.items()
    }
    comparison_checks = {
        "cycles": all(
            item["end_to_end_non_regression"]
            and item["overlay_non_regression"]
            for item in comparisons.values()
        ),
        "strict": any(
            item["strict_cycle_improvement"] for item in comparisons.values()
        ),
        "queue": all(item["queue_non_regression"] for item in comparisons.values()),
    }
    patch = patch_audit(config)
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    source_text = "\n".join(
        (PROJECT_ROOT / path).read_text(errors="replace")
        for name, path in config["source_layout"].items()
        if name not in {"memory_header", "memory_source", "driver", "patch"}
    )
    forbidden = (
        "fig22-resource" + "-targets-run065.json",
        "fig22-coupled" + "-transfer-run124.json",
    )
    target_free_checks = {
        "source": not any(name in source_text for name in forbidden),
        "compile": compiled["paper_performance_targets_consumed"] is False,
        "run": run["paper_performance_targets_consumed"] is False,
        "parents": not any(name in spec["path"] for name in forbidden for spec in config["frozen_inputs"].values()),
    }
    counts = {
        "outputs": len(compiled["outputs"]) == 16,
        "records": len(run["records"]) == int(config["execution"]["required_executions"]),
        "optimized": sum(item["mode"] == "optimized" for item in run["records"])
        == int(config["execution"]["required_optimized_executions"]),
        "sanitizers": sum(item["mode"] in {"asan", "ubsan"} for item in run["records"])
        == int(config["execution"]["required_sanitizer_executions"]),
        "measurements": len(measurements) == 16,
    }
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        all(compile_checks.values()) and all(counts.values()),
        all(partition_checks.values()),
        patch["pass"] and run["checks"]["regressions"] is True,
        all(run["checks"].values())
        and all(run["replay_checks"].values())
        and all(run["sanitizer_checks"].values()),
        all(work_checks.values()),
        all(port_checks.values()),
        comparison_checks["cycles"] and comparison_checks["strict"],
        comparison_checks["queue"],
        all(counter_checks.values()) and all(utilization_checks.values()),
        all(target_free_checks.values()) and all(item["pass"] for item in source_files.values()),
        config["validation_eligible"] is False,
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "h118_run": h118_run_file["pass"],
        "compile_manifest": compile_file["pass"] and all(compiled["checks"].values()),
        "run_manifest": run_file["pass"] and all(run["checks"].values()),
        "compile": all(compile_checks.values()),
        "execution": all(execution_checks.values()),
        "work": all(work_checks.values()),
        "ports": all(port_checks.values()),
        "records": all(record_checks.values()),
        "counters": all(counter_checks.values()),
        "counts": all(counts.values()),
        "patch": patch["pass"],
        "target_free": all(target_free_checks.values()),
        "source_files": all(item["pass"] for item in source_files.values()),
        "acceptance_evaluated": len(acceptance_gates) == 12
        and all(isinstance(value, bool) for value in acceptance_gates),
    }
    integrity = all(integrity_checks.values())
    supported = integrity and all(acceptance_gates)
    speedups = [item["cycle_speedup"] for item in comparisons.values()]
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "classification": config["classification"],
        "validation_eligible": config["validation_eligible"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_commit": git_commit(),
        "hypothesis_status": "supported" if supported else "rejected",
        "audit_integrity": integrity,
        "paper_performance_targets_consumed": False,
        "paper_reproduction_claim": "none_target_free_multiport_mechanism_only",
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "compile_manifest": compile_file,
        "run_manifest": run_file,
        "compile_checks": compile_checks,
        "partition_checks": partition_checks,
        "execution_checks": execution_checks,
        "work_checks": work_checks,
        "port_checks": port_checks,
        "counter_checks": counter_checks,
        "record_checks": record_checks,
        "comparisons": comparisons,
        "comparison_checks": comparison_checks,
        "measurements": measurements,
        "utilization_checks": utilization_checks,
        "patch_checks": patch,
        "target_free_checks": target_free_checks,
        "counts": counts,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "paths": len(measurements),
            "executions": len(run["records"]),
            "sanitizer_executions": sum(item["mode"] in {"asan", "ubsan"} for item in run["records"]),
            "cycle_speedup_minimum": min(speedups),
            "cycle_speedup_maximum": max(speedups),
            "strictly_improved_paths": sum(item["strict_cycle_improvement"] for item in comparisons.values()),
            "queue_non_regression_paths": sum(item["queue_non_regression"] for item in comparisons.values()),
            "acceptance_gates_passed": sum(acceptance_gates),
            "acceptance_gates_total": len(acceptance_gates),
            "active_simulator_figures_reproduced": 0,
            "active_simulator_figures_total": 8,
        },
        "source_files": source_files,
        "integrity_checks": integrity_checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    report = build_audit(config)
    output = PROJECT_ROOT / config["result_path"]
    if args.verify_existing:
        existing = json.loads(output.read_text())
        keys = (
            "hypothesis_status",
            "audit_integrity",
            "comparisons",
            "measurements",
            "acceptance_gates",
            "summary",
            "integrity_checks",
        )
        matches = all(
            json.dumps(existing.get(key), sort_keys=True)
            == json.dumps(report.get(key), sort_keys=True)
            for key in keys
        )
        print(json.dumps({"existing_matches": matches, **report}, indent=2))
        return 0 if matches else 1
    if args.preflight_only:
        print(json.dumps(report, indent=2))
        return 0 if report["audit_integrity"] and not output.exists() else 1
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["hypothesis_status"], **report["summary"]}, indent=2))
    return 0 if report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
