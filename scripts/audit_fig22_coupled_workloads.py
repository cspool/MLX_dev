#!/usr/bin/env python3
"""Audit H118's exact target-free coupled Figure 22 workload executions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml

from mlxsim.dsagen_overlay import canonical_json
from mlxsim.fig22_coupled_workloads import compile_fig22_coupled_workload

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig22_coupled_workloads_v1.yaml"
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


def reverse_patch_check(spec: dict[str, Any]) -> dict[str, Any]:
    source = PROJECT_ROOT / spec["path"]
    patch = PROJECT_ROOT / spec["patch"]
    apply_root = (PROJECT_ROOT / spec["apply_root"]).resolve()
    relative_source = source.resolve().relative_to(apply_root)
    current = sha256_file(source)
    descendant_checks: dict[str, bool] = {}
    with tempfile.TemporaryDirectory(prefix="h118-patch-") as directory:
        temporary_root = Path(directory)
        temporary_source = temporary_root / relative_source
        temporary_source.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, temporary_source)
        descendants = list(spec.get("descendant_patches", []))
        if (
            not descendants
            and Path(spec["path"]).name == "mlx_overlay.cc"
            and Path(spec["patch"]).name
            == "dsa-gem5-active-window-instruction-capacity-v1.patch"
        ):
            descendants = [
                "patches/dsagen/dsa-gem5-mlx-physical-timing-v1.patch",
                "patches/dsagen/dsa-gem5-mlx-latency-service-v1.patch",
                "patches/dsagen/dsa-gem5-functional-payload-v1.patch"
            ]
        for descendant in descendants:
            descendant_path = PROJECT_ROOT / descendant
            descendant_result = subprocess.run(
                [
                    "git",
                    "apply",
                    "--reverse",
                    f"--include={relative_source}",
                    str(descendant_path),
                ],
                cwd=temporary_root,
                capture_output=True,
                text=True,
                check=False,
            )
            descendant_checks[descendant_path.name] = (
                descendant_result.returncode == 0
            )
            if descendant_result.returncode != 0:
                break
        after_descendants = sha256_file(temporary_source)
        result = subprocess.run(
            ["git", "apply", "--reverse", str(patch)],
            cwd=temporary_root,
            capture_output=True,
            text=True,
            check=False,
        )
        reversed_digest = (
            sha256_file(temporary_source) if result.returncode == 0 else None
        )
    checks = {
        "current": current == spec.get("current_sha256", current),
        "descendants": all(descendant_checks.values()),
        "after": after_descendants == spec["after_sha256"],
        "reverse_apply": result.returncode == 0,
        "before": reversed_digest == spec["before_sha256"],
    }
    return {
        "source": str(source.relative_to(PROJECT_ROOT)),
        "patch": str(patch.relative_to(PROJECT_ROOT)),
        "current_sha256": current,
        "after_descendants_sha256": after_descendants,
        "descendant_checks": descendant_checks,
        "reversed_sha256": reversed_digest,
        "stderr": result.stderr.strip(),
        "checks": checks,
        "pass": all(checks.values()),
    }


def historical_memory_descendant_check(
    h114_sources: dict[str, Any],
) -> dict[str, bool]:
    patch = (
        PROJECT_ROOT
        / "patches/dsagen/dsa-gem5-historical-multiport-spad-v1.patch"
    )
    current_header = PROJECT_ROOT / h114_sources["adapter_header"]["path"]
    current_source = PROJECT_ROOT / h114_sources["adapter_source"]["path"]
    with tempfile.TemporaryDirectory(prefix="h118-descendant-") as directory:
        root = Path(directory)
        target = root / "simulator_ext/dsagen"
        target.mkdir(parents=True)
        header = target / "historical_dpu_memory.hh"
        source = target / "historical_dpu_memory.cc"
        shutil.copy2(current_header, header)
        shutil.copy2(current_source, source)
        reverse = subprocess.run(
            ["git", "apply", "--unidiff-zero", "--reverse", str(patch)],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        return {
            "header": reverse.returncode == 0
            and sha256_file(header) == h114_sources["adapter_header"]["sha256"],
            "source": reverse.returncode == 0
            and sha256_file(source) == h114_sources["adapter_source"]["sha256"],
        }


def functional_overlay_descendant_check(
    h114_sources: dict[str, Any],
) -> dict[str, bool]:
    patch = PROJECT_ROOT / "patches/dsagen/dsa-gem5-functional-payload-v1.patch"
    current_header = PROJECT_ROOT / h114_sources["overlay_header"]["path"]
    current_source = PROJECT_ROOT / h114_sources["overlay_source"]["path"]
    with tempfile.TemporaryDirectory(prefix="h118-functional-descendant-") as directory:
        root = Path(directory)
        target = root / "src/cpu/minor/ssim"
        target.mkdir(parents=True)
        header = target / "mlx_overlay.hh"
        source = target / "mlx_overlay.cc"
        shutil.copy2(current_header, header)
        shutil.copy2(current_source, source)
        reverse = subprocess.run(
            ["git", "apply", "--reverse", str(patch)],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        return {
            "header": reverse.returncode == 0
            and sha256_file(header) == h114_sources["overlay_header"]["sha256"],
            "source": reverse.returncode == 0
            and sha256_file(source) == h114_sources["overlay_source"]["sha256"],
        }
def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    parents = {
        name: json.loads((PROJECT_ROOT / spec["path"]).read_text())
        for name, spec in config["frozen_inputs"].items()
        if name != "h62_compile"
    }
    parent_checks = {
        name: parent["hypothesis_status"] == spec["required_status"]
        and parent["audit_integrity"] is spec["required_integrity"]
        for name, spec in config["frozen_inputs"].items()
        if name != "h62_compile"
        for parent in [parents[name]]
    }
    h117_nonresidence = [
        holdout["pass_5pct"]
        for path in parents["h117"]["models"].values()
        for metric_name, metric in path["metrics"].items()
        if metric_name != "productive_fma_pe_cycles"
        for holdout in metric["holdouts"]
    ]
    parent_checks["h117_nonresidence_80_of_80"] = (
        len(h117_nonresidence) == 80 and all(h117_nonresidence)
    )

    output_root = PROJECT_ROOT / config["output_root"]
    compile_path = output_root / "fig22-coupled-compile-manifest.json"
    run_path = output_root / "fig22-coupled-run-manifest.json"
    compile_file = qualify(compile_path)
    run_file = qualify(run_path)
    compiled = json.loads(compile_path.read_text())
    run = json.loads(run_path.read_text())
    h62_manifest = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h62_compile"]["path"]).read_text()
    )

    compile_checks: dict[str, bool] = {}
    source_replay_checks: dict[str, bool] = {}
    structure_checks: dict[str, bool] = {}
    hardware_checks: dict[str, bool] = {}
    memory_contract_checks: dict[str, bool] = {}
    for key, item in compiled["outputs"].items():
        operator, size_text = key.split("-", 1)
        overlay, memory, metadata, source = compile_fig22_coupled_workload(
            operator, int(size_text), config
        )
        overlay_path = PROJECT_ROOT / item["overlay"]["path"]
        memory_path = PROJECT_ROOT / item["memory"]["path"]
        parent_path = PROJECT_ROOT / item["parent"]["path"]
        compile_checks[key] = (
            qualify(overlay_path, item["overlay"])["pass"]
            and qualify(memory_path, item["memory"])["pass"]
            and overlay_path.read_text() == canonical_json(overlay)
            and memory_path.read_text() == canonical_json(memory)
            and item["metadata"] == metadata
            and all(metadata["checks"].values())
        )
        source_record = h62_manifest["outputs"][key]["primary"]
        source_replay_checks[key] = (
            qualify(parent_path, source_record)["pass"]
            and parent_path.read_text() == canonical_json(source)
            and compiled["source_checks"][key] is True
        )
        structure_checks[key] = (
            overlay["blocks"] == source["blocks"]
            and overlay["pipelines"] == source["pipelines"]
            and overlay["functional_units"] == source["functional_units"]
            and overlay["routing"] == source["routing"]
            and overlay["register_file"] == source["register_file"]
            and all(
                overlay["metadata"].get(name) == value
                for name, value in source["metadata"].items()
            )
            and metadata["dynamic_instruction_count"]
            == source["metadata"]["instruction_count"]
        )
        dpu = overlay["dpu"]
        hardware = config["hardware"]
        hardware_checks[key] = (
            overlay["pe_dependency_model"] == "dpu_pipelined"
            and overlay["memory_backend"] == "dpu_memory"
            and overlay["active_window"] == int(hardware["active_window"])
            and source["metadata"]["mesh"] == hardware["mesh"]
            and source["metadata"]["simd_width"] == int(hardware["simd_width"])
            and dpu["instruction_slots_per_pe"]
            == int(hardware["instruction_slots_per_pe"])
            and dpu["active_blocks_per_pe"]
            == int(hardware["active_blocks_per_pe"])
            and dpu["operand_contexts_per_pe"]
            == int(hardware["operand_contexts_per_pe"])
            and dpu["iteration_contexts_per_block"]
            == int(hardware["iteration_contexts_per_block"])
            and metadata["max_active_instruction_footprint_per_pe"]
            <= int(hardware["instruction_slots_per_pe"])
        )
        size = int(size_text)
        memory_contract_checks[key] = (
            memory["tile_count"] == 1
            and memory["input_bytes_per_tile"]
            == int(hardware["input_vectors_per_output"])
            * size
            * int(hardware["vector_bytes"])
            and memory["output_bytes_per_tile"]
            == int(hardware["output_vectors_per_output"])
            * size
            * int(hardware["vector_bytes"])
            and memory["stores_per_tile"] == metadata["external_stores"]
            and memory["input_bytes_per_tile"]
            <= int(hardware["spm_bytes"]) // int(hardware["buffer_halves"])
            and memory["output_bytes_per_tile"]
            <= int(hardware["spm_bytes"]) // int(hardware["buffer_halves"])
            and metadata["checks"]["request_addresses"] is True
        )

    records = {
        (item["key"], item["mode"], int(item["replay"])): item
        for item in run["records"]
    }
    execution_checks: dict[str, bool] = {}
    work_checks: dict[str, bool] = {}
    memory_checks: dict[str, bool] = {}
    counter_checks: dict[str, bool] = {}
    record_checks: dict[str, bool] = {}
    measurements: dict[str, Any] = {}
    for key, item in compiled["outputs"].items():
        record = records[(key, "optimized", 1)]
        summary = record["summary"]
        overlay = summary["overlay"]
        memory = summary["memory"]
        metadata = item["metadata"]
        record_checks[key] = qualify(
            PROJECT_ROOT / record["summary_path"],
            {"sha256": record["summary_sha256"]},
        )["pass"]
        execution_checks[key] = (
            overlay["done"] is True
            and memory["idle"] is True
            and summary["end_to_end_cycles"] > 0
            and summary["overlay_cycles"] == overlay["cycles"]
            and summary["end_to_end_cycles"] >= summary["overlay_cycles"]
            and overlay["memory_backend"] == "dpu_memory"
            and overlay["pe_dependency_model"] == "dpu_pipelined"
            and overlay["physical_pe_count"]
            == overlay["mapped_pe_count"]
            == int(config["hardware"]["physical_pes"])
            and overlay["iteration_contexts_per_block"]
            == int(config["hardware"]["iteration_contexts_per_block"])
            and overlay["max_active_tags"] <= int(config["hardware"]["active_window"])
            and overlay["max_active_blocks_per_pe"]
            <= int(config["hardware"]["active_blocks_per_pe"])
        )
        work_checks[key] = (
            overlay["instructions_issued"]
            == overlay["instructions_completed"]
            == metadata["dynamic_instruction_count"]
            and overlay["issued_by_pipeline"]
            == metadata["expected_pipeline_instructions"]
            and overlay["boundary_events_emitted"] == metadata["boundary_events"]
            and overlay["route_hops"] == metadata["route_hops"]
        )
        memory_checks[key] = (
            overlay["external_memory_requests"]
            == overlay["external_memory_completions"]
            == memory["requests"]
            == memory["responses"]
            == metadata["memory_requests"]
            and memory["read_requests"] == metadata["external_loads"]
            and memory["write_requests"] == metadata["external_stores"]
            and memory["released_tiles"] == memory["drained_tiles"] == 1
            and memory["offchip_read_bytes"] == metadata["input_bytes"]
            and memory["offchip_write_bytes"] == metadata["output_bytes"]
            and memory["ownership_violations"] == 0
        )
        overlay_capacity = overlay["cycles"] * int(
            config["hardware"]["physical_pes"]
        )
        productive = overlay["productive_pe_cycles_by_pipeline"]
        resident = overlay["resident_pe_cycles_by_pipeline"]
        counter_checks[key] = all(
            0 <= int(productive[name]) <= int(resident[name]) <= overlay_capacity
            for name in RESOURCES
        ) and all(int(productive[name]) > 0 for name in RESOURCES)
        primary_capacity = summary["end_to_end_cycles"] * int(
            config["hardware"]["physical_pes"]
        )
        primary = {
            name: int(productive[name]) / primary_capacity for name in RESOURCES
        }
        diagnostic = {
            name: int(productive[name]) / overlay_capacity for name in RESOURCES
        }
        measurements[key] = {
            "operator": metadata["operator"],
            "size": metadata["size"],
            "overlay_cycles": summary["overlay_cycles"],
            "end_to_end_cycles": summary["end_to_end_cycles"],
            "productive_pe_cycles": {
                name: int(productive[name]) for name in RESOURCES
            },
            "primary_end_to_end_utilization": primary,
            "diagnostic_overlay_utilization": diagnostic,
            "launch_cycles": None,
        }

    monotonic_checks = {}
    for operator in config["workloads"]["operators"]:
        values = sorted(
            (
                item["size"],
                item["end_to_end_cycles"],
                item["overlay_cycles"],
            )
            for item in measurements.values()
            if item["operator"] == operator
        )
        monotonic_checks[operator] = all(
            later[1] >= earlier[1] and later[2] >= earlier[2]
            for earlier, later in pairwise(values)
        )
    utilization_checks = {
        key: item["launch_cycles"] is None
        and all(
            math.isfinite(value) and 0 <= value <= 1
            for group in (
                item["primary_end_to_end_utilization"],
                item["diagnostic_overlay_utilization"],
            )
            for value in group.values()
        )
        for key, item in measurements.items()
    }

    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    source_text = "\n".join(
        (PROJECT_ROOT / path).read_text(errors="replace")
        for name, path in config["source_layout"].items()
        if name != "driver"
    )
    forbidden_basenames = [
        Path(path).name for path in config["target_exclusion"]["forbidden_paths"]
    ]
    target_free_checks = {
        "no_forbidden_path_in_source": not any(
            name in source_text for name in forbidden_basenames
        ),
        "compile_manifest": compiled["paper_performance_targets_consumed"] is False,
        "run_manifest": run["paper_performance_targets_consumed"] is False,
        "measurements": all(
            item["launch_cycles"] is None for item in measurements.values()
        ),
        "frozen_inputs": not any(
            spec["path"] in config["target_exclusion"]["forbidden_paths"]
            for spec in config["frozen_inputs"].values()
        ),
    }
    patch_checks = {
        name: reverse_patch_check(spec)
        for name, spec in config["patch_control"].items()
    }
    h114_sources = parents["h114"]["source_files"]
    historical_descendant = historical_memory_descendant_check(h114_sources)
    functional_descendant = functional_overlay_descendant_check(h114_sources)
    cpp_source_checks = {
        "overlay_header": functional_descendant["header"],
        "overlay_source": functional_descendant["source"],
        "driver": qualify(PROJECT_ROOT / h114_sources["driver"]["path"])["sha256"]
        == h114_sources["driver"]["sha256"],
    }
    cpp_source_checks.update(
        {
            "adapter_header": historical_descendant["header"],
            "adapter_source": historical_descendant["source"],
        }
    )
    counts = {
        "compile_outputs": len(compiled["outputs"])
        == int(config["workloads"]["required_paths"]),
        "records": len(run["records"])
        == int(config["execution"]["required_executions"]),
        "optimized": sum(item["mode"] == "optimized" for item in run["records"])
        == int(config["execution"]["required_optimized_executions"]),
        "sanitizers": sum(item["mode"] in {"asan", "ubsan"} for item in run["records"])
        == int(config["execution"]["required_sanitizer_executions"]),
        "measurements": len(measurements)
        == int(config["workloads"]["required_paths"]),
    }

    acceptance_gates = [
        all(item["pass"] for item in frozen.values())
        and all(parent_checks.values()),
        all(counts.values()) and all(target_free_checks.values()),
        all(source_replay_checks.values()),
        all(compile_checks.values()) and all(structure_checks.values()),
        all(hardware_checks.values()),
        all(memory_contract_checks.values()),
        all(run["checks"].values())
        and all(run["replay_checks"].values())
        and all(run["sanitizer_checks"].values()),
        all(work_checks.values()),
        all(memory_checks.values()),
        all(monotonic_checks.values()) and all(counter_checks.values()),
        all(utilization_checks.values())
        and config["metrics"]["launch_cycles"] is None,
        all(cpp_source_checks.values())
        and all(item["pass"] for item in patch_checks.values())
        and config["validation_eligible"] is False
        and config["classification"] == "target_free_exact_fig22_coupled_workloads",
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "compile_manifest": compile_file["pass"]
        and all(compiled["checks"].values()),
        "run_manifest": run_file["pass"] and all(run["checks"].values()),
        "compile": all(compile_checks.values()),
        "source_replay": all(source_replay_checks.values()),
        "structure_evaluated": len(structure_checks) == 16,
        "execution": all(execution_checks.values()),
        "work": all(work_checks.values()),
        "memory": all(memory_checks.values()),
        "records": all(record_checks.values()),
        "counters_evaluated": len(counter_checks) == 16,
        "counts": all(counts.values()),
        "source_files": all(item["pass"] for item in source_files.values()),
        "patches": all(item["pass"] for item in patch_checks.values()),
        "target_free": all(target_free_checks.values()),
        "acceptance_evaluated": len(acceptance_gates) == 12
        and all(isinstance(value, bool) for value in acceptance_gates),
    }
    integrity = all(integrity_checks.values())
    supported = integrity and all(acceptance_gates)
    primary_ranges = {
        resource: {
            "minimum": min(
                item["primary_end_to_end_utilization"][resource]
                for item in measurements.values()
            ),
            "maximum": max(
                item["primary_end_to_end_utilization"][resource]
                for item in measurements.values()
            ),
        }
        for resource in RESOURCES
    }
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
        "paper_reproduction_claim": "none_target_free_fig22_source_execution_only",
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "compile_manifest": compile_file,
        "run_manifest": run_file,
        "compile_checks": compile_checks,
        "source_replay_checks": source_replay_checks,
        "structure_checks": structure_checks,
        "hardware_checks": hardware_checks,
        "memory_contract_checks": memory_contract_checks,
        "execution_checks": execution_checks,
        "work_checks": work_checks,
        "memory_checks": memory_checks,
        "counter_checks": counter_checks,
        "monotonic_checks": monotonic_checks,
        "record_checks": record_checks,
        "measurements": measurements,
        "utilization_checks": utilization_checks,
        "target_free_checks": target_free_checks,
        "patch_checks": patch_checks,
        "cpp_source_checks": cpp_source_checks,
        "counts": counts,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "paths": len(measurements),
            "executions": len(run["records"]),
            "optimized_executions": sum(
                item["mode"] == "optimized" for item in run["records"]
            ),
            "sanitizer_executions": sum(
                item["mode"] in {"asan", "ubsan"} for item in run["records"]
            ),
            "primary_utilization_ranges": primary_ranges,
            "minimum_end_to_end_cycles": min(
                item["end_to_end_cycles"] for item in measurements.values()
            ),
            "maximum_end_to_end_cycles": max(
                item["end_to_end_cycles"] for item in measurements.values()
            ),
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
