#!/usr/bin/env python3
"""Audit H64's target-free Figure 10 scalability mechanism."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig10_scalability_v1.yaml"


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
    started = time.perf_counter()
    frozen_spec = config["frozen_input"]["mapping"]
    frozen_file = qualify(PROJECT_ROOT / frozen_spec["path"], frozen_spec)
    frozen = json.loads((PROJECT_ROOT / frozen_spec["path"]).read_text(encoding="utf-8"))
    root = PROJECT_ROOT / config["output_root"]
    compile_path = root / "fig10-scalability-compile-manifest.json"
    run_path = root / "fig10-scalability-run-manifest.json"
    compile_file = qualify(compile_path)
    run_file = qualify(run_path)
    compiler = json.loads(compile_path.read_text(encoding="utf-8"))
    run = json.loads(run_path.read_text(encoding="utf-8"))
    outputs: dict[str, Any] = {}
    compilation_checks: dict[str, bool] = {}
    execution_checks: dict[str, bool] = {}
    for key, record in compiler["outputs"].items():
        primary_path = PROJECT_ROOT / record["primary"]["path"]
        replay_path = PROJECT_ROOT / record["replay"]["path"]
        primary_file = qualify(primary_path, record["primary"])
        replay_file = qualify(replay_path, record["replay"])
        document = json.loads(primary_path.read_text(encoding="utf-8"))
        metadata = record["metadata"]
        sequence_text, hardware_name = key.split("-", 1)
        sequence = int(sequence_text)
        hardware = config["configurations"][hardware_name]
        mesh_x, mesh_y = (int(value) for value in hardware["mesh"])
        physical_pes = mesh_x * mesh_y
        simd_width = int(hardware["simd_width"])
        expected_groups = sequence * int(config["workload"]["batch"]) // simd_width
        checks = {
            "primary": primary_file["pass"],
            "replay": replay_file["pass"],
            "byte_identical": primary_file["sha256"] == replay_file["sha256"],
            "record_identical": record["identical"] is True,
            "fixed_memory": document["memory_backend"] == "fixed",
            "paper_static": document["pe_dependency_model"] == "paper_static",
            "mesh": metadata["mesh"] == [mesh_x, mesh_y],
            "simd": metadata["simd_width"] == simd_width,
            "groups": metadata["outer_vector_groups"] == expected_groups,
            "local_i1": metadata["local_i1_trip"] == 64 // physical_pes,
            "trip_count": {
                int(block["trip_count"]) for block in document["blocks"]
            }
            == {int(config["workload"]["hidden_width"]) // physical_pes * expected_groups},
            "footprint": metadata["max_active_instruction_footprint_per_pe"] <= 32,
            "no_address_sequences": all(
                "memory_address_sequence" not in instruction
                for block in document["blocks"]
                for instruction in block["instructions"]
            ),
            "no_targets": metadata["paper_performance_targets_consumed"] is False,
        }
        compilation_checks[key] = all(checks.values())
        first_record = run["runs"][key]["first"]
        second_record = run["runs"][key]["second"]
        first_path = PROJECT_ROOT / first_record["path"]
        second_path = PROJECT_ROOT / second_record["path"]
        first_file = qualify(first_path, first_record)
        second_file = qualify(second_path, second_record)
        summary = first_record["summary"]
        expected_pipeline = metadata["expected_pipeline_instructions"]
        run_checks = {
            "first": first_file["pass"],
            "second": second_file["pass"],
            "replay": first_file["sha256"] == second_file["sha256"],
            "summary_binding": json.loads(first_path.read_text(encoding="utf-8"))
            == summary,
            "done": summary["done"] is True,
            "physical_pes": summary["physical_pe_count"] == physical_pes,
            "mapped_pes": summary["mapped_pe_count"] == physical_pes,
            "instructions": summary["instructions_issued"]
            == summary["instructions_completed"]
            == metadata["instruction_count"],
            "pipelines": all(
                summary["issued_by_pipeline"][name] == count
                for name, count in expected_pipeline.items()
            ),
            "events": summary["boundary_events_emitted"]
            == metadata["boundary_events"],
            "routes": summary["route_hops"] == metadata["route_hops"],
            "fixed_no_external": summary["external_memory_requests"] == 0,
        }
        execution_checks[key] = all(run_checks.values())
        outputs[key] = {
            "primary": primary_file,
            "replay": replay_file,
            "compilation_checks": checks,
            "first": first_file,
            "second": second_file,
            "execution_checks": run_checks,
        }
    mechanism_checks: dict[str, Any] = {}
    sequences = [str(value) for value in config["workload"]["sequence_lengths"]]
    for sequence in sequences:
        cycles = run["cycles"][sequence]
        simd = cycles["baseline"] / cycles["simd32_4x4"]
        mesh = cycles["baseline"] / cycles["simd8_8x8"]
        joint = cycles["baseline"] / cycles["simd32_8x8"]
        mechanism_checks[sequence] = {
            "all_optimized": all(
                cycles[name] < cycles["baseline"]
                for name in config["configurations"]
                if name != "baseline"
            ),
            "simd_near_four": 3.9 <= simd <= 4.1,
            "mesh_positive_scaling": mesh > 1.0,
            "joint_composes": math.isclose(joint, simd * mesh, rel_tol=0.01),
        }
    monotonic_checks = {
        name: all(
            int(run["cycles"][right][name]) > int(run["cycles"][left][name])
            for left, right in pairwise(sequences)
        )
        for name in config["configurations"]
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    source_text = "\n".join(
        (PROJECT_ROOT / config["source_layout"][name]).read_text(encoding="utf-8")
        for name in ("module", "compiler", "runner")
    )
    integrity_checks = {
        "frozen_file": frozen_file["pass"],
        "frozen_mapping": frozen.get("hypothesis_status")
        == frozen_spec["required_status"]
        and frozen.get("audit_integrity") is frozen_spec["required_integrity"],
        "compile_manifest": compile_file["pass"],
        "run_manifest": run_file["pass"],
        "twenty_outputs": compiler["output_count"] == len(run["runs"]) == 20,
        "compiler_replay": compiler["all_identical"] is True,
        "work_conservation": all(
            all(values.values()) for values in compiler["conservation"].values()
        ),
        "compilations": all(compilation_checks.values()),
        "executions": all(execution_checks.values()),
        "run_replay_checks": all(run["checks"].values()),
        "mechanism": all(
            all(values.values()) for values in mechanism_checks.values()
        ),
        "monotonic": all(monotonic_checks.values()),
        "source_files": all(item["pass"] for item in source_files.values()),
        "target_paths_absent": "fig23_scalability" not in source_text
        and "paper_targets" not in source_text,
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
        "frozen_mapping": frozen_file,
        "compile_manifest": compile_file,
        "run_manifest": run_file,
        "outputs": outputs,
        "compilation_checks": compilation_checks,
        "execution_checks": execution_checks,
        "cycles": run["cycles"],
        "speedups": run["speedups"],
        "mechanism_checks": mechanism_checks,
        "monotonic_checks": monotonic_checks,
        "source_files": source_files,
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
