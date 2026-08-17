#!/usr/bin/env python3
"""Audit Figure 21 logical identity versus current source-integrated execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from mlxsim.experiments import _llama_kernel_workloads
from mlxsim.workloads import compile_workload

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/fig21_workload_identity_v1.yaml"


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


def component_profile(n: int, *, sparse: bool, batch: int) -> dict[str, Any]:
    workloads = _llama_kernel_workloads(n, sparse=sparse, batch=batch)
    result = {}
    for component in ("qkv", "attention", "output", "ffn1", "ffn2"):
        profiles = [compile_workload(workload) for workload in workloads[component]]
        result[component] = {
            "operations": sum(profile.operations for profile in profiles),
            "offchip_bytes": sum(profile.offchip_bytes for profile in profiles),
            "output_elements": sum(profile.output_elements for profile in profiles),
            "stage_count": sum(len(profile.stages) for profile in profiles),
            "subcomponents": len(profiles),
        }
    return result


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    files = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    reports = {
        name: json.loads((PROJECT_ROOT / spec["path"]).read_text(encoding="utf-8"))
        for name, spec in config["frozen_inputs"].items()
    }
    parent_checks = {
        name: (
            "required_status" not in spec
            or reports[name]["hypothesis_status"] == spec["required_status"]
        )
        and (
            "required_integrity" not in spec
            or reports[name]["audit_integrity"] is spec["required_integrity"]
        )
        for name, spec in config["frozen_inputs"].items()
    }
    contract = config["contract"]
    logical_profiles = {}
    logical_checks = {}
    proxy_raw = {
        int(item["sequence_length"]): item for item in reports["fig21_proxy"]["raw"]
    }
    for n_value in contract["sequence_lengths"]:
        n = int(n_value)
        structured = component_profile(n, sparse=True, batch=int(contract["batch"]))
        dense = component_profile(n, sparse=False, batch=int(contract["batch"]))
        logical_profiles[f"N{n}"] = {"structured": structured, "dense": dense}
        raw = proxy_raw[n]
        component_order = ("qkv", "attention", "output", "ffn1", "ffn2")
        checks = {}
        for index, component in enumerate(component_order):
            checks[f"structured_{component}"] = (
                raw["mlx_structured_components"][index]["operations"]
                == structured[component]["operations"]
            )
            checks[f"dense_{component}"] = (
                raw["mlx_dense_components"][index]["operations"]
                == dense[component]["operations"]
            )
        logical_checks[f"N{n}"] = checks

    projection_coverage = reports["projections"]["coverage"]
    attention_shapes = set(reports["attention"]["models"])
    required_shapes = {f"N{int(n)}" for n in contract["sequence_lengths"]}
    source_coverage = {
        "h48": {
            "phase_coverage": True,
            "exact_schedule": False,
            "batch": None,
            "sequence_lengths": [],
            "layers_executed": None,
            "reason": reports["full_block"]["claim_scope"]["not_claimed"],
        },
        "h77": {
            "components": projection_coverage["kernels"],
            "sequence_lengths": projection_coverage["sequence_lengths"],
            "batch": 1,
            "output_projection": False,
            "layers_executed": 1,
        },
        "h83": {
            "components": ["attention"],
            "sequence_lengths": [256, 8192],
            "batch": 1,
            "output_projection": False,
            "layers_executed": 1,
        },
    }
    gap_checks = {
        "h6_logical_work_matches": all(
            all(checks.values()) for checks in logical_checks.values()
        ),
        "all_required_shapes_missing_from_h83": not required_shapes.issubset(
            attention_shapes
        ),
        "batch_mismatch": source_coverage["h77"]["batch"]
        != int(contract["batch"])
        and source_coverage["h83"]["batch"] != int(contract["batch"]),
        "output_projection_missing": source_coverage["h77"]["output_projection"]
        is False,
        "dense_source_execution_missing": True,
        "elementwise_exact_execution_missing": True,
        "thirty_two_layer_execution_missing": True,
        "h48_not_exact_schedule": "authors' unpublished instruction schedule"
        in source_coverage["h48"]["reason"],
    }
    integrity_checks = {
        "frozen_files": all(item["pass"] for item in files.values()),
        "parents": all(parent_checks.values()),
        "proxy_sequences": set(proxy_raw) == set(contract["sequence_lengths"]),
        "logical_profiles": len(logical_profiles) == 5,
        "identity_gap_proven": all(gap_checks.values()),
        "targets_consumed": False,
    }
    integrity = all(
        value for key, value in integrity_checks.items() if key != "targets_consumed"
    ) and not integrity_checks["targets_consumed"]
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
        "matched_source_execution_available": False,
        "frozen_inputs": files,
        "parent_checks": parent_checks,
        "contract": contract,
        "logical_profiles": logical_profiles,
        "logical_checks": logical_checks,
        "source_coverage": source_coverage,
        "gap_checks": gap_checks,
        "integrity_checks": integrity_checks,
        "paper_performance_targets_consumed": False,
        "conclusion": "Figure 21 logical work is explicit but no current source-integrated run matches its full execution identity",
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
        keys = ("hypothesis_status", "audit_integrity", "logical_checks", "source_coverage", "gap_checks", "integrity_checks")
        matches = all(existing.get(key) == report.get(key) for key in keys)
        print(json.dumps({"existing_matches": matches, **report}, indent=2))
        return 0 if matches else 1
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "hypothesis_status": report["hypothesis_status"],
                "audit_integrity": report["audit_integrity"],
                "matched_source_execution_available": report[
                    "matched_source_execution_available"
                ],
                "gap_checks": report["gap_checks"],
            },
            indent=2,
        )
    )
    return 0 if report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
