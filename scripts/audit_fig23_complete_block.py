#!/usr/bin/env python3
"""Audit H141 target-free complete-block scaling robustness."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig23_complete_block_robustness_v1.yaml"


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    parents = {
        name: json.loads((PROJECT_ROOT / spec["path"]).read_text())
        for name, spec in config["frozen_inputs"].items()
        if "required_status" in spec
    }
    parent_checks = {
        name: parent["hypothesis_status"] == config["frozen_inputs"][name]["required_status"]
        and parent["audit_integrity"] is config["frozen_inputs"][name]["required_integrity"]
        for name, parent in parents.items()
    }
    output_root = PROJECT_ROOT / config["output_root"]
    compile_manifest_path = output_root / "complete-block-compile-manifest.json"
    run_manifest_path = output_root / "complete-block-run-manifest.json"
    compile_manifest = json.loads(compile_manifest_path.read_text())
    run_manifest = json.loads(run_manifest_path.read_text())
    generated_inputs = {
        "compile_manifest": qualify(compile_manifest_path),
        "run_manifest": qualify(run_manifest_path),
    }
    sequences = config["paper_disclosed_workload"]["sequence_lengths"]
    windows = config["robustness_grid"]["active_windows"]
    configurations = config["robustness_grid"]["configurations"]
    contract_checks = {
        "sequence_lengths": sequences == [512, 1024, 2048, 4096, 8192],
        "hidden_dimension": config["paper_disclosed_workload"]["hidden_dimension"] == 512,
        "batch": config["paper_disclosed_workload"]["batch"] == 8,
        "windows": windows == [2, 4],
        "hardware": configurations
        == {
            "baseline": {"simd_width": 8, "mesh": [4, 4]},
            "simd32_4x4": {"simd_width": 32, "mesh": [4, 4]},
            "simd8_8x8": {"simd_width": 8, "mesh": [8, 8]},
            "simd32_8x8": {"simd_width": 32, "mesh": [8, 8]},
        },
    }
    expected_keys = {
        f"N{sequence}-w{window}-{name}"
        for window in windows
        for sequence in sequences
        for name in configurations
    }
    compile_checks = {
        "experiment": compile_manifest["experiment_id"] == "H141",
        "target_free": compile_manifest["paper_performance_targets_consumed"] is False,
        "count": compile_manifest["output_count"] == int(config["acceptance"]["expected_configs"]),
        "keys": set(compile_manifest["outputs"]) == expected_keys,
        "identical": compile_manifest["all_identical"] is True
        and all(item["identical"] for item in compile_manifest["outputs"].values()),
        "conservation": all(
            all(group.values()) for group in compile_manifest["conservation"].values()
        ),
    }
    expected_operations = set(config["acceptance"]["expected_operation_classes"])
    graph_checks: dict[str, bool] = {}
    shape_checks: dict[str, bool] = {}
    for key, record in compile_manifest["outputs"].items():
        metadata = record["metadata"]
        hardware_name = key.rsplit("-", 1)[-1]
        hardware = configurations[hardware_name]
        graph_checks[key] = (
            metadata["stage_count"] == int(config["acceptance"]["expected_stage_count"])
            and len(metadata["stage_groups"]) == 28
            and set(metadata["operation_classes"]) == expected_operations
            and all(metadata["event_checks"].values())
            and metadata["final_event_count"] == metadata["spatial_shards"]
        )
        shape_checks[key] = (
            metadata["hardware_name"] == hardware_name
            and metadata["simd_width"] == hardware["simd_width"]
            and metadata["mesh"] == hardware["mesh"]
            and metadata["spatial_shards"] == hardware["mesh"][0] * hardware["mesh"][1]
            and metadata["active_window"] in windows
            and metadata["sequence_length"] in sequences
        )
    work_checks: dict[str, bool] = {}
    for window in windows:
        for sequence in sequences:
            group = f"N{sequence}-w{window}"
            records = {
                name: compile_manifest["outputs"][f"{group}-{name}"]["metadata"]
                for name in configurations
            }
            reference = records["baseline"]["work"]
            work_checks[group] = all(
                all(
                    metadata["work"][key] == reference[key]
                    for key in reference
                    if key.startswith("scalarized_")
                )
                for metadata in records.values()
            )
    run_checks = {
        "experiment": run_manifest["experiment_id"] == "H141",
        "target_free": run_manifest["paper_performance_targets_consumed"] is False,
        "configs": set(run_manifest["runs"]) == expected_keys,
        "runs": sum(len(builds) for builds in run_manifest["runs"].values())
        == int(config["acceptance"]["expected_runs"]),
        "checks": all(run_manifest["checks"].values()),
    }
    execution_checks: dict[str, bool] = {}
    for key, builds in run_manifest["runs"].items():
        metadata = compile_manifest["outputs"][key]["metadata"]
        execution_checks[key] = set(builds) == {"debug", "opt", "sanitize"} and all(
            item["summary"]["done"] is True
            and item["summary"]["cycles"] > 0
            and item["summary"]["instructions_issued"]
            == item["summary"]["instructions_completed"]
            == metadata["work"]["instruction_instances"]
            and item["summary"]["boundary_events_emitted"] == metadata["work"]["boundary_events"]
            and item["summary"]["max_active_tags"] <= metadata["active_window"]
            for item in builds.values()
        )
    minimum_speedup = float(config["acceptance"]["minimum_clear_speedup"])
    individual_speedups = []
    joint_speedups = []
    joint_dominance = []
    for group, speedups in run_manifest["speedups"].items():
        simd = float(speedups["simd32_4x4"])
        mesh = float(speedups["simd8_8x8"])
        joint = float(speedups["simd32_8x8"])
        individual_speedups.extend((simd, mesh))
        joint_speedups.append(joint)
        joint_dominance.append(joint > simd and joint > mesh)
    finite_speedups = all(
        math.isfinite(value) and value > 0 for value in [*individual_speedups, *joint_speedups]
    )
    individual_passes = sum(value >= minimum_speedup for value in individual_speedups)
    joint_passes = sum(value >= minimum_speedup for value in joint_speedups)
    source_files = {
        name: qualify(PROJECT_ROOT / path) for name, path in config["source_layout"].items()
    }
    source_text = "\n".join(
        (PROJECT_ROOT / path).read_text(errors="replace")
        for path in config["source_layout"].values()
    )
    forbidden = (
        "fig10-fig23" + "-transfer-run070.json",
        "dsagen-mlx-fig23" + "-run052.json",
        "multiport-fig23" + "-transfer-run075.json",
        "residual" + "_factor",
        "fit" + "_affine",
    )
    target_free_check = (
        not any(token in source_text for token in forbidden)
        and config["acceptance"]["targets_consumed"] is False
    )
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        all(contract_checks.values()),
        all(compile_checks.values()) and all(graph_checks.values()),
        all(shape_checks.values()),
        all(work_checks.values()),
        all(run_checks.values()) and all(execution_checks.values()),
        all(run_manifest["checks"].values()),
        finite_speedups
        and individual_passes == int(config["acceptance"]["required_individual_speedups"]),
        joint_passes == int(config["acceptance"]["required_joint_speedups"])
        and all(joint_dominance),
        target_free_check and all(item["pass"] for item in source_files.values()),
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "contract": all(contract_checks.values()),
        "compile": all(compile_checks.values()),
        "graph": all(graph_checks.values()),
        "shape": all(shape_checks.values()),
        "work": all(work_checks.values()),
        "runs": all(run_checks.values()) and all(execution_checks.values()),
        "speedups_finite": finite_speedups,
        "source": target_free_check and all(item["pass"] for item in source_files.values()),
        "acceptance_evaluated": len(acceptance_gates) == 10
        and all(isinstance(value, bool) for value in acceptance_gates),
    }
    integrity = all(integrity_checks.values())
    supported = integrity and all(acceptance_gates)
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
        "paper_reproduction_claim": "none_target_free_complete_block_robustness",
        "surrogate_identity_claim": "representative_complete_block_not_exact_author_schedule",
        "frozen_inputs": frozen,
        "generated_inputs": generated_inputs,
        "parent_checks": parent_checks,
        "contract_checks": contract_checks,
        "compile_checks": compile_checks,
        "graph_checks": graph_checks,
        "shape_checks": shape_checks,
        "work_checks": work_checks,
        "run_checks": run_checks,
        "execution_checks": execution_checks,
        "cycles": run_manifest["cycles"],
        "speedups": run_manifest["speedups"],
        "source_files": source_files,
        "target_free_check": target_free_check,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "compiled_configs": len(compile_manifest["outputs"]),
            "executions": sum(len(builds) for builds in run_manifest["runs"].values()),
            "individual_speedup_passes": individual_passes,
            "individual_speedup_total": len(individual_speedups),
            "joint_speedup_passes": joint_passes,
            "joint_speedup_total": len(joint_speedups),
            "minimum_simd_speedup": min(
                float(item["simd32_4x4"]) for item in run_manifest["speedups"].values()
            ),
            "maximum_simd_speedup": max(
                float(item["simd32_4x4"]) for item in run_manifest["speedups"].values()
            ),
            "minimum_mesh_speedup": min(
                float(item["simd8_8x8"]) for item in run_manifest["speedups"].values()
            ),
            "maximum_mesh_speedup": max(
                float(item["simd8_8x8"]) for item in run_manifest["speedups"].values()
            ),
            "minimum_joint_speedup": min(joint_speedups),
            "maximum_joint_speedup": max(joint_speedups),
            "all_work_conserved": all(work_checks.values()),
            "all_builds_identical_and_clean": all(run_manifest["checks"].values()),
            "figure23_target_join_eligible": supported,
            "active_simulator_figures_reproduced": 2,
            "active_simulator_figures_total": 8,
            "acceptance_gates_passed": sum(acceptance_gates),
            "acceptance_gates_total": len(acceptance_gates),
        },
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
            "compile_checks",
            "work_checks",
            "run_checks",
            "cycles",
            "speedups",
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
