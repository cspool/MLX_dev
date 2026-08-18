#!/usr/bin/env python3
"""Audit H110 corrected pipelined full-mesh throughput paths."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from mlxsim.dsagen_overlay import canonical_json
from mlxsim.pipelined_full_mesh_paths import compile_pipelined_path
from mlxsim.repeat_folding import fit_affine, relative_error

try:
    from scripts.audit_fig24_25_exact_paths import (
        git_commit,
        qualify,
        reconstruct_full_work,
    )
    from scripts.audit_fig24_25_full_mesh_paths import (
        expected_fu_classes,
        full_mesh_coordinates,
    )
except ModuleNotFoundError:
    from audit_fig24_25_exact_paths import (
        git_commit,
        qualify,
        reconstruct_full_work,
    )
    from audit_fig24_25_full_mesh_paths import (
        expected_fu_classes,
        full_mesh_coordinates,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/pipelined_full_mesh_paths_v1.yaml"


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / item["path"], item)
        for name, item in config["frozen_inputs"].items()
    }
    h109 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h109"]["path"]).read_text()
    )
    contracts_snapshot = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["contracts"]["path"]).read_text()
    )
    output_root = PROJECT_ROOT / config["output_root"]
    compile_path = output_root / "pipelined-full-mesh-compile-manifest.json"
    run_path = output_root / "pipelined-full-mesh-run-manifest.json"
    compile_file = qualify(compile_path)
    run_file = qualify(run_path)
    compiled = json.loads(compile_path.read_text())
    run = json.loads(run_path.read_text())

    compile_checks = {}
    execution_checks = {}
    execution_details = {}
    coordinate_checks = {}
    measurements = {}
    scale_work = {}
    for run_key, item in compiled["outputs"].items():
        path_key, scale_text = run_key.rsplit("-q", maxsplit=1)
        scale = int(scale_text)
        contract = compiled["path_contracts"][path_key]
        document, metadata, original = compile_pipelined_path(
            run_key=run_key,
            contract=contract,
            scale=scale,
            active_window=int(config["hardware"]["active_window"]),
            contexts=int(config["hardware"]["iteration_contexts_per_block"]),
            operand_contexts_per_pe=int(
                config["hardware"]["operand_contexts_per_pe"]
            ),
        )
        artifact_path = PROJECT_ROOT / item["artifact"]["path"]
        compile_checks[run_key] = (
            qualify(artifact_path, item["artifact"])["pass"]
            and artifact_path.read_text() == canonical_json(document)
            and item["metadata"] == metadata
            and document["blocks"] == original["blocks"]
            and document["functional_units"] == original["functional_units"]
            and document["routing"] == original["routing"]
            and document["pipelines"] == original["pipelines"]
            and document["memory_backend"] == original["memory_backend"]
        )
        coordinate_checks[run_key] = full_mesh_coordinates(metadata)
        first, second = (
            run["records"][run_key]["first"],
            run["records"][run_key]["second"],
        )
        summary, adapter = first["summary"], first["adapter"]
        classes = expected_fu_classes(document, metadata)
        physical = summary["productive_pe_cycles_by_fu_class"]
        checks = {
            "files": all(
                qualify(
                    PROJECT_ROOT / record[path_key_name],
                    {"sha256": record[hash_key]},
                )["pass"]
                for record in (first, second)
                for path_key_name, hash_key in (
                    ("summary_path", "summary_sha256"),
                    ("adapter_path", "adapter_sha256"),
                )
            ),
            "replay": first["summary_sha256"] == second["summary_sha256"]
            and first["adapter_sha256"] == second["adapter_sha256"],
            "done": summary["done"] is True,
            "mode": summary["pe_dependency_model"] == "dpu_pipelined",
            "contexts": summary["iteration_contexts_per_block"]
            == int(config["hardware"]["iteration_contexts_per_block"])
            and summary["max_inflight_iterations_per_block"]
            <= int(config["hardware"]["iteration_contexts_per_block"]),
            "instructions": summary["instructions_issued"]
            == summary["instructions_completed"]
            == sum(metadata["pipeline_counts"].values()),
            "pipelines": summary["issued_by_pipeline"]
            == metadata["pipeline_counts"],
            "events": summary["boundary_events_emitted"]
            == metadata["dynamic_event_count"]
            and metadata["dynamic_event_count"]
            == metadata["dynamic_event_demand_count"]
            and metadata["event_names_balanced"] is True,
            "memory": summary["external_memory_requests"]
            == summary["external_memory_completions"]
            == metadata["memory_requests"]
            == adapter["requests"]
            == adapter["responses"],
            "spad": adapter["ports"] == int(config["hardware"]["spad_ports"])
            and adapter["axis"] == config["hardware"]["spad_axis"],
            "physical": summary["physical_pe_count"] == 16
            and summary["mapped_pe_count"] == 16
            and sorted(physical) == classes
            and all(int(physical[name]) > 0 for name in classes),
            "mesh": coordinate_checks[run_key]["pass"],
        }
        execution_checks[run_key] = all(checks.values())
        execution_details[run_key] = {
            "checks": checks,
            "physical_fu_cycles": physical,
        }
        measurements[run_key] = {
            "cycles": int(summary["cycles"]),
            "physical_fma_pe_cycles": int(physical["fma"]),
            "metadata": metadata,
        }
        scale_work[run_key] = reconstruct_full_work(
            contract=contract,
            metadata=metadata,
            vector_bytes=int(config["hardware"]["vector_bytes"]),
            simd_width=int(config["hardware"]["simd_width"]),
        )

    fit_scales = [int(value) for value in config["fit_scales"]]
    holdout_scales = [int(value) for value in config["holdout_scales"]]
    cycle_limit = float(config["cycle_relative_error_limit"])
    residence_limit = float(config["physical_fma_relative_error_limit"])
    issue_minimum = float(config["fma_issue_utilization_minimum"])
    old_speedup_minimum = float(config["qkv_old_cycle_speedup_minimum"])
    models = {}
    full_estimates = {}
    cycle_errors = []
    residence_errors = []
    qkv_issue_utilizations = {}
    old_speedups = {}
    for path_key, contract in compiled["path_contracts"].items():
        cycle_model = fit_affine(
            fit_scales[0],
            measurements[f"{path_key}-q{fit_scales[0]}"]["cycles"],
            fit_scales[1],
            measurements[f"{path_key}-q{fit_scales[1]}"]["cycles"],
        )
        residence_model = fit_affine(
            fit_scales[0],
            measurements[f"{path_key}-q{fit_scales[0]}"][
                "physical_fma_pe_cycles"
            ],
            fit_scales[1],
            measurements[f"{path_key}-q{fit_scales[1]}"][
                "physical_fma_pe_cycles"
            ],
        )
        holdouts = []
        for scale in holdout_scales:
            actual_cycles = measurements[f"{path_key}-q{scale}"]["cycles"]
            actual_residence = measurements[f"{path_key}-q{scale}"][
                "physical_fma_pe_cycles"
            ]
            predicted_cycles = cycle_model.predict(scale)
            predicted_residence = residence_model.predict(scale)
            cycle_error = relative_error(predicted_cycles, actual_cycles)
            residence_error = relative_error(
                predicted_residence, actual_residence
            )
            cycle_errors.append(cycle_error)
            residence_errors.append(residence_error)
            holdouts.append(
                {
                    "scale": scale,
                    "actual_cycles": actual_cycles,
                    "predicted_cycles": predicted_cycles,
                    "cycle_relative_error": cycle_error,
                    "cycle_pass_5pct": cycle_error <= cycle_limit,
                    "actual_physical_fma_pe_cycles": actual_residence,
                    "predicted_physical_fma_pe_cycles": predicted_residence,
                    "physical_fma_relative_error": residence_error,
                    "physical_fma_pass_5pct": residence_error
                    <= residence_limit,
                }
            )
        full_scale = (
            int(contract["normalized"]["full_scale"])
            if contract["family"] == "qkv_bsmm"
            else int(contract["full_scale"])
        )
        full_cycles = cycle_model.predict(full_scale)
        full_residence = residence_model.predict(full_scale)
        scalar_fma = int(contract["actual"]["fu"]["fma"])
        issue_utilization = scalar_fma / (
            full_cycles
            * int(config["hardware"]["physical_pes"])
            * int(config["hardware"]["simd_width"])
        )
        residence_utilization = full_residence / (
            full_cycles * int(config["hardware"]["physical_pes"])
        )
        old_cycles = float(
            contracts_snapshot["old_full_estimates"][path_key]["cycles"]
        )
        old_speedup = old_cycles / full_cycles
        old_speedups[path_key] = old_speedup
        models[path_key] = {
            "family": contract["family"],
            "cycle_intercept": cycle_model.intercept,
            "cycle_slope": cycle_model.slope,
            "residence_intercept": residence_model.intercept,
            "residence_slope": residence_model.slope,
            "holdouts": holdouts,
            "full_scale": full_scale,
            "full_cycles": full_cycles,
            "full_physical_fma_pe_cycles": full_residence,
            "full_fma_issue_utilization": issue_utilization,
            "full_fma_residence_utilization": residence_utilization,
            "old_h102_cycles": old_cycles,
            "old_h102_cycle_speedup": old_speedup,
        }
        full_estimates[path_key] = {
            "cycles": full_cycles,
            "physical_fma_pe_cycles": full_residence,
            "fma_issue_utilization": issue_utilization,
            "fma_residence_utilization": residence_utilization,
            "old_h102_cycle_speedup": old_speedup,
        }
        if contract["family"] == "qkv_bsmm":
            qkv_issue_utilizations[path_key] = issue_utilization

    full_work_checks = {
        key: all(
            scale_work[f"{key}-q{scale}"]["pass"]
            for scale in [*fit_scales, *holdout_scales]
        )
        for key in compiled["path_contracts"]
    }
    numerical = {
        "cycle_holdouts_passed": sum(
            error <= cycle_limit for error in cycle_errors
        ),
        "cycle_holdouts_total": len(cycle_errors),
        "cycle_mape": sum(cycle_errors) / len(cycle_errors),
        "cycle_max_error": max(cycle_errors),
        "all_cycle_holdouts_pass": all(
            error <= cycle_limit for error in cycle_errors
        ),
        "residence_holdouts_passed": sum(
            error <= residence_limit for error in residence_errors
        ),
        "residence_holdouts_total": len(residence_errors),
        "residence_mape": sum(residence_errors) / len(residence_errors),
        "residence_max_error": max(residence_errors),
        "all_residence_holdouts_pass": all(
            error <= residence_limit for error in residence_errors
        ),
        "qkv_issue_paths_passed": sum(
            value >= issue_minimum for value in qkv_issue_utilizations.values()
        ),
        "qkv_issue_paths_total": len(qkv_issue_utilizations),
        "minimum_qkv_issue_utilization": min(qkv_issue_utilizations.values()),
        "all_qkv_issue_utilizations_pass": all(
            value >= issue_minimum for value in qkv_issue_utilizations.values()
        ),
        "corrected_paths_faster": sum(value > 1 for value in old_speedups.values()),
        "corrected_paths_total": len(old_speedups),
        "minimum_qkv_old_cycle_speedup": min(
            old_speedups[key] for key in qkv_issue_utilizations
        ),
        "all_qkv_old_cycle_speedups_pass": all(
            old_speedups[key] >= old_speedup_minimum
            for key in qkv_issue_utilizations
        ),
    }
    h109_manifest = qualify(
        PROJECT_ROOT / h109["run_manifest"]["path"], h109["run_manifest"]
    )
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    source_text = "\n".join(
        (PROJECT_ROOT / path).read_text()
        for name, path in config["source_layout"].items()
        if name != "auditor"
    ).lower()
    counts = {
        "paths": len(models) == int(config["required"]["paths"]),
        "configs": len(measurements) == int(config["required"]["configs"]),
        "cycle_holdouts": len(cycle_errors)
        == int(config["required"]["cycle_holdouts"]),
        "residence_holdouts": len(residence_errors)
        == int(config["required"]["physical_fma_holdouts"]),
        "qkv": len(qkv_issue_utilizations) == int(config["required"]["qkv_paths"]),
    }
    acceptance_gates = [
        counts["paths"] and counts["configs"],
        all(compiled["checks"][key]["semantic_identity"] for key in compiled["checks"]),
        all(full_work_checks.values())
        and all(item["pass"] for item in coordinate_checks.values()),
        all(
            execution_details[key]["checks"]["mode"]
            and execution_details[key]["checks"]["contexts"]
            and execution_details[key]["checks"]["instructions"]
            for key in execution_details
        ),
        all(
            execution_details[key]["checks"]["memory"]
            and execution_details[key]["checks"]["spad"]
            for key in execution_details
        ),
        all(run["checks"].values())
        and len(run["records"]) == int(config["required"]["configs"]),
        numerical["all_cycle_holdouts_pass"],
        numerical["all_residence_holdouts_pass"],
        numerical["all_qkv_issue_utilizations_pass"],
        numerical["corrected_paths_faster"] == numerical["corrected_paths_total"]
        and numerical["all_qkv_old_cycle_speedups_pass"],
        all(
            execution_details[key]["checks"]["contexts"] for key in execution_details
        ),
        "paper_targets" not in source_text and h109_manifest["pass"],
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "compile_manifest": compile_file["pass"]
        and compiled["paper_performance_targets_consumed"] is False,
        "run_manifest": run_file["pass"]
        and run["paper_performance_targets_consumed"] is False,
        "compile": all(compile_checks.values()),
        "execution": all(execution_checks.values()),
        "replays": all(run["checks"].values()),
        "work": all(full_work_checks.values()),
        "coordinates": all(item["pass"] for item in coordinate_checks.values()),
        "counts": all(counts.values()),
        "h109": h109_manifest["pass"],
        "source_files": all(item["pass"] for item in source_files.values()),
        "targets_absent": "paper_targets" not in source_text,
        "acceptance_evaluated": len(acceptance_gates) == 12
        and all(isinstance(value, bool) for value in acceptance_gates),
    }
    integrity = all(integrity_checks.values())
    supported = (
        integrity
        and all(acceptance_gates)
        and numerical["all_cycle_holdouts_pass"]
        and numerical["all_residence_holdouts_pass"]
        and numerical["all_qkv_issue_utilizations_pass"]
        and numerical["all_qkv_old_cycle_speedups_pass"]
        and numerical["corrected_paths_faster"] == numerical["corrected_paths_total"]
    )
    summary = {
        "paths": len(models),
        "configs": len(measurements),
        "double_runs": 2 * len(run["records"]),
        **numerical,
        "acceptance_gates_passed": sum(acceptance_gates),
        "acceptance_gates_total": len(acceptance_gates),
        "residence_failure_family_counts": {
            family: sum(
                not holdout["physical_fma_pass_5pct"]
                for model in models.values()
                if model["family"] == family
                for holdout in model["holdouts"]
            )
            for family in ("fft", "qkv_bsmm", "swa")
        },
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
        "paper_reproduction_claim": "none_target_free_corrected_throughput_only",
        "frozen_inputs": frozen,
        "h109_run_manifest": h109_manifest,
        "compile_manifest": compile_file,
        "run_manifest": run_file,
        "compile_checks": compile_checks,
        "execution_checks": execution_checks,
        "execution_details": execution_details,
        "coordinate_checks": coordinate_checks,
        "scale_work": scale_work,
        "full_work_checks": full_work_checks,
        "measurements": measurements,
        "models": models,
        "full_estimates": full_estimates,
        "qkv_issue_utilizations": qkv_issue_utilizations,
        "old_speedups": old_speedups,
        "summary": summary,
        "source_files": source_files,
        "integrity_checks": integrity_checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report = build_audit(config)
    output = PROJECT_ROOT / config["result_path"]
    if args.verify_existing:
        existing = json.loads(output.read_text())
        keys = (
            "hypothesis_status",
            "audit_integrity",
            "full_estimates",
            "summary",
            "integrity_checks",
        )
        matches = all(existing.get(key) == report.get(key) for key in keys)
        print(json.dumps({"existing_matches": matches, **report}, indent=2))
        return 0 if matches else 1
    if args.preflight_only:
        print(json.dumps(report, indent=2))
        return 0 if report["audit_integrity"] and not output.exists() else 1
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["hypothesis_status"], **report["summary"]}, indent=2))
    return 0 if report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
