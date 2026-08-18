#!/usr/bin/env python3
"""Audit H102 exact Figure 24/25 paths over the full 4x4 mesh."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from mlxsim.dsagen_overlay import canonical_json
from mlxsim.fig24_25_full_mesh_paths import (
    compile_full_mesh_fft_cmp_path,
    compile_full_mesh_swa_path,
    compile_full_mesh_timed_path,
)
from mlxsim.repeat_folding import fit_affine, relative_error
from scripts.audit_fig24_25_exact_paths import (
    git_commit,
    qualify,
    reconstruct_full_work,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig24_25_full_mesh_paths_v1.yaml"


def recompile(
    *, run_key: str, contract: dict[str, Any], scale: int, active_window: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    family = contract["family"]
    case = contract["case"]
    operator = contract["operator"]
    if family == "qkv_bsmm":
        return compile_full_mesh_timed_path(
            name=run_key,
            normalized=contract["normalized"],
            scale=scale,
            active_window=active_window,
        )
    if family == "fft":
        return compile_full_mesh_fft_cmp_path(
            name=run_key,
            sequence_length=int(case["n"]),
            hidden_dimension=int(case["d"]),
            batch=int(case["batch"]),
            scale=scale,
        )
    if family == "swa":
        return compile_full_mesh_swa_path(
            name=run_key,
            sequence_length=int(case["n"]),
            hidden_dimension=int(case["d"]),
            batch=int(case["batch"]),
            window=int(operator["window"]),
            query_tile=int(operator["query_tile"]),
            scale=scale,
        )
    raise ValueError(family)


def expected_fu_classes(document: dict[str, Any], metadata: dict[str, Any]) -> list[str]:
    units = document["functional_units"]
    return sorted(
        {
            str(units[operation]["class"])
            for operation, count in metadata["operation_counts"].items()
            if int(count) > 0
        }
    )


def full_mesh_coordinates(metadata: dict[str, Any]) -> dict[str, Any]:
    expected = {(x, y) for y in range(4) for x in range(4)}
    groups = metadata.get("compute_coordinates_by_step")
    if groups is None:
        groups = [
            item["coordinates"] for item in metadata["compute_coordinates_by_phase"]
        ]
    actual = [{tuple(coord) for coord in group} for group in groups]
    checks = {
        "has_compute_phases": bool(actual),
        "all_16_coordinates": all(group == expected for group in actual),
        "instruction_capacity": int(metadata["max_active_instructions_per_pe"])
        <= 32,
    }
    return {
        "phase_count": len(actual),
        "coordinate_counts": [len(group) for group in actual],
        "max_active_instructions_per_pe": metadata["max_active_instructions_per_pe"],
        "checks": checks,
        "pass": all(checks.values()),
    }


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    files = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    source_inputs = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_sources"].items()
    }
    parents = {
        name: json.loads((PROJECT_ROOT / spec["path"]).read_text(encoding="utf-8"))
        for name, spec in config["frozen_inputs"].items()
    }
    parent_checks = {
        name: report["hypothesis_status"]
        == config["frozen_inputs"][name]["required_status"]
        and report["audit_integrity"]
        is config["frozen_inputs"][name]["required_integrity"]
        for name, report in parents.items()
    }
    output_root = PROJECT_ROOT / config["output_root"]
    compile_path = output_root / "fig24-25-full-mesh-compile-manifest.json"
    run_path = output_root / "fig24-25-full-mesh-run-manifest.json"
    compile_file = qualify(compile_path)
    run_file = qualify(run_path)
    compiler = json.loads(compile_path.read_text(encoding="utf-8"))
    run = json.loads(run_path.read_text(encoding="utf-8"))

    compile_checks: dict[str, bool] = {}
    execution_checks: dict[str, bool] = {}
    execution_details: dict[str, dict[str, Any]] = {}
    measurements: dict[str, dict[str, Any]] = {}
    scale_work: dict[str, dict[str, Any]] = {}
    coordinate_checks: dict[str, dict[str, Any]] = {}
    for run_key, item in compiler["outputs"].items():
        path_key, scale_text = run_key.rsplit("-q", maxsplit=1)
        scale = int(scale_text)
        contract = compiler["path_contracts"][path_key]
        document, metadata = recompile(
            run_key=run_key,
            contract=contract,
            scale=scale,
            active_window=int(config["hardware"]["active_window"]),
        )
        path = PROJECT_ROOT / item["artifact"]["path"]
        artifact = qualify(path, item["artifact"])
        compile_checks[run_key] = (
            artifact["pass"]
            and path.read_text(encoding="utf-8") == canonical_json(document)
            and item["metadata"] == metadata
        )
        coordinate_checks[run_key] = full_mesh_coordinates(metadata)
        first = run["records"][run_key]["first"]
        second = run["records"][run_key]["second"]
        first_summary = qualify(
            PROJECT_ROOT / first["summary_path"], {"sha256": first["summary_sha256"]}
        )
        second_summary = qualify(
            PROJECT_ROOT / second["summary_path"], {"sha256": second["summary_sha256"]}
        )
        first_adapter = qualify(
            PROJECT_ROOT / first["adapter_path"], {"sha256": first["adapter_sha256"]}
        )
        second_adapter = qualify(
            PROJECT_ROOT / second["adapter_path"], {"sha256": second["adapter_sha256"]}
        )
        summary = first["summary"]
        adapter = first["adapter"]
        classes = expected_fu_classes(document, metadata)
        physical = summary.get("productive_pe_cycles_by_fu_class", {})
        checks = {
            "files": first_summary["pass"]
            and second_summary["pass"]
            and first_adapter["pass"]
            and second_adapter["pass"],
            "replay": first["summary_sha256"] == second["summary_sha256"]
            and first["adapter_sha256"] == second["adapter_sha256"],
            "done": summary["done"] is True,
            "instructions": summary["instructions_issued"]
            == summary["instructions_completed"]
            == sum(metadata["pipeline_counts"].values()),
            "pipelines": summary["issued_by_pipeline"] == metadata["pipeline_counts"],
            "events": summary["boundary_events_emitted"]
            == metadata["dynamic_event_count"],
            "event_balance": metadata["event_names_balanced"] is True
            and metadata["dynamic_event_count"]
            == metadata["dynamic_event_demand_count"],
            "memory": summary["external_memory_requests"]
            == summary["external_memory_completions"]
            == metadata["memory_requests"]
            == adapter["requests"]
            == adapter["responses"],
            "ports": adapter["ports"] == int(config["hardware"]["spad_ports"]),
            "axis": adapter["axis"] == config["hardware"]["spad_axis"],
            "physical_pes": summary["physical_pe_count"] == 16
            and summary["mapped_pe_count"] == 16,
            "physical_fu_classes": sorted(physical) == classes
            and all(int(physical[name]) > 0 for name in classes),
            "full_mesh": coordinate_checks[run_key]["pass"],
        }
        execution_checks[run_key] = all(checks.values())
        execution_details[run_key] = {
            "checks": checks,
            "expected_fu_classes": classes,
            "physical_fu_cycles": physical,
        }
        measurements[run_key] = {
            "cycles": int(summary["cycles"]),
            "physical_fma_pe_cycles": int(physical["fma"]),
            "metadata": metadata,
            "physical_fu_cycles": physical,
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
    fma_limit = float(config["physical_fma_relative_error_limit"])
    models: dict[str, dict[str, Any]] = {}
    full_estimates: dict[str, dict[str, float]] = {}
    cycle_errors: list[float] = []
    fma_errors: list[float] = []
    qkv_utilizations: dict[str, float] = {}
    for path_key, contract in compiler["path_contracts"].items():
        cycle_model = fit_affine(
            fit_scales[0],
            measurements[f"{path_key}-q{fit_scales[0]}"]["cycles"],
            fit_scales[1],
            measurements[f"{path_key}-q{fit_scales[1]}"]["cycles"],
        )
        fma_model = fit_affine(
            fit_scales[0],
            measurements[f"{path_key}-q{fit_scales[0]}"]["physical_fma_pe_cycles"],
            fit_scales[1],
            measurements[f"{path_key}-q{fit_scales[1]}"]["physical_fma_pe_cycles"],
        )
        holdouts = []
        for scale in holdout_scales:
            actual_cycles = measurements[f"{path_key}-q{scale}"]["cycles"]
            actual_fma = measurements[f"{path_key}-q{scale}"][
                "physical_fma_pe_cycles"
            ]
            predicted_cycles = cycle_model.predict(scale)
            predicted_fma = fma_model.predict(scale)
            cycle_error = relative_error(predicted_cycles, actual_cycles)
            fma_error = relative_error(predicted_fma, actual_fma)
            cycle_errors.append(cycle_error)
            fma_errors.append(fma_error)
            holdouts.append(
                {
                    "scale": scale,
                    "actual_cycles": actual_cycles,
                    "predicted_cycles": predicted_cycles,
                    "cycle_relative_error": cycle_error,
                    "cycle_pass_5pct": cycle_error <= cycle_limit,
                    "actual_physical_fma_pe_cycles": actual_fma,
                    "predicted_physical_fma_pe_cycles": predicted_fma,
                    "physical_fma_relative_error": fma_error,
                    "physical_fma_pass_5pct": fma_error <= fma_limit,
                }
            )
        full_scale = (
            int(contract["normalized"]["full_scale"])
            if contract["family"] == "qkv_bsmm"
            else int(contract["full_scale"])
        )
        full_cycles = cycle_model.predict(full_scale)
        full_fma = fma_model.predict(full_scale)
        utilization = full_fma / (full_cycles * 16)
        models[path_key] = {
            "family": contract["family"],
            "cycle_intercept": cycle_model.intercept,
            "cycle_slope": cycle_model.slope,
            "physical_fma_intercept": fma_model.intercept,
            "physical_fma_slope": fma_model.slope,
            "holdouts": holdouts,
            "full_scale": full_scale,
            "full_predicted_cycles": full_cycles,
            "full_predicted_physical_fma_pe_cycles": full_fma,
            "full_predicted_fma_utilization": utilization,
        }
        full_estimates[path_key] = {
            "cycles": full_cycles,
            "physical_fma_pe_cycles": full_fma,
            "fma_utilization": utilization,
        }
        if contract["family"] == "qkv_bsmm":
            qkv_utilizations[path_key] = utilization

    full_work_checks = {
        key: all(
            scale_work[f"{key}-q{scale}"]["pass"]
            for scale in [*fit_scales, *holdout_scales]
        )
        for key in compiler["path_contracts"]
    }
    qkv_minimum = float(config["qkv_full_fma_utilization_minimum"])
    numerical = {
        "passing_cycle_holdouts": sum(error <= cycle_limit for error in cycle_errors),
        "total_cycle_holdouts": len(cycle_errors),
        "cycle_mape": sum(cycle_errors) / len(cycle_errors),
        "cycle_max_error": max(cycle_errors),
        "all_cycle_holdouts_pass": all(error <= cycle_limit for error in cycle_errors),
        "passing_physical_fma_holdouts": sum(
            error <= fma_limit for error in fma_errors
        ),
        "total_physical_fma_holdouts": len(fma_errors),
        "physical_fma_mape": sum(fma_errors) / len(fma_errors),
        "physical_fma_max_error": max(fma_errors),
        "all_physical_fma_holdouts_pass": all(
            error <= fma_limit for error in fma_errors
        ),
        "passing_qkv_utilizations": sum(
            value >= qkv_minimum for value in qkv_utilizations.values()
        ),
        "total_qkv_utilizations": len(qkv_utilizations),
        "minimum_qkv_utilization": min(qkv_utilizations.values()),
        "all_qkv_utilizations_pass": all(
            value >= qkv_minimum for value in qkv_utilizations.values()
        ),
    }
    summary = {
        "path_count": len(models),
        "run_count": len(measurements),
        "full_estimate_count": len(full_estimates),
        "full_work_passing_paths": sum(full_work_checks.values()),
        "full_mesh_passing_runs": sum(item["pass"] for item in coordinate_checks.values()),
        **numerical,
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    implementation_text = "\n".join(
        (PROJECT_ROOT / path).read_text(encoding="utf-8")
        for name, path in config["source_layout"].items()
        if name != "auditor"
    ).lower()
    integrity_checks = {
        "frozen_files": all(item["pass"] for item in files.values()),
        "frozen_sources": all(item["pass"] for item in source_inputs.values()),
        "parents": all(parent_checks.values()),
        "compile_manifest": compile_file["pass"]
        and compiler["paper_performance_targets_consumed"] is False
        and len(compiler["outputs"]) == int(config["required"]["configs"]),
        "run_manifest": run_file["pass"]
        and run["paper_performance_targets_consumed"] is False
        and len(run["records"]) == int(config["required"]["configs"]),
        "compiler_replay": all(compile_checks.values()),
        "executions": all(execution_checks.values()),
        "run_replays": all(run["checks"].values()),
        "full_work": all(full_work_checks.values()),
        "full_mesh": all(item["pass"] for item in coordinate_checks.values()),
        "source_files": all(item["pass"] for item in source_files.values()),
        "targets_absent": "paper_targets" not in implementation_text,
        "counts": summary["path_count"] == int(config["required"]["paths"])
        and summary["run_count"] == int(config["required"]["configs"])
        and summary["total_cycle_holdouts"]
        == int(config["required"]["cycle_holdouts"])
        and summary["total_physical_fma_holdouts"]
        == int(config["required"]["cycle_holdouts"])
        and summary["total_qkv_utilizations"] == int(config["required"]["qkv_paths"]),
    }
    integrity = all(integrity_checks.values())
    supported = (
        integrity
        and numerical["all_cycle_holdouts_pass"]
        and numerical["all_physical_fma_holdouts_pass"]
        and numerical["all_qkv_utilizations_pass"]
    )
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
        "frozen_inputs": files,
        "frozen_sources": source_inputs,
        "parent_checks": parent_checks,
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
        "qkv_utilizations": qkv_utilizations,
        "summary": summary,
        "source_files": source_files,
        "integrity_checks": integrity_checks,
        "paper_performance_targets_consumed": False,
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
        keys = (
            "hypothesis_status",
            "audit_integrity",
            "full_work_checks",
            "models",
            "full_estimates",
            "summary",
            "integrity_checks",
        )
        matches = all(existing.get(key) == report.get(key) for key in keys)
        print(json.dumps({"existing_matches": matches, **report}, indent=2))
        return 0 if matches else 1
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0 if report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
