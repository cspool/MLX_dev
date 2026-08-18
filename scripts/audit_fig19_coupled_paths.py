#!/usr/bin/env python3
"""Audit H128 current-coupled Figure 19 path execution and folding."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from mlxsim.dsagen_overlay import canonical_json
from mlxsim.fig19_coupled_paths import compile_fig19_coupled_path
from mlxsim.repeat_folding import fit_affine
from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig19_coupled_paths_v1.yaml"
RESOURCES = ("compute", "load", "store", "xfer")


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    h98 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h98"]["path"]).read_text()
    )
    h120 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h120"]["path"]).read_text()
    )
    parent_manifest = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h98_manifest"]["path"]).read_text()
    )
    parent_checks = {
        "h98": h98["hypothesis_status"] == "supported"
        and h98["audit_integrity"] is True,
        "h120": h120["hypothesis_status"] == "supported"
        and h120["audit_integrity"] is True
        and all(h120["target_free_checks"].values()),
    }
    output_root = PROJECT_ROOT / config["output_root"]
    compile_path = output_root / "fig19-coupled-compile-manifest.json"
    run_path = output_root / "fig19-coupled-run-manifest.json"
    compile_file = qualify(compile_path)
    run_file = qualify(run_path)
    compiled = json.loads(compile_path.read_text())
    run = json.loads(run_path.read_text())
    records = {
        (item["key"], item["mode"], int(item["replay"])): item
        for item in run["records"]
    }
    compile_checks: dict[str, bool] = {}
    transform_checks: dict[str, bool] = {}
    execution_checks: dict[str, bool] = {}
    work_checks: dict[str, bool] = {}
    port_checks: dict[str, bool] = {}
    record_checks: dict[str, bool] = {}
    measurements: dict[str, Any] = {}
    for run_key, item in compiled["outputs"].items():
        parent_item = parent_manifest["outputs"][run_key]
        source_path = PROJECT_ROOT / parent_item["artifact"]["path"]
        source = json.loads(source_path.read_text())
        overlay, memory_document, metadata = compile_fig19_coupled_path(
            run_key=run_key,
            source=source,
            source_metadata=parent_item["metadata"],
            config=config,
        )
        overlay_path = PROJECT_ROOT / item["overlay"]["path"]
        memory_path = PROJECT_ROOT / item["memory"]["path"]
        compile_checks[run_key] = (
            qualify(PROJECT_ROOT / item["parent"]["path"], item["parent"])["pass"]
            and qualify(overlay_path, item["overlay"])["pass"]
            and qualify(memory_path, item["memory"])["pass"]
            and source_path.read_bytes() == (PROJECT_ROOT / item["parent"]["path"]).read_bytes()
            and overlay_path.read_text() == canonical_json(overlay)
            and memory_path.read_text() == canonical_json(memory_document)
            and item["metadata"] == metadata
            and all(metadata["checks"].values())
        )
        transform_checks[run_key] = (
            metadata["source_block_count"]
            * metadata["tile_count"]
            == metadata["coupled_block_count"]
            and metadata["source_tag_count"]
            * metadata["tile_count"]
            == metadata["coupled_tag_count"]
            and metadata["max_active_instruction_footprint_per_pe"]
            <= int(config["hardware"]["instruction_slots_per_pe"])
            and memory_document["spad_port_axis"]
            == config["hardware"]["operator_axis"][metadata["family"]]
        )
        record = records[(run_key, "optimized", 1)]
        summary = record["summary"]
        overlay_summary = summary["overlay"]
        memory = summary["memory"]
        record_checks[run_key] = qualify(
            PROJECT_ROOT / record["summary_path"],
            {"sha256": record["summary_sha256"]},
        )["pass"]
        execution_checks[run_key] = (
            overlay_summary["done"] is True
            and memory["idle"] is True
            and overlay_summary["pe_dependency_model"] == "dpu_pipelined"
            and overlay_summary["memory_backend"] == "dpu_memory"
            and overlay_summary["iteration_contexts_per_block"]
            == int(config["hardware"]["iteration_contexts_per_block"])
            and overlay_summary["max_active_tags"]
            <= int(config["hardware"]["active_window"])
            and summary["end_to_end_cycles"] > 0
        )
        work_checks[run_key] = (
            overlay_summary["instructions_issued"]
            == overlay_summary["instructions_completed"]
            == sum(metadata["pipeline_counts"].values())
            and overlay_summary["issued_by_pipeline"] == metadata["pipeline_counts"]
            and overlay_summary["boundary_events_emitted"]
            == metadata["coupled_dynamic_event_count"]
            and overlay_summary["external_memory_requests"]
            == overlay_summary["external_memory_completions"]
            == memory["requests"]
            == memory["responses"]
            == metadata["memory_requests"]
            and memory["read_requests"] == metadata["pipeline_counts"]["load"]
            and memory["write_requests"] == metadata["pipeline_counts"]["store"]
            and memory["offchip_read_bytes"] == metadata["input_bytes"]
            and memory["offchip_write_bytes"] == metadata["output_bytes"]
            and memory["released_tiles"]
            == memory["drained_tiles"]
            == metadata["tile_count"]
            and memory["ownership_violations"] == 0
        )
        spad = memory["spad"]
        per_port = spad["per_port"]
        port_checks[run_key] = (
            spad["ports"] == int(config["hardware"]["spad_ports"])
            and spad["axis"] == memory_document["spad_port_axis"]
            and len(per_port) == int(config["hardware"]["spad_ports"])
            and sum(port["requests"] for port in per_port) == spad["requests"]
            and sum(port["responses"] for port in per_port) == spad["responses"]
            and spad["requests"] == spad["responses"] == memory["requests"]
            and any(port["requests"] > 0 for port in per_port)
        )
        productive = overlay_summary["productive_pe_cycles_by_pipeline"]
        capacity = summary["end_to_end_cycles"] * int(config["hardware"]["physical_pes"])
        measurements[run_key] = {
            "path_key": metadata["path_key"],
            "family": metadata["family"],
            "scale": metadata["scale"],
            "cycles": summary["end_to_end_cycles"],
            "overlay_cycles": summary["overlay_cycles"],
            "productive_utilization": {
                name: productive[name] / capacity for name in RESOURCES
            },
            "tile_count": metadata["tile_count"],
        }
    fit_scales = [int(value) for value in config["scales"]["fit"]]
    holdout_scales = [int(value) for value in config["scales"]["holdout"]]
    limit = float(config["scales"]["cycle_relative_error_limit"])
    path_keys = sorted(compiled["path_contracts"])
    models: dict[str, Any] = {}
    errors = []
    full_estimates: dict[str, Any] = {}
    for path_key in path_keys:
        model = fit_affine(
            fit_scales[0],
            measurements[f"{path_key}-q{fit_scales[0]}"]["cycles"],
            fit_scales[1],
            measurements[f"{path_key}-q{fit_scales[1]}"]["cycles"],
        )
        holdouts = []
        for scale in holdout_scales:
            actual = measurements[f"{path_key}-q{scale}"]["cycles"]
            prediction = model.predict(scale)
            error = abs(prediction - actual) / actual
            errors.append(error)
            holdouts.append(
                {
                    "scale": scale,
                    "actual": actual,
                    "prediction": prediction,
                    "relative_error": error,
                    "pass_5pct": error <= limit,
                }
            )
        eligible = all(item["pass_5pct"] for item in holdouts)
        full_scale = int(
            measurements[f"{path_key}-q{fit_scales[0]}"]["scale"]
            * 0
            + compiled["outputs"][f"{path_key}-q{fit_scales[0]}"]["metadata"][
                "full_scale"
            ]
        )
        cycles = model.predict(full_scale)
        models[path_key] = {
            "intercept": model.intercept,
            "slope": model.slope,
            "holdouts": holdouts,
            "eligible": eligible,
        }
        full_estimates[path_key] = {
            "cycles": cycles if eligible else None,
            "old_h98_cycles": h98["full_estimates"][path_key],
            "speedup_vs_h98": (
                h98["full_estimates"][path_key] / cycles if eligible else None
            ),
            "eligible": eligible,
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
    target_free_checks = {
        "config": config["execution"]["targets_consumed"] is False,
        "compile": compiled["paper_performance_targets_consumed"] is False,
        "run": run["paper_performance_targets_consumed"] is False,
        "no_target": "fig19" + "_components" not in source_text,
        "no_residual": "residual" + "_factor" not in source_text,
    }
    counts = {
        "paths": len(path_keys) == int(config["execution"]["required_paths"]),
        "configs": len(compiled["outputs"])
        == int(config["execution"]["required_configs"]),
        "records": len(run["records"])
        == int(config["execution"]["required_executions"]),
        "optimized": sum(item["mode"] == "optimized" for item in run["records"])
        == int(config["execution"]["required_optimized_executions"]),
        "sanitizers": sum(item["mode"] in {"asan", "ubsan"} for item in run["records"])
        == int(config["execution"]["required_sanitizer_executions"]),
        "holdouts": len(errors) == int(config["execution"]["required_holdouts"]),
    }
    all_holdouts_pass = all(
        holdout["pass_5pct"]
        for model in models.values()
        for holdout in model["holdouts"]
    )
    utilization_checks = all(
        math.isfinite(value) and 0 <= value <= 1
        for measurement in measurements.values()
        for value in measurement["productive_utilization"].values()
    )
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        all(compile_checks.values()) and counts["configs"] and counts["paths"],
        all(transform_checks.values()),
        all(item["metadata"]["checks"]["capacity"] for item in compiled["outputs"].values()),
        all(run["checks"].values())
        and all(run["replay_checks"].values())
        and all(run["sanitizer_checks"].values()),
        all(work_checks.values()) and all(execution_checks.values()),
        all(port_checks.values()),
        all_holdouts_pass,
        len(full_estimates) == 12
        and all(item["cycles"] is not None for item in full_estimates.values()),
        all(target_free_checks.values()) and all(item["pass"] for item in source_files.values()),
        h120["hypothesis_status"] == "supported" and utilization_checks,
        config["validation_eligible"] is False,
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "compile_manifest": compile_file["pass"] and all(compiled["checks"].values()),
        "run_manifest": run_file["pass"] and all(run["checks"].values()),
        "compile": all(compile_checks.values()),
        "transform": all(transform_checks.values()),
        "execution": all(execution_checks.values()),
        "work": all(work_checks.values()),
        "ports": all(port_checks.values()),
        "records": all(record_checks.values()),
        "models_evaluated": len(models) == 12,
        "counts": all(counts.values()),
        "source": all(target_free_checks.values())
        and all(item["pass"] for item in source_files.values()),
        "acceptance_evaluated": len(acceptance_gates) == 12
        and all(isinstance(value, bool) for value in acceptance_gates),
    }
    integrity = all(integrity_checks.values())
    supported = integrity and all(acceptance_gates)
    finite_errors = [error for error in errors if math.isfinite(error)]
    speedups = [
        item["speedup_vs_h98"]
        for item in full_estimates.values()
        if item["speedup_vs_h98"] is not None
    ]
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
        "paper_reproduction_claim": "none_target_free_current_coupled_paths_only",
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "compile_manifest": compile_file,
        "run_manifest": run_file,
        "compile_checks": compile_checks,
        "transform_checks": transform_checks,
        "execution_checks": execution_checks,
        "work_checks": work_checks,
        "port_checks": port_checks,
        "record_checks": record_checks,
        "measurements": measurements,
        "models": models,
        "full_estimates": full_estimates,
        "counts": counts,
        "target_free_checks": target_free_checks,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "paths": len(path_keys),
            "configs": len(compiled["outputs"]),
            "executions": len(run["records"]),
            "sanitizer_executions": sum(
                item["mode"] in {"asan", "ubsan"} for item in run["records"]
            ),
            "holdouts": len(errors),
            "holdouts_passed": sum(error <= limit for error in errors),
            "holdout_mape": sum(finite_errors) / len(finite_errors),
            "holdout_max_error": max(finite_errors),
            "full_estimates": len(full_estimates),
            "speedup_vs_h98_minimum": min(speedups) if speedups else None,
            "speedup_vs_h98_maximum": max(speedups) if speedups else None,
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
            "models",
            "full_estimates",
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
