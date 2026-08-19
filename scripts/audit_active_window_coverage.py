#!/usr/bin/env python3
"""Audit H165's target-free active-tag coverage sweep."""

from __future__ import annotations

import argparse
import ast
import copy
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from mlxsim.active_window_coverage import compile_active_window_path
from mlxsim.dsagen_overlay import canonical_json
from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/simulators/active_window_coverage_sweep_v1.yaml"
)
PIPELINES = ("compute", "load", "store", "xfer")


def _work_signature(summary: dict[str, Any]) -> dict[str, Any]:
    overlay = summary["overlay"]
    memory = summary["memory"]
    return {
        "instructions_issued": overlay["instructions_issued"],
        "instructions_completed": overlay["instructions_completed"],
        "issued_by_pipeline": overlay["issued_by_pipeline"],
        "boundary_events_emitted": overlay["boundary_events_emitted"],
        "route_hops": overlay["route_hops"],
        "skip_hops": overlay["skip_hops"],
        "unit_hops": overlay["unit_hops"],
        "external_memory_requests": overlay["external_memory_requests"],
        "external_memory_completions": overlay["external_memory_completions"],
        "memory_requests": memory["requests"],
        "memory_responses": memory["responses"],
        "offchip_read_bytes": memory["offchip_read_bytes"],
        "offchip_write_bytes": memory["offchip_write_bytes"],
    }


def _h120_projection(summary: dict[str, Any]) -> dict[str, Any]:
    projected = copy.deepcopy(summary)
    functional = projected["overlay"].pop("functional", None)
    if functional is not None and functional != {
        "enabled": False,
        "operations": 0,
        "nan_values": 0,
        "errors": 0,
    }:
        raise ValueError("non-neutral functional field in timing-only sweep")
    return projected


def _source_audit(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, bool]]:
    source_names = ("compiler_core", "compiler", "runner", "auditor", "test")
    source_files = {
        name: qualify(PROJECT_ROOT / config["source_layout"][name])
        for name in source_names
    }
    source_text = "\n".join(
        (PROJECT_ROOT / config["source_layout"][name]).read_text(errors="replace")
        for name in source_names
    )
    forbidden = tuple(config["target_exclusion"]["forbidden_paths"])
    tree = ast.parse((PROJECT_ROOT / config["source_layout"]["auditor"]).read_text())
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    checks = {
        "no_target_paths": not any(path in source_text for path in forbidden),
        "no_fit_calls": calls.isdisjoint(
            {"polyfit", "curve_fit", "lstsq", "minimize", "least_squares"}
        ),
        "source_files": all(item["pass"] for item in source_files.values()),
    }
    return source_files, checks


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    h120 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h120"]["path"]).read_text()
    )
    h120_run = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h120_run"]["path"]).read_text()
    )
    h120_config = yaml.safe_load(
        (PROJECT_ROOT / config["frozen_inputs"]["h120_config"]["path"]).read_text()
    )
    h118_config = yaml.safe_load(
        (PROJECT_ROOT / config["frozen_inputs"]["h118_config"]["path"]).read_text()
    )
    paper_text = (
        PROJECT_ROOT / config["frozen_inputs"]["mlx_paper_text"]["path"]
    ).read_text()
    parent_checks = {
        "h120": h120["hypothesis_status"]
        == config["frozen_inputs"]["h120"]["required_status"]
        and h120["audit_integrity"]
        is config["frozen_inputs"]["h120"]["required_integrity"],
        "h120_target_free": h120["paper_performance_targets_consumed"] is False,
        "h120_run": all(h120_run["checks"].values())
        and h120_run["paper_performance_targets_consumed"] is False,
        "paper_coverage_condition": "B_T \\cdot C \\geq T_{\\text{load}} + T_{\\text{xfer}}"
        in paper_text,
        "paper_instruction_capacity": "32 instructions per PE" in paper_text,
    }

    output_root = PROJECT_ROOT / config["output_root"]
    compile_path = output_root / "active-window-compile-manifest.json"
    run_path = output_root / "active-window-run-manifest.json"
    compile_file = qualify(compile_path)
    run_file = qualify(run_path)
    compiled = json.loads(compile_path.read_text())
    run = json.loads(run_path.read_text())
    records = {
        (item["key"], item["mode"], int(item["replay"])): item
        for item in run["records"]
    }

    compile_checks: dict[str, bool] = {}
    config_invariant_checks: dict[str, bool] = {}
    for key, item in compiled["outputs"].items():
        metadata = item["metadata"]
        overlay, memory, rebuilt = compile_active_window_path(
            metadata["operator"],
            int(metadata["size"]),
            int(metadata["window"]),
            config,
            h120_config,
            h118_config,
        )
        overlay_path = PROJECT_ROOT / item["overlay"]["path"]
        memory_path = PROJECT_ROOT / item["memory"]["path"]
        compile_checks[key] = (
            qualify(overlay_path, item["overlay"])["pass"]
            and qualify(memory_path, item["memory"])["pass"]
            and overlay_path.read_text() == canonical_json(overlay)
            and memory_path.read_text() == canonical_json(memory)
            and rebuilt == metadata
            and all(metadata["checks"].values())
        )
        config_invariant_checks[key] = (
            overlay["active_window"] == int(metadata["window"])
            and overlay["memory_backend"] == "dpu_memory"
            and overlay["pe_dependency_model"] == "dpu_pipelined"
            and memory["spad_ports"] == config["hardware_invariants"]["spad_ports"]
            and memory["dma_bytes_per_cycle"]
            == config["hardware_invariants"]["dma_bytes_per_cycle"]
            and memory["dma_setup_cycles"]
            == config["hardware_invariants"]["dma_setup_cycles"]
        )

    expected_maxima = {
        str(key): int(value)
        for key, value in config["window_sweep"][
            "expected_max_footprint_by_window"
        ].items()
    }
    expected_feasible = {
        str(window): int(window)
        in {int(value) for value in config["window_sweep"]["globally_feasible_windows"]}
        for window in config["window_sweep"]["compiled_windows"]
    }
    static_checks = {
        "outputs": len(compiled["outputs"]) == 128,
        "manifest": all(compiled["checks"].values()),
        "maxima": compiled["maximum_footprint_by_window"] == expected_maxima,
        "feasibility": compiled["global_feasibility_by_window"]
        == expected_feasible,
        "selected": int(config["window_sweep"]["selected_candidate_window"])
        == max(int(value) for value in config["window_sweep"]["globally_feasible_windows"]),
    }

    feasible = [int(value) for value in config["window_sweep"]["globally_feasible_windows"]]
    workload_keys = sorted(
        f"{operator}-{int(size)}"
        for operator in config["workloads"]["operators"]
        for size in config["workloads"]["sizes"]
    )
    record_checks: dict[str, bool] = {}
    execution_checks: dict[str, bool] = {}
    counter_checks: dict[str, bool] = {}
    measurements: dict[str, Any] = {}
    signatures: dict[str, dict[int, dict[str, Any]]] = {
        workload: {} for workload in workload_keys
    }
    for workload in workload_keys:
        measurements[workload] = {}
        for window in feasible:
            key = f"w{window}--{workload}"
            record = records[(key, "optimized", 1)]
            summary = record["summary"]
            overlay = summary["overlay"]
            memory = summary["memory"]
            record_checks[key] = qualify(
                PROJECT_ROOT / record["summary_path"],
                {"sha256": record["summary_sha256"]},
            )["pass"]
            execution_checks[key] = (
                record["pass"]
                and overlay["done"] is True
                and memory["idle"] is True
                and overlay["max_active_tags"] <= window
                and overlay["instructions_issued"] == overlay["instructions_completed"]
                and overlay["external_memory_requests"]
                == overlay["external_memory_completions"]
                == memory["requests"]
                == memory["responses"]
            )
            capacity = float(summary["end_to_end_cycles"] * 16)
            productive = overlay["productive_pe_cycles_by_pipeline"]
            resident = overlay["resident_pe_cycles_by_pipeline"]
            spad = memory["spad"]
            counter_checks[key] = (
                all(
                    math.isfinite(float(productive[pipeline]))
                    and 0 <= productive[pipeline] <= resident[pipeline] <= capacity
                    for pipeline in PIPELINES
                )
                and len(spad["per_port"]) == 4
                and spad["requests"] == spad["responses"]
                == sum(port["requests"] for port in spad["per_port"])
                and all(
                    math.isfinite(float(value)) and value >= 0
                    for group in (
                        overlay["productive_global_cycles_by_pipeline"],
                        overlay["issue_cycles_by_pipeline"],
                        overlay["productive_pe_cycles_by_fu_class"],
                    )
                    for value in group.values()
                )
            )
            signatures[workload][window] = _work_signature(summary)
            measurements[workload][str(window)] = {
                "end_to_end_cycles": summary["end_to_end_cycles"],
                "overlay_cycles": summary["overlay_cycles"],
                "max_active_tags": overlay["max_active_tags"],
                "max_active_blocks_per_pe": overlay["max_active_blocks_per_pe"],
                "productive_pe_cycles_by_pipeline": productive,
                "primary_capacity_fraction": {
                    pipeline: productive[pipeline] / capacity
                    for pipeline in PIPELINES
                },
            }

    work_checks = {
        workload: all(
            signature == signatures[workload][feasible[0]]
            for signature in signatures[workload].values()
        )
        for workload in workload_keys
    }
    h120_records = {
        item["key"]: item
        for item in h120_run["records"]
        if item["mode"] == "optimized" and int(item["replay"]) == 1
    }
    window3_checks = {
        workload: _h120_projection(
            records[(f"w3--{workload}", "optimized", 1)]["summary"]
        )
        == h120_records[workload]["summary"]
        for workload in workload_keys
    }
    selected = int(config["window_sweep"]["selected_candidate_window"])
    current = int(config["window_sweep"]["current_window"])
    comparisons: dict[str, Any] = {}
    for workload in workload_keys:
        current_cycles = measurements[workload][str(current)]["end_to_end_cycles"]
        selected_cycles = measurements[workload][str(selected)]["end_to_end_cycles"]
        current_overlay = measurements[workload][str(current)]["overlay_cycles"]
        selected_overlay = measurements[workload][str(selected)]["overlay_cycles"]
        comparisons[workload] = {
            "current_window": current,
            "selected_window": selected,
            "current_end_to_end_cycles": current_cycles,
            "selected_end_to_end_cycles": selected_cycles,
            "cycle_speedup": current_cycles / selected_cycles,
            "end_to_end_non_regression": selected_cycles <= current_cycles,
            "strict_end_to_end_improvement": selected_cycles < current_cycles,
            "current_overlay_cycles": current_overlay,
            "selected_overlay_cycles": selected_overlay,
            "overlay_non_regression": selected_overlay <= current_overlay,
        }
    non_regressions = sum(
        item["end_to_end_non_regression"] and item["overlay_non_regression"]
        for item in comparisons.values()
    )
    strict_improvements = sum(
        item["strict_end_to_end_improvement"] for item in comparisons.values()
    )
    speedups = [item["cycle_speedup"] for item in comparisons.values()]
    source_files, source_checks = _source_audit(config)
    target_free_checks = {
        "compile": compiled["paper_performance_targets_consumed"] is False,
        "run": run["paper_performance_targets_consumed"] is False,
        "source": all(source_checks.values()),
        "parents": set(config["frozen_inputs"])
        == {
            "h120",
            "h120_run",
            "h120_config",
            "h118_config",
            "mlx_paper_text",
            "source_refresh",
        },
    }
    counts = {
        "compile": len(compiled["outputs"]) == 128,
        "records": len(run["records"]) == int(config["execution"]["required_executions"]),
        "optimized": sum(item["mode"] == "optimized" for item in run["records"])
        == int(config["execution"]["required_optimized_executions"]),
        "sanitizers": sum(item["mode"] in {"asan", "ubsan"} for item in run["records"])
        == int(config["execution"]["required_sanitizer_executions"]),
        "measurements": len(measurements) == 16
        and all(len(item) == 5 for item in measurements.values()),
    }
    candidate_gate = non_regressions == 16 and strict_improvements >= 1
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        all(compile_checks.values()) and all(static_checks.values()),
        all(run["checks"].values()) and all(counts.values()),
        all(work_checks.values()) and all(execution_checks.values()),
        all(window3_checks.values()),
        candidate_gate,
        all(counter_checks.values()),
        all(config_invariant_checks.values()),
        all(target_free_checks.values())
        and all(item["pass"] for item in source_files.values()),
        config["validation_eligible"] is True
        and config["execution"]["paper_performance_targets_consumed"] is False,
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "compile_manifest": compile_file["pass"] and all(compiled["checks"].values()),
        "run_manifest": run_file["pass"] and all(run["checks"].values()),
        "compile": all(compile_checks.values()),
        "static": all(static_checks.values()),
        "records": all(record_checks.values()),
        "execution": all(execution_checks.values()),
        "work": all(work_checks.values()),
        "window3": all(window3_checks.values()),
        "counters": all(counter_checks.values()),
        "config_invariants": all(config_invariant_checks.values()),
        "target_free": all(target_free_checks.values()),
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
        "paper_reproduction_claim": "none_target_free_schedule_candidate",
        "candidate_status": (
            "accepted_for_held_out_transfer" if supported else "rejected_non_regression"
        ),
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "compile_manifest": compile_file,
        "run_manifest": run_file,
        "static_checks": static_checks,
        "compile_checks": compile_checks,
        "config_invariant_checks": config_invariant_checks,
        "record_checks": record_checks,
        "execution_checks": execution_checks,
        "work_checks": work_checks,
        "window3_checks": window3_checks,
        "counter_checks": counter_checks,
        "measurements": measurements,
        "comparisons": comparisons,
        "source_files": source_files,
        "source_checks": source_checks,
        "target_free_checks": target_free_checks,
        "counts": counts,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "compiled_paths": len(compiled["outputs"]),
            "executions": len(run["records"]),
            "workloads": len(workload_keys),
            "executed_windows": feasible,
            "selected_window": selected,
            "current_window": current,
            "candidate_non_regressions": non_regressions,
            "candidate_strict_improvements": strict_improvements,
            "candidate_min_speedup": min(speedups),
            "candidate_median_speedup": statistics.median(speedups),
            "candidate_max_speedup": max(speedups),
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
            "candidate_status",
            "static_checks",
            "measurements",
            "comparisons",
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
    print(
        json.dumps(
            {
                "status": report["hypothesis_status"],
                "candidate_status": report["candidate_status"],
                **report["summary"],
            },
            indent=2,
        )
    )
    return 0 if report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
