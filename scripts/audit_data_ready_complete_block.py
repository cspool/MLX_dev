#!/usr/bin/env python3
"""Audit H171's data-ready MLX versus one serial spatial baseline."""

from __future__ import annotations

import argparse
import copy
import json
import math
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_complete_block_functional import (
    boundary_addresses,
    full_chain_reference,
    without_functional,
)
from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify
from scripts.compile_data_ready_complete_block import build_documents

DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/data_ready_complete_block_v1.yaml"


def normalize_mlx_to_baseline(
    baseline: dict[str, Any], mlx: dict[str, Any]
) -> tuple[dict[str, Any], int]:
    normalized = copy.deepcopy(mlx)
    removed_restored = 0
    for previous, current in pairwise(normalized["metadata"]["components"]):
        previous_final = int(previous["tag_range"][1])
        current_first = int(current["tag_range"][0])
        for block in normalized["blocks"]:
            if int(block["tag"]) != current_first:
                continue
            if previous_final not in block["predecessors"]:
                block["predecessors"].append(previous_final)
                block["predecessors"].sort()
                removed_restored += 1
    normalized["active_window"] = baseline["active_window"]
    for field in ("architecture", "active_window", "boundary_mode"):
        normalized["metadata"][field] = baseline["metadata"][field]
    normalized["metadata"]["data_ready"]["coarse_predecessors_removed"] = 0
    return normalized, removed_restored


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    h170 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h170"]["path"]).read_text()
    )
    h161 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h161"]["path"]).read_text()
    )
    h170_config = yaml.safe_load(
        (PROJECT_ROOT / config["frozen_inputs"]["h170_config"]["path"]).read_text()
    )
    h161_config = yaml.safe_load(
        (
            PROJECT_ROOT
            / h170_config["frozen_inputs"]["h161_config"]["path"]
        ).read_text()
    )
    parent_checks = {
        "h170": h170["hypothesis_status"]
        == config["frozen_inputs"]["h170"]["required_status"]
        and h170["audit_integrity"]
        is config["frozen_inputs"]["h170"]["required_integrity"],
        "h170_negative_retained": h170["summary"]["complete_block_speedup"]
        < h170["summary"]["minimum_clear_speedup"]
        and h170["summary"]["both_architectures_functionally_correct"] is True,
        "h161": h161["hypothesis_status"]
        == config["frozen_inputs"]["h161"]["required_status"]
        and h161["audit_integrity"]
        is config["frozen_inputs"]["h161"]["required_integrity"],
        "target_free": h170["paper_performance_targets_consumed"] is False
        and h161["paper_performance_targets_consumed"] is False,
    }
    output_root = PROJECT_ROOT / config["output_root"]
    compile_path = output_root / "data-ready-complete-compile-manifest.json"
    run_path = output_root / "data-ready-complete-run-manifest.json"
    compile_file = qualify(compile_path)
    run_file = qualify(run_path)
    compiler = json.loads(compile_path.read_text())
    run = json.loads(run_path.read_text())
    rebuilt = build_documents(config)
    compile_checks: dict[str, bool] = {}
    for key, item in compiler["outputs"].items():
        path = PROJECT_ROOT / item["artifact"]["path"]
        document = json.loads(path.read_text())
        compile_checks[key] = (
            qualify(path, item["artifact"])["pass"]
            and document == rebuilt[key]
            and document["metadata"]["schedule_counts"] == item["schedule_counts"]
            and item["deterministic"] is True
        )
    baseline_id = config["baseline"]["id"]
    mlx_id = config["mlx"]["id"]
    pair_checks: dict[str, bool] = {}
    input_checks: dict[str, bool] = {}
    work_checks: dict[str, bool] = {}
    mode_checks: dict[str, bool] = {}
    predecessor_checks: dict[str, Any] = {}
    event_checks: dict[str, bool] = {}
    for prefix in config["prefixes"]:
        for mode in config["execution"]["functional_modes"]:
            baseline_key = f"{prefix}--{baseline_id}--{mode}"
            mlx_key = f"{prefix}--{mlx_id}--{mode}"
            baseline_document = rebuilt[baseline_key]
            mlx_document = rebuilt[mlx_key]
            normalized, restored = normalize_mlx_to_baseline(
                baseline_document, mlx_document
            )
            pair_checks[f"{prefix}-{mode}"] = normalized == baseline_document
            input_checks[f"{prefix}-{mode}"] = compiler["outputs"][baseline_key][
                "input_memory_sha256"
            ] == compiler["outputs"][mlx_key]["input_memory_sha256"]
            work_checks[f"{prefix}-{mode}"] = compiler["outputs"][baseline_key][
                "schedule_counts"
            ] == compiler["outputs"][mlx_key]["schedule_counts"]
            baseline_data = baseline_document["metadata"]["data_ready"]
            mlx_data = mlx_document["metadata"]["data_ready"]
            event_checks[f"{prefix}-{mode}"] = (
                baseline_data["linked_addresses"] == mlx_data["linked_addresses"]
                and baseline_data["event_definitions"]
                == mlx_data["event_definitions"]
                and baseline_data["event_emissions"] == mlx_data["event_emissions"]
                and compiler["outputs"][baseline_key]["schedule_counts"][
                    "boundary_events"
                ]
                == compiler["outputs"][mlx_key]["schedule_counts"][
                    "boundary_events"
                ]
            )
            predecessor_checks[f"{prefix}-{mode}"] = {
                "restored": restored,
                "mlx_registered": mlx_data["coarse_predecessors_removed"],
                "baseline_registered": baseline_data[
                    "coarse_predecessors_removed"
                ],
                "pass": restored == mlx_data["coarse_predecessors_removed"]
                and baseline_data["coarse_predecessors_removed"] == 0,
            }
        baseline_enabled = rebuilt[f"{prefix}--{baseline_id}--enabled"]
        baseline_disabled = rebuilt[f"{prefix}--{baseline_id}--disabled"]
        mlx_enabled = rebuilt[f"{prefix}--{mlx_id}--enabled"]
        mlx_disabled = rebuilt[f"{prefix}--{mlx_id}--disabled"]
        mode_checks[prefix] = (
            {**baseline_enabled["functional_execution"], "enabled": False}
            == baseline_disabled["functional_execution"]
            and {**mlx_enabled["functional_execution"], "enabled": False}
            == mlx_disabled["functional_execution"]
        )
    full_prefix_key = "complete-enabled"
    boundary_contract = config["boundary_contract"]
    full_static_checks = {
        "linked_addresses": rebuilt[f"complete--{mlx_id}--enabled"]["metadata"][
            "data_ready"
        ]["linked_addresses"]
        == int(boundary_contract["linked_addresses"]),
        "event_definitions": rebuilt[f"complete--{mlx_id}--enabled"]["metadata"][
            "data_ready"
        ]["event_definitions"]
        == int(boundary_contract["store_event_definitions"]),
        "event_emissions": rebuilt[f"complete--{mlx_id}--enabled"]["metadata"][
            "data_ready"
        ]["event_emissions"]
        == int(boundary_contract["store_event_emissions"]),
        "boundary_events": compiler["outputs"][f"complete--{mlx_id}--enabled"][
            "schedule_counts"
        ]["boundary_events"]
        == int(boundary_contract["expected_boundary_events"]),
        "removed": predecessor_checks[full_prefix_key]["restored"]
        == int(boundary_contract["mlx_removed_coarse_predecessors"]),
    }
    run_checks = {
        "experiment": run["experiment_id"] == config["experiment_id"],
        "target_free": run["paper_performance_targets_consumed"] is False,
        "keys": set(run["records"]) == set(compiler["outputs"]),
        "count": sum(len(builds) for builds in run["records"].values())
        == int(config["execution"]["expected_runs"]),
        "checks": all(run["checks"].values()),
    }
    execution_checks: dict[str, bool] = {}
    for key, builds in run["records"].items():
        expected_work = compiler["outputs"][key]["schedule_counts"]
        execution_checks[key] = (
            set(builds) == set(config["execution"]["builds"])
            and all(item["pass"] and item["stderr_bytes"] == 0 for item in builds.values())
            and all(
                item["summary"]["done"] is True
                and item["summary"]["instructions_issued"]
                == item["summary"]["instructions_completed"]
                == expected_work["functional_operations"]
                and item["summary"]["issued_by_pipeline"]
                == expected_work["pipelines"]
                and item["summary"]["boundary_events_emitted"]
                == expected_work["boundary_events"]
                and item["summary"]["route_hops"] == expected_work["route_hops"]
                for item in builds.values()
            )
        )
    reference = full_chain_reference(h161_config)
    addresses = boundary_addresses()
    numeric_checks: dict[str, bool] = {}
    pair_output_checks: dict[str, bool] = {}
    numeric_details: dict[str, Any] = {}
    limit = float(config["acceptance"]["absolute_error_limit"])
    for prefix, prefix_spec in config["prefixes"].items():
        components = prefix_spec["required_components"]
        numeric_details[prefix] = {}
        outputs_by_architecture: dict[str, dict[str, list[float]]] = {}
        for architecture in (baseline_id, mlx_id):
            key = f"{prefix}--{architecture}--enabled"
            functional = run["records"][key]["opt"]["summary"]["functional"]
            actual = {
                component: [
                    float(functional["memory"][str(address)])
                    for address in addresses[component]
                ]
                for component in components
            }
            errors = [
                abs(value - expected)
                for component in components
                for value, expected in zip(
                    actual[component], reference[component], strict=True
                )
            ]
            maximum_error = max(errors)
            expected_operations = compiler["outputs"][key]["schedule_counts"][
                "functional_operations"
            ]
            numeric_checks[f"{prefix}-{architecture}"] = (
                functional["enabled"] is True
                and functional["operations"] == expected_operations
                and functional["nan_values"] == 0
                and functional["errors"] == 0
                and maximum_error <= limit
            )
            outputs_by_architecture[architecture] = actual
            numeric_details[prefix][architecture] = {
                "components": components,
                "maximum_absolute_error": maximum_error,
                "final_output_count": len(actual[prefix_spec["final_component"]]),
                "functional_operations": functional["operations"],
            }
        pair_output_checks[prefix] = (
            outputs_by_architecture[baseline_id]
            == outputs_by_architecture[mlx_id]
        )
    timing_identity_checks: dict[str, bool] = {}
    performance: dict[str, Any] = {}
    for prefix in config["prefixes"]:
        for architecture in (baseline_id, mlx_id):
            enabled = run["records"][f"{prefix}--{architecture}--enabled"]["opt"][
                "summary"
            ]
            disabled = run["records"][f"{prefix}--{architecture}--disabled"][
                "opt"
            ]["summary"]
            timing_identity_checks[f"{prefix}-{architecture}"] = (
                without_functional(enabled) == without_functional(disabled)
            )
        baseline_summary = run["records"][
            f"{prefix}--{baseline_id}--enabled"
        ]["opt"]["summary"]
        mlx_summary = run["records"][f"{prefix}--{mlx_id}--enabled"]["opt"][
            "summary"
        ]
        speedup = baseline_summary["cycles"] / mlx_summary["cycles"]
        performance[prefix] = {
            "baseline_cycles": baseline_summary["cycles"],
            "mlx_cycles": mlx_summary["cycles"],
            "speedup": speedup,
            "clear_improvement": speedup
            >= float(config["acceptance"]["minimum_clear_speedup"]),
            "mlx_non_regression": speedup >= 1.0,
            "baseline_max_active_tags": baseline_summary["max_active_tags"],
            "mlx_max_active_tags": mlx_summary["max_active_tags"],
            "baseline_event_unblocked_before_tag_complete": baseline_summary[
                "event_unblocked_issues_before_tag_complete"
            ],
            "mlx_event_unblocked_before_tag_complete": mlx_summary[
                "event_unblocked_issues_before_tag_complete"
            ],
            "same_instructions": baseline_summary["instructions_completed"]
            == mlx_summary["instructions_completed"],
            "same_events": baseline_summary["boundary_events_emitted"]
            == mlx_summary["boundary_events_emitted"],
            "same_routes": baseline_summary["route_hops"]
            == mlx_summary["route_hops"],
        }
    complete_speedup = float(performance["complete"]["speedup"])
    performance_checks = {
        "finite": all(
            math.isfinite(item["speedup"]) and item["speedup"] > 0
            for item in performance.values()
        ),
        "all_non_regression": all(
            item["mlx_non_regression"] for item in performance.values()
        ),
        "complete_clear": complete_speedup
        >= float(config["acceptance"]["minimum_clear_speedup"]),
        "same_work": all(
            item["same_instructions"] and item["same_events"] and item["same_routes"]
            for item in performance.values()
        ),
        "baseline_serial": all(
            item["baseline_max_active_tags"] == 1
            and item["baseline_event_unblocked_before_tag_complete"] == 0
            for item in performance.values()
        ),
        "mlx_overlap": performance["complete"]["mlx_max_active_tags"] > 1
        and performance["complete"]["mlx_event_unblocked_before_tag_complete"] > 0,
        "improves_h170": complete_speedup
        > float(h170["summary"]["complete_block_speedup"]),
    }
    complete_counts = compiler["outputs"][f"complete--{mlx_id}--enabled"][
        "schedule_counts"
    ]
    complete_checks = {
        "operations": complete_counts["functional_operations"] == 466,
        "pipelines": complete_counts["pipelines"]
        == {"compute": 231, "load": 130, "store": 32, "xfer": 73},
        "memory": complete_counts["memory_requests"] == 162
        and complete_counts["memory_bytes"] == 1296,
        "events_routes": complete_counts["boundary_events"] == 97
        and complete_counts["route_hops"] == 139,
        "outputs": numeric_details["complete"][baseline_id]["final_output_count"]
        == numeric_details["complete"][mlx_id]["final_output_count"]
        == 8,
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    source_text = "\n".join(
        (PROJECT_ROOT / path).read_text(errors="replace")
        for path in config["source_layout"].values()
    )
    target_free_check = (
        config["execution"]["paper_performance_targets_consumed"] is False
        and "paper_" + "targets.yaml" not in source_text
        and "target_" + "factor" not in source_text
    )
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        len(compiler["outputs"]) == int(config["execution"]["expected_configs"])
        and all(compiler["checks"].values())
        and all(compile_checks.values())
        and all(run_checks.values())
        and all(execution_checks.values()),
        all(event_checks.values()) and all(full_static_checks.values()),
        all(item["pass"] for item in predecessor_checks.values()),
        all(pair_checks.values())
        and all(input_checks.values())
        and all(work_checks.values()),
        all(numeric_checks.values()) and all(pair_output_checks.values()),
        all(mode_checks.values()) and all(timing_identity_checks.values()),
        all(performance_checks.values()),
        all(complete_checks.values()),
        target_free_check and all(item["pass"] for item in source_files.values()),
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "generated": compile_file["pass"] and run_file["pass"],
        "compile_run": all(compile_checks.values())
        and all(run_checks.values())
        and all(execution_checks.values()),
        "events": all(event_checks.values()) and all(full_static_checks.values()),
        "predecessors": all(
            item["pass"] for item in predecessor_checks.values()
        ),
        "pairs": all(pair_checks.values())
        and all(input_checks.values())
        and all(work_checks.values()),
        "numeric": all(numeric_checks.values()) and all(pair_output_checks.values()),
        "timing": all(mode_checks.values()) and all(timing_identity_checks.values()),
        "performance": performance_checks["finite"]
        and performance_checks["same_work"]
        and performance_checks["baseline_serial"]
        and performance_checks["mlx_overlap"],
        "complete": all(complete_checks.values()),
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
        "paper_reproduction_claim": "complete_same_input_one_baseline_trend",
        "goal_claim": (
            "complete_one_baseline_functional_performance" if supported else "incomplete"
        ),
        "frozen_inputs": frozen,
        "generated_inputs": {"compile": compile_file, "run": run_file},
        "parent_checks": parent_checks,
        "compile_checks": compile_checks,
        "pair_checks": pair_checks,
        "input_checks": input_checks,
        "work_checks": work_checks,
        "mode_checks": mode_checks,
        "event_checks": event_checks,
        "predecessor_checks": predecessor_checks,
        "full_static_checks": full_static_checks,
        "run_checks": run_checks,
        "execution_checks": execution_checks,
        "numeric_checks": numeric_checks,
        "pair_output_checks": pair_output_checks,
        "numeric_details": numeric_details,
        "timing_identity_checks": timing_identity_checks,
        "performance": performance,
        "performance_checks": performance_checks,
        "complete_checks": complete_checks,
        "source_files": source_files,
        "target_free_check": target_free_check,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "prefixes": len(config["prefixes"]),
            "architectures": 2,
            "configs": len(compiler["outputs"]),
            "executions": sum(len(builds) for builds in run["records"].values()),
            "both_architectures_functionally_correct": all(numeric_checks.values()),
            "same_input_and_work": all(input_checks.values())
            and all(work_checks.values())
            and performance_checks["same_work"],
            "clear_improvement_prefixes": sum(
                item["clear_improvement"] for item in performance.values()
            ),
            "clear_improvement_prefix_total": len(performance),
            "complete_block_speedup": complete_speedup,
            "minimum_clear_speedup": float(
                config["acceptance"]["minimum_clear_speedup"]
            ),
            "complete_block_clear_improvement": performance_checks[
                "complete_clear"
            ],
            "maximum_functional_error": max(
                item[architecture]["maximum_absolute_error"]
                for item in numeric_details.values()
                for architecture in (baseline_id, mlx_id)
            ),
            "complete_functional_operations": complete_counts[
                "functional_operations"
            ],
            "complete_boundary_events": complete_counts["boundary_events"],
            "complete_route_hops": complete_counts["route_hops"],
            "complete_outputs": 8,
            "baseline_complete_max_active_tags": performance["complete"][
                "baseline_max_active_tags"
            ],
            "mlx_complete_max_active_tags": performance["complete"][
                "mlx_max_active_tags"
            ],
            "mlx_data_ready_issues_before_tag_complete": performance["complete"][
                "mlx_event_unblocked_before_tag_complete"
            ],
            "acceptance_gates_passed": sum(acceptance_gates),
            "acceptance_gates_total": len(acceptance_gates),
            "goal_complete": supported,
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
            "goal_claim",
            "predecessor_checks",
            "numeric_details",
            "performance",
            "performance_checks",
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
                "performance": report["performance"],
                **report["summary"],
            },
            indent=2,
        )
    )
    return 0 if report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
