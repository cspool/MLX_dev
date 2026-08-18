#!/usr/bin/env python3
"""Audit H156 same-input hierarchical BSMM functional execution."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify
from scripts.compile_bsmm_functional import output_address, schedule_counts

DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/bsmm_functional_v1.yaml"


def without_functional(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "functional"}


def numpy_golden(config: dict[str, Any]) -> tuple[list[float], list[list[float]], list[list[list[float]]]]:
    contract = config["operator_contract"]
    width = int(contract["width"])
    matrices = []
    for pairs, pair_weights in zip(
        contract["stage_pairs"], contract["stage_pair_weights"], strict=True
    ):
        matrix = np.zeros((width, width), dtype=np.float64)
        for indices, weights in zip(pairs, pair_weights, strict=True):
            matrix[np.ix_(indices, indices)] = np.asarray(weights, dtype=np.float64)
        matrices.append(matrix)
    expected: list[float] = []
    intermediates: list[list[float]] = []
    for values in contract["inputs"]:
        vector = np.asarray(values, dtype=np.float64)
        intermediate = matrices[0] @ vector
        output = matrices[1] @ intermediate
        intermediates.append(intermediate.tolist())
        expected.extend(output.tolist())
    return expected, intermediates, [matrix.tolist() for matrix in matrices]


def register_value(
    registers: list[dict[str, Any]],
    *,
    pe: list[int],
    tag: int,
    iteration: int,
    reg: int,
) -> float | None:
    matches = [
        float(item["value"])
        for item in registers
        if item["pe"] == pe
        and int(item["tag"]) == tag
        and int(item["iteration"]) == iteration
        and int(item["reg"]) == reg
    ]
    return matches[0] if len(matches) == 1 else None


def xfer_wiring_check(document: dict[str, Any], config: dict[str, Any]) -> bool:
    contract = config["operator_contract"]
    stage1_pairs = contract["stage_pairs"][1]
    stage1_placements = contract["placement"]["stage1"]
    source_blocks = [block for block in document["blocks"] if block["stage"] == 0]
    if len(source_blocks) != 2:
        return False
    seen = set()
    for block in source_blocks:
        xfers = [
            instruction
            for instruction in block["instructions"]
            if instruction["pipeline"] == "xfer"
        ]
        if len(xfers) != 2:
            return False
        for instruction in xfers:
            row = int(instruction["logical_output_row"])
            index = int(block["logical_pair"][row])
            matches = [
                (pair, indices.index(index))
                for pair, indices in enumerate(stage1_pairs)
                if index in indices
            ]
            if len(matches) != 1:
                return False
            destination_pair, destination_register = matches[0]
            expected_event = f"bsmm_s0_p{block['pair_id']}_i{index}_ready"
            if not (
                instruction["destination"] == stage1_placements[destination_pair]
                and int(instruction["destination_tag"]) == 2
                and int(instruction["destination_register"]) == destination_register
                and instruction["emit_event"] == expected_event
            ):
                return False
            seen.add(expected_event)
    wait_events = {
        event
        for block in document["blocks"]
        if block["stage"] == 1
        for event in block["wait_events"]
    }
    return len(seen) == 4 and seen == wait_events


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    parents = {
        name: json.loads((PROJECT_ROOT / spec["path"]).read_text())
        for name, spec in config["frozen_inputs"].items()
    }
    parent_checks = {
        name: parent["hypothesis_status"] == config["frozen_inputs"][name]["required_status"]
        and parent["audit_integrity"]
        is config["frozen_inputs"][name]["required_integrity"]
        for name, parent in parents.items()
    }
    output_root = PROJECT_ROOT / config["output_root"]
    compile_path = output_root / "bsmm-functional-compile-manifest.json"
    run_path = output_root / "bsmm-functional-run-manifest.json"
    compiler = json.loads(compile_path.read_text())
    run = json.loads(run_path.read_text())
    generated_inputs = {
        "compile_manifest": qualify(compile_path),
        "run_manifest": qualify(run_path),
    }
    compile_checks = {
        "experiment": compiler["experiment_id"] == "H156",
        "target_free": compiler["paper_performance_targets_consumed"] is False,
        "outputs": set(compiler["outputs"]) == {"enabled", "disabled"},
        "deterministic": all(item["deterministic"] for item in compiler["outputs"].values()),
        "files": all(
            qualify(PROJECT_ROOT / item["artifact"]["path"], item["artifact"])["pass"]
            for item in compiler["outputs"].values()
        ),
    }
    enabled_document = json.loads(
        (PROJECT_ROOT / compiler["outputs"]["enabled"]["artifact"]["path"]).read_text()
    )
    disabled_document = json.loads(
        (PROJECT_ROOT / compiler["outputs"]["disabled"]["artifact"]["path"]).read_text()
    )
    contract = config["operator_contract"]
    contract_checks = {
        "family": enabled_document["metadata"]["operator_family"] == "bsmm",
        "shape": int(contract["width"]) == 4
        and int(contract["stages"]) == 2
        and int(contract["batch"]) == 2,
        "blocks": len(enabled_document["blocks"]) == int(config["acceptance"]["expected_blocks"]),
        "tags": [block["tag"] for block in enabled_document["blocks"]] == [1, 1, 2, 2],
        "pes": {tuple(block["pe"]) for block in enabled_document["blocks"]}
        == {(0, 0), (0, 1), (1, 0), (1, 1)},
        "trips": all(block["trip_count"] == 2 for block in enabled_document["blocks"]),
        "memory_seeds": len(enabled_document["functional_execution"]["memory"]) == 24,
        "mode_only_difference": {
            **enabled_document["functional_execution"],
            "enabled": False,
        }
        == disabled_document["functional_execution"],
        "xfer_wiring": xfer_wiring_check(enabled_document, config),
    }
    counts = schedule_counts(enabled_document)
    expected_pipelines = config["acceptance"]["expected_pipeline_operations"]
    static_checks = {
        "parameters": enabled_document["metadata"]["parameters"]
        == int(config["acceptance"]["expected_parameters"]),
        "scalar_multiplies": counts["scalar_multiplies"]
        == int(config["acceptance"]["expected_scalar_multiplies"]),
        "scalar_adds": counts["scalar_adds"]
        == int(config["acceptance"]["expected_scalar_adds"]),
        "pipelines": counts["pipelines"] == expected_pipelines,
        "operations": counts["operations"]
        == {"fma": 16, "load": 40, "mul": 16, "store": 8, "xfer": 8},
        "functional_operations": counts["functional_operations"]
        == int(config["acceptance"]["expected_functional_operations"]),
        "memory_requests": counts["memory_requests"]
        == int(config["acceptance"]["expected_memory_requests"]),
        "memory_bytes": counts["memory_bytes"]
        == int(config["acceptance"]["expected_memory_bytes"]),
        "events": counts["boundary_events"]
        == int(config["acceptance"]["expected_boundary_events"]),
        "routes": counts["route_hops"]
        == int(config["acceptance"]["expected_route_hops"]),
        "manifest": all(
            item["schedule_counts"] == counts for item in compiler["outputs"].values()
        ),
    }
    run_checks = {
        "experiment": run["experiment_id"] == "H156",
        "target_free": run["paper_performance_targets_consumed"] is False,
        "checks": all(run["checks"].values()),
        "modes": set(run["records"]) == {"enabled", "disabled"},
        "builds": all(
            set(builds) == set(config["acceptance"]["required_builds"])
            for builds in run["records"].values()
        ),
    }
    enabled_builds = run["records"]["enabled"]
    disabled_builds = run["records"]["disabled"]
    execution_checks = {}
    for build in config["acceptance"]["required_builds"]:
        enabled = enabled_builds[build]
        disabled = disabled_builds[build]
        execution_checks[build] = (
            enabled["pass"]
            and disabled["pass"]
            and enabled["returncode"] == disabled["returncode"] == 0
            and enabled["stderr_bytes"] == disabled["stderr_bytes"] == 0
            and enabled["trace_bytes"] > 0
            and enabled["trace_sha256"] == disabled["trace_sha256"]
            and without_functional(enabled["summary"]) == without_functional(disabled["summary"])
        )
    summary = enabled_builds["opt"]["summary"]
    functional = summary["functional"]
    expected_outputs, intermediate_outputs, stage_matrices = numpy_golden(config)
    output_addresses = enabled_document["metadata"]["output_addresses"]
    actual_outputs = [float(functional["memory"][str(address)]) for address in output_addresses]
    errors = [
        abs(actual - expected)
        for actual, expected in zip(actual_outputs, expected_outputs, strict=True)
    ]
    numeric_checks = {
        "enabled": functional["enabled"] is True,
        "outputs": len(errors) == int(config["acceptance"]["expected_outputs"])
        and max(errors) <= float(config["acceptance"]["absolute_error_limit"]),
        "operations": functional["operations"]
        == int(config["acceptance"]["expected_functional_operations"]),
        "finite": functional["nan_values"] == 0 and functional["errors"] == 0,
    }
    registers = functional["registers"]
    transfer_checks = []
    for iteration, intermediate in enumerate(intermediate_outputs):
        for pair, indices in enumerate(contract["stage_pairs"][1]):
            pe = contract["placement"]["stage1"][pair]
            for reg, index in enumerate(indices):
                actual = register_value(
                    registers, pe=pe, tag=2, iteration=iteration, reg=reg
                )
                transfer_checks.append(
                    actual is not None
                    and math.isclose(actual, intermediate[index], rel_tol=0.0, abs_tol=1e-12)
                )
    expected_addresses = {
        output_address(batch, index)
        for batch in range(int(contract["batch"]))
        for index in range(int(contract["width"]))
    }
    transfer_store_checks = {
        "register_values": len(transfer_checks) == 8 and all(transfer_checks),
        "events": summary["boundary_events_emitted"]
        == int(config["acceptance"]["expected_boundary_events"]),
        "routes": summary["route_hops"]
        == summary["unit_hops"]
        == int(config["acceptance"]["expected_route_hops"])
        and summary["skip_hops"] == 0,
        "stores": set(output_addresses) == expected_addresses
        and expected_addresses <= {int(address) for address in functional["memory"]},
    }
    timing_checks = {
        "cycles": summary["cycles"] == disabled_builds["opt"]["summary"]["cycles"],
        "instructions": summary["instructions_issued"]
        == summary["instructions_completed"]
        == int(config["acceptance"]["expected_functional_operations"]),
        "pipelines": summary["issued_by_pipeline"] == expected_pipelines,
        "events": transfer_store_checks["events"],
        "routes": transfer_store_checks["routes"],
    }
    bsmm_rows = [
        row
        for row in parents["core_full_array"]["rows"]
        if row["family"] in {"structured_qkv", "structured_ffn1"}
    ]
    performance_context = {
        "qualified_rows": len(bsmm_rows),
        "all_same_work": all(row["same_work"] for row in bsmm_rows),
        "all_clear_gain": all(row["clear_gain"] for row in bsmm_rows),
        "minimum_speedup": min(row["speedup"] for row in bsmm_rows),
        "maximum_speedup": max(row["speedup"] for row in bsmm_rows),
    }
    parent_semantic_checks = {
        "h155_infrastructure": parents["functional_payload"]["summary"][
            "integrated_scalar_functional_execution_complete"
        ]
        is True,
        "h153_bsmm_rows": performance_context["qualified_rows"] == 4
        and performance_context["all_same_work"]
        and performance_context["all_clear_gain"]
        and performance_context["minimum_speedup"] >= 1.2,
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path) for name, path in config["source_layout"].items()
    }
    source_text = "\n".join(
        (PROJECT_ROOT / path).read_text(errors="replace")
        for path in config["source_layout"].values()
    )
    forbidden = (
        "fig21-target" + "s-run094.json",
        "target" + "_factor",
        "paper_speedup" + "_fit",
    )
    target_free_check = config["acceptance"]["paper_targets_consumed"] is False and not any(
        token in source_text for token in forbidden
    )
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        all(compile_checks.values()) and all(contract_checks.values()),
        all(static_checks.values()),
        all(run_checks.values()) and all(execution_checks.values()),
        numeric_checks["outputs"],
        numeric_checks["enabled"] and numeric_checks["operations"] and numeric_checks["finite"],
        all(transfer_store_checks.values()),
        all(timing_checks.values()),
        all(parent_semantic_checks.values()),
        target_free_check and all(item["pass"] for item in source_files.values()),
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "compile": all(compile_checks.values()),
        "contract": all(contract_checks.values()),
        "static": all(static_checks.values()),
        "runs": all(run_checks.values()) and all(execution_checks.values()),
        "numeric_evaluated": len(errors) == 8,
        "transfer_evaluated": len(transfer_checks) == 8,
        "timing_evaluated": all(timing_checks.values()),
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
        "paper_reproduction_claim": "bsmm_functional_plus_existing_core_gain_only",
        "functional_claim": "same_input_hierarchical_bsmm_on_timed_spatial_schedule",
        "frozen_inputs": frozen,
        "generated_inputs": generated_inputs,
        "parent_checks": parent_checks,
        "compile_checks": compile_checks,
        "contract_checks": contract_checks,
        "static_checks": static_checks,
        "run_checks": run_checks,
        "execution_checks": execution_checks,
        "numeric_checks": numeric_checks,
        "transfer_store_checks": transfer_store_checks,
        "timing_checks": timing_checks,
        "parent_semantic_checks": parent_semantic_checks,
        "stage_matrices": stage_matrices,
        "expected_outputs": expected_outputs,
        "actual_outputs": actual_outputs,
        "absolute_errors": errors,
        "performance_context": performance_context,
        "source_files": source_files,
        "target_free_check": target_free_check,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "builds": len(enabled_builds),
            "batch": int(contract["batch"]),
            "outputs": len(actual_outputs),
            "functional_operations": functional["operations"],
            "maximum_absolute_error": max(errors),
            "cycles": summary["cycles"],
            "boundary_events": summary["boundary_events_emitted"],
            "route_hops": summary["route_hops"],
            "enabled_disabled_timing_identical": all(execution_checks.values()),
            "bsmm_functional_complete": supported,
            "completed_operator_payloads": ["bsmm"] if supported else [],
            "operator_payload_coverage": 1 if supported else 0,
            "required_operator_payloads": 6,
            "existing_bsmm_speedup_minimum": performance_context["minimum_speedup"],
            "existing_bsmm_speedup_maximum": performance_context["maximum_speedup"],
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
            "functional_claim",
            "static_checks",
            "numeric_checks",
            "transfer_store_checks",
            "timing_checks",
            "actual_outputs",
            "performance_context",
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
