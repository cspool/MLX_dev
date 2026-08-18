#!/usr/bin/env python3
"""Audit H160 same-input residual-scale-SiLU execution."""

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
from scripts.compile_elementwise_functional import output_address
from scripts.compile_fft_cmp_functional import schedule_counts

DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/elementwise_functional_v1.yaml"


def without_functional(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "functional"}


def numpy_reference(config: dict[str, Any], parent: dict[str, Any]) -> dict[str, Any]:
    contract = config["operator_contract"]
    inputs = np.asarray(parent["actual_outputs"], dtype=np.float64).reshape(4, 2)
    residual = np.asarray(contract["residual"], dtype=np.float64)
    scales = np.asarray(contract["channel_scale"], dtype=np.float64)
    preactivation = (inputs + residual) * scales
    sigmoid = 1.0 / (1.0 + np.exp(-preactivation))
    output = preactivation * sigmoid
    return {
        "input": inputs.tolist(),
        "preactivation": preactivation.tolist(),
        "sigmoid": sigmoid.tolist(),
        "output": output.tolist(),
    }


def register_value(
    registers: list[dict[str, Any]], *, pe: list[int], tag: int, reg: int
) -> float | None:
    matches = [
        float(item["value"])
        for item in registers
        if item["pe"] == pe
        and int(item["tag"]) == tag
        and int(item["iteration"]) == 0
        and int(item["reg"]) == reg
    ]
    return matches[0] if len(matches) == 1 else None


def xfer_wiring_check(document: dict[str, Any], config: dict[str, Any]) -> bool:
    placement = config["operator_contract"]["placement"]
    preprocess = [
        block for block in document["blocks"] if block["stage"] == "preprocess"
    ]
    activation = [
        block for block in document["blocks"] if block["stage"] == "activation"
    ]
    if len(preprocess) != 8 or len(activation) != 8:
        return False
    events = set()
    for block in preprocess:
        index = int(block["index"])
        xfers = [item for item in block["instructions"] if item["pipeline"] == "xfer"]
        if len(xfers) != 1:
            return False
        item = xfers[0]
        if not (
            item["destination"] == placement["activation"][index]
            and int(item["destination_tag"]) == 2
            and int(item["destination_register"]) == 0
        ):
            return False
        events.add(item["emit_event"])
    return events == {event for block in activation for event in block["wait_events"]}


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
    compile_path = output_root / "elementwise-functional-compile-manifest.json"
    run_path = output_root / "elementwise-functional-run-manifest.json"
    compiler = json.loads(compile_path.read_text())
    run = json.loads(run_path.read_text())
    generated_inputs = {
        "compile_manifest": qualify(compile_path),
        "run_manifest": qualify(run_path),
    }
    compile_checks = {
        "experiment": compiler["experiment_id"] == "H160",
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
        "family": enabled_document["metadata"]["operator_family"] == "elementwise",
        "shape": contract["shape"] == [4, 2],
        "input_parent": enabled_document["metadata"]["input_parent_path"]
        == config["frozen_inputs"]["swa_functional"]["path"]
        and enabled_document["metadata"]["input_values"]
        == parents["swa_functional"]["actual_outputs"],
        "blocks": len(enabled_document["blocks"]) == int(config["acceptance"]["expected_blocks"]),
        "tags": [block["tag"] for block in enabled_document["blocks"]]
        == [1] * 8 + [2] * 8,
        "pes": len({tuple(block["pe"]) for block in enabled_document["blocks"]})
        == int(config["acceptance"]["expected_pes"]),
        "trips": all(block["trip_count"] == 1 for block in enabled_document["blocks"]),
        "memory_seeds": len(enabled_document["functional_execution"]["memory"]) == 16,
        "constant_seeds": len(enabled_document["functional_execution"]["registers"]) == 8
        and all(float(item["value"]) == 1.0 for item in enabled_document["functional_execution"]["registers"]),
        "mode_only_difference": {
            **enabled_document["functional_execution"],
            "enabled": False,
        }
        == disabled_document["functional_execution"],
        "xfer_wiring": xfer_wiring_check(enabled_document, config),
    }
    counts = schedule_counts(enabled_document)
    expected_pipelines = config["acceptance"]["expected_pipeline_operations"]
    expected_compute = config["acceptance"]["expected_compute_operations"]
    static_checks = {
        "scalar_multiplies": counts["scalar_multiplies"]
        == int(config["acceptance"]["expected_scalar_multiplies"]),
        "scalar_adds": counts["scalar_adds"]
        == int(config["acceptance"]["expected_scalar_adds"]),
        "pipelines": counts["pipelines"] == expected_pipelines,
        "compute_operations": {
            name: counts["operations"][name] for name in ("add", "fdiv", "fexp", "mul")
        }
        == expected_compute,
        "functional_operations": counts["functional_operations"]
        == int(config["acceptance"]["expected_functional_operations"]),
        "memory_requests": counts["memory_requests"]
        == int(config["acceptance"]["expected_memory_requests"]),
        "memory_bytes": counts["memory_bytes"]
        == int(config["acceptance"]["expected_memory_bytes"]),
        "events": counts["boundary_events"]
        == int(config["acceptance"]["expected_boundary_events"]),
        "route_hops": counts["route_hops"]
        == int(config["acceptance"]["expected_route_hops"]),
        "skip_hops": counts["skip_hops"]
        == int(config["acceptance"]["expected_skip_hops"]),
        "unit_hops": counts["unit_hops"]
        == int(config["acceptance"]["expected_unit_hops"]),
        "manifest": all(
            item["schedule_counts"] == counts for item in compiler["outputs"].values()
        ),
    }
    run_checks = {
        "experiment": run["experiment_id"] == "H160",
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
    reference = numpy_reference(config, parents["swa_functional"])
    expected_preactivation = [
        value for row in reference["preactivation"] for value in row
    ]
    expected_outputs = [value for row in reference["output"] for value in row]
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
    preactivation_checks = []
    actual_preactivation = []
    for index, expected in enumerate(expected_preactivation):
        actual = register_value(
            registers, pe=contract["placement"]["activation"][index], tag=2, reg=0
        )
        actual_preactivation.append(actual)
        preactivation_checks.append(
            actual is not None
            and math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
        )
    intermediate_checks = {
        "preactivation": len(preactivation_checks) == 8
        and all(preactivation_checks),
        "sigmoid_range": all(
            0.0 < value < 1.0
            for row in reference["sigmoid"]
            for value in row
        ),
    }
    transfer_route_checks = {
        "events": summary["boundary_events_emitted"]
        == int(config["acceptance"]["expected_boundary_events"]),
        "route_hops": summary["route_hops"]
        == int(config["acceptance"]["expected_route_hops"]),
        "skip_hops": summary["skip_hops"]
        == int(config["acceptance"]["expected_skip_hops"]),
        "unit_hops": summary["unit_hops"]
        == int(config["acceptance"]["expected_unit_hops"]),
        "stores": set(output_addresses) == {output_address(index) for index in range(8)},
    }
    timing_checks = {
        "cycles": summary["cycles"] == disabled_builds["opt"]["summary"]["cycles"],
        "instructions": summary["instructions_issued"]
        == summary["instructions_completed"]
        == int(config["acceptance"]["expected_functional_operations"]),
        "pipelines": summary["issued_by_pipeline"] == expected_pipelines,
        "events": transfer_route_checks["events"],
        "routes": all(
            transfer_route_checks[name]
            for name in ("route_hops", "skip_hops", "unit_hops")
        ),
    }
    elementwise_rows = [
        row
        for row in parents["full_array_performance"]["rows"]
        if row["family"] == "elementwise"
    ]
    performance_context = {
        "qualified_rows": len(elementwise_rows),
        "minimum_speedup": min(float(row["speedup"]) for row in elementwise_rows),
        "maximum_speedup": max(float(row["speedup"]) for row in elementwise_rows),
        "all_same_work": all(row["same_work"] for row in elementwise_rows),
        "all_clear_gain": all(row["clear_gain"] for row in elementwise_rows),
    }
    parent_semantic_checks = {
        "h159_chain": parents["swa_functional"]["summary"][
            "completed_operator_payloads"
        ]
        == ["bsmm", "fft_cmp", "attention", "swa"],
        "h153_elementwise": performance_context["qualified_rows"] == 2
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
        numeric_checks["enabled"]
        and numeric_checks["operations"]
        and numeric_checks["finite"]
        and all(intermediate_checks.values()),
        all(transfer_route_checks.values()),
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
        "intermediates_evaluated": len(preactivation_checks) == 8,
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
        "paper_reproduction_claim": "elementwise_functional_plus_existing_same_work_gain_only",
        "functional_claim": "same_input_residual_scale_silu_on_timed_spatial_schedule",
        "frozen_inputs": frozen,
        "generated_inputs": generated_inputs,
        "parent_checks": parent_checks,
        "compile_checks": compile_checks,
        "contract_checks": contract_checks,
        "static_checks": static_checks,
        "run_checks": run_checks,
        "execution_checks": execution_checks,
        "numeric_checks": numeric_checks,
        "intermediate_checks": intermediate_checks,
        "transfer_route_checks": transfer_route_checks,
        "timing_checks": timing_checks,
        "parent_semantic_checks": parent_semantic_checks,
        "numpy_reference": reference,
        "actual_preactivation": actual_preactivation,
        "expected_outputs": expected_outputs,
        "actual_outputs": actual_outputs,
        "absolute_errors": errors,
        "performance_context": performance_context,
        "source_files": source_files,
        "target_free_check": target_free_check,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "builds": len(enabled_builds),
            "outputs": len(actual_outputs),
            "functional_operations": functional["operations"],
            "maximum_absolute_error": max(errors),
            "cycles": summary["cycles"],
            "boundary_events": summary["boundary_events_emitted"],
            "route_hops": summary["route_hops"],
            "skip_hops": summary["skip_hops"],
            "unit_hops": summary["unit_hops"],
            "enabled_disabled_timing_identical": all(execution_checks.values()),
            "elementwise_functional_complete": supported,
            "completed_operator_payloads": [
                "bsmm",
                "fft_cmp",
                "attention",
                "swa",
                "elementwise",
            ]
            if supported
            else ["bsmm", "fft_cmp", "attention", "swa"],
            "operator_payload_coverage": 5 if supported else 4,
            "required_operator_payloads": 6,
            "existing_elementwise_speedup_minimum": performance_context[
                "minimum_speedup"
            ],
            "existing_elementwise_speedup_maximum": performance_context[
                "maximum_speedup"
            ],
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
            "intermediate_checks",
            "transfer_route_checks",
            "timing_checks",
            "actual_preactivation",
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
