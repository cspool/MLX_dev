#!/usr/bin/env python3
"""Audit H161 one-execution complete Transformer block composition."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scripts.audit_attention_functional import numpy_reference as attention_reference
from scripts.audit_bsmm_functional import numpy_golden as bsmm_reference
from scripts.audit_elementwise_functional import numpy_reference as elementwise_reference
from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify
from scripts.audit_swa_functional import numpy_reference as swa_reference
from scripts.compile_attention_functional import output_address as attention_output_address
from scripts.compile_bsmm_functional import output_address as bsmm_output_address
from scripts.compile_complete_block_functional import link_address_maps
from scripts.compile_elementwise_functional import output_address as elementwise_output_address
from scripts.compile_fft_cmp_functional import output_address as fft_output_address
from scripts.compile_fft_cmp_functional import schedule_counts
from scripts.compile_swa_functional import output_address as swa_output_address

DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/complete_block_functional_v1.yaml"


def without_functional(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "functional"}


def component_configs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        item["name"]: yaml.safe_load((PROJECT_ROOT / item["config"]).read_text())
        for item in config["components"]
    }


def fft_compress(values: list[float], fft_config: dict[str, Any]) -> list[float]:
    contract = fft_config["operator_contract"]
    chunk_length = int(contract["chunk_length"])
    output_length = int(contract["compressed_length"])
    result = []
    for start in range(0, len(values), chunk_length):
        vector = np.asarray(values[start : start + chunk_length], dtype=np.float64)
        spectrum = np.fft.rfft(vector)
        resized = spectrum[: output_length // 2 + 1].copy()
        if output_length % 2 == 0:
            resized[-1] *= 2.0
        compressed = np.fft.irfft(resized, n=output_length)
        compressed *= output_length / chunk_length
        result.extend(compressed.tolist())
    return result


def full_chain_reference(config: dict[str, Any]) -> dict[str, list[float]]:
    configs = component_configs(config)
    bsmm_values, _, _ = bsmm_reference(configs["bsmm"])
    fft_values = fft_compress(bsmm_values, configs["fft_cmp"])
    attention = attention_reference(
        configs["attention"], {"actual_outputs": fft_values}
    )
    attention_values = [value for row in attention["output"] for value in row]
    swa = swa_reference(configs["swa"], {"actual_outputs": attention_values})
    swa_values = [value for row in swa["output"] for value in row]
    elementwise = elementwise_reference(
        configs["elementwise"], {"actual_outputs": swa_values}
    )
    final_values = [value for row in elementwise["output"] for value in row]
    return {
        "bsmm": bsmm_values,
        "fft_cmp": fft_values,
        "attention": attention_values,
        "swa": swa_values,
        "elementwise": final_values,
    }


def boundary_addresses() -> dict[str, list[int]]:
    return {
        "bsmm": [
            bsmm_output_address(batch, index)
            for batch in range(2)
            for index in range(4)
        ],
        "fft_cmp": [
            fft_output_address(batch, index)
            for batch in range(2)
            for index in range(2)
        ],
        "attention": [
            attention_output_address(row, dimension)
            for row in range(2)
            for dimension in range(2)
        ],
        "swa": [
            swa_output_address(row, dimension)
            for row in range(4)
            for dimension in range(2)
        ],
        "elementwise": [elementwise_output_address(index) for index in range(8)],
    }


def sum_component_counts(metadata: list[dict[str, Any]]) -> dict[str, Any]:
    pipelines: Counter[str] = Counter()
    operations: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    scalar_fields = (
        "functional_operations",
        "memory_requests",
        "memory_bytes",
        "boundary_events",
        "route_hops",
        "skip_hops",
        "unit_hops",
        "scalar_multiplies",
        "scalar_adds",
    )
    for component in metadata:
        counts = component["source_schedule_counts"]
        pipelines.update(counts["pipelines"])
        operations.update(counts["operations"])
        for field in scalar_fields:
            totals[field] += int(counts[field])
    return {
        "pipelines": dict(sorted(pipelines.items())),
        "operations": dict(sorted(operations.items())),
        **{field: totals[field] for field in scalar_fields},
    }


def link_checks(document: dict[str, Any]) -> dict[str, bool]:
    replacements = link_address_maps()
    seeds = {
        int(item["address"]) for item in document["functional_execution"]["memory"]
    }
    all_load_addresses = [
        int(address)
        for block in document["blocks"]
        for instruction in block["instructions"]
        if instruction["pipeline"] == "load"
        for address in instruction.get(
            "memory_address_sequence", [instruction["memory_address"]]
        )
    ]
    metadata = document["metadata"]["components"]
    predecessor_checks = []
    for previous, current in pairwise(metadata):
        previous_final = int(previous["tag_range"][1])
        current_first = int(current["tag_range"][0])
        first_blocks = [
            block for block in document["blocks"] if int(block["tag"]) == current_first
        ]
        predecessor_checks.append(
            bool(first_blocks)
            and all(previous_final in block["predecessors"] for block in first_blocks)
        )
    downstream_addresses = {
        address for mapping in replacements.values() for address in mapping
    }
    upstream_addresses = [
        upstream for mapping in replacements.values() for upstream in mapping.values()
    ]
    return {
        "linked_seeds_absent": not (downstream_addresses & seeds),
        "upstream_addresses_loaded": all(
            address in all_load_addresses for address in upstream_addresses
        ),
        "link_cardinality": sum(len(mapping) for mapping in replacements.values())
        == 24,
        "component_link_count": sum(bool(mapping) for mapping in replacements.values())
        == 4,
        "predecessor_chain": len(predecessor_checks) == 4
        and all(predecessor_checks),
    }


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
    compile_path = output_root / "complete-block-functional-compile-manifest.json"
    run_path = output_root / "complete-block-functional-run-manifest.json"
    compiler = json.loads(compile_path.read_text())
    run = json.loads(run_path.read_text())
    generated_inputs = {
        "compile_manifest": qualify(compile_path),
        "run_manifest": qualify(run_path),
    }
    compile_checks = {
        "experiment": compiler["experiment_id"] == "H161",
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
    composition = config["composition_contract"]
    component_metadata = enabled_document["metadata"]["components"]
    contract_checks = {
        "family": enabled_document["metadata"]["operator_family"]
        == "complete_transformer_block",
        "components": [item["name"] for item in component_metadata]
        == composition["order"],
        "component_count": len(component_metadata)
        == int(config["acceptance"]["expected_components"]),
        "dynamic_links": enabled_document["metadata"]["dynamic_link_count"]
        == int(config["acceptance"]["expected_dynamic_links"]),
        "tags": sorted({int(block["tag"]) for block in enabled_document["blocks"]})
        == list(range(1, int(config["acceptance"]["expected_tags"]) + 1)),
        "blocks": len(enabled_document["blocks"])
        == int(config["acceptance"]["expected_blocks"]),
        "pes": len({tuple(block["pe"]) for block in enabled_document["blocks"]})
        == int(config["acceptance"]["expected_pes"]),
        "mode_only_difference": {
            **enabled_document["functional_execution"],
            "enabled": False,
        }
        == disabled_document["functional_execution"],
    }
    dynamic_link_checks = link_checks(enabled_document)
    counts = schedule_counts(enabled_document)
    summed_counts = sum_component_counts(component_metadata)
    expected_pipelines = config["acceptance"]["expected_pipeline_operations"]
    expected_compute = config["acceptance"]["expected_compute_operations"]
    static_checks = {
        "component_sum": counts == summed_counts,
        "scalar_multiplies": counts["scalar_multiplies"]
        == int(config["acceptance"]["expected_scalar_multiplies"]),
        "scalar_adds": counts["scalar_adds"]
        == int(config["acceptance"]["expected_scalar_adds"]),
        "pipelines": counts["pipelines"] == expected_pipelines,
        "compute_operations": {
            name: counts["operations"][name]
            for name in ("add", "fdiv", "fexp", "fma", "fmax", "mul")
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
        "experiment": run["experiment_id"] == "H161",
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
    reference = full_chain_reference(config)
    addresses = boundary_addresses()
    actual_boundaries = {
        name: [float(functional["memory"][str(address)]) for address in values]
        for name, values in addresses.items()
    }
    boundary_errors = {
        name: [
            abs(actual - expected)
            for actual, expected in zip(
                actual_boundaries[name], reference[name], strict=True
            )
        ]
        for name in reference
    }
    boundary_checks = {
        name: len(errors) == len(reference[name])
        and max(errors) <= float(config["acceptance"]["absolute_error_limit"])
        for name, errors in boundary_errors.items()
    }
    expected_outputs = reference["elementwise"]
    actual_outputs = actual_boundaries["elementwise"]
    errors = boundary_errors["elementwise"]
    numeric_checks = {
        "enabled": functional["enabled"] is True,
        "outputs": len(errors) == int(config["acceptance"]["expected_outputs"])
        and max(errors) <= float(config["acceptance"]["absolute_error_limit"]),
        "boundaries": all(boundary_checks.values()),
        "operations": functional["operations"]
        == int(config["acceptance"]["expected_functional_operations"]),
        "finite": functional["nan_values"] == 0 and functional["errors"] == 0,
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
        "mapped_pes": summary["mapped_pe_count"]
        == int(config["acceptance"]["expected_pes"]),
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
    performance = parents["complete_block_performance"]["summary"]
    performance_context = {
        "individual_passes": int(performance["individual_speedup_passes"]),
        "individual_total": int(performance["individual_speedup_total"]),
        "joint_passes": int(performance["joint_speedup_passes"]),
        "joint_total": int(performance["joint_speedup_total"]),
        "minimum_joint_speedup": float(performance["minimum_joint_speedup"]),
        "maximum_joint_speedup": float(performance["maximum_joint_speedup"]),
        "all_work_conserved": performance["all_work_conserved"] is True,
    }
    parent_semantic_checks = {
        "h160_chain": parents["elementwise_functional"]["summary"][
            "completed_operator_payloads"
        ]
        == ["bsmm", "fft_cmp", "attention", "swa", "elementwise"],
        "h141_complete_block": performance_context["individual_passes"]
        == performance_context["individual_total"]
        == 20
        and performance_context["joint_passes"]
        == performance_context["joint_total"]
        == 10
        and performance_context["minimum_joint_speedup"] >= 1.2
        and performance_context["all_work_conserved"],
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
        all(dynamic_link_checks.values()),
        all(static_checks.values()),
        all(run_checks.values()) and all(execution_checks.values()),
        all(numeric_checks.values()),
        all(boundary_checks.values()),
        all(timing_checks.values()) and all(transfer_route_checks.values()),
        all(parent_semantic_checks.values()),
        target_free_check and all(item["pass"] for item in source_files.values()),
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "compile": all(compile_checks.values()),
        "contract": all(contract_checks.values()),
        "links": all(dynamic_link_checks.values()),
        "static": all(static_checks.values()),
        "runs": all(run_checks.values()) and all(execution_checks.values()),
        "numeric_evaluated": len(errors) == 8,
        "boundaries_evaluated": set(boundary_checks)
        == {"bsmm", "fft_cmp", "attention", "swa", "elementwise"},
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
        "paper_reproduction_claim": "complete_block_functional_plus_existing_core_gain_only",
        "functional_claim": "single_execution_dynamic_bsmm_fft_attention_swa_elementwise_chain",
        "frozen_inputs": frozen,
        "generated_inputs": generated_inputs,
        "parent_checks": parent_checks,
        "compile_checks": compile_checks,
        "contract_checks": contract_checks,
        "dynamic_link_checks": dynamic_link_checks,
        "static_checks": static_checks,
        "component_sum_counts": summed_counts,
        "run_checks": run_checks,
        "execution_checks": execution_checks,
        "numeric_checks": numeric_checks,
        "boundary_checks": boundary_checks,
        "transfer_route_checks": transfer_route_checks,
        "timing_checks": timing_checks,
        "parent_semantic_checks": parent_semantic_checks,
        "numpy_boundaries": reference,
        "actual_boundaries": actual_boundaries,
        "boundary_errors": boundary_errors,
        "expected_outputs": expected_outputs,
        "actual_outputs": actual_outputs,
        "absolute_errors": errors,
        "performance_context": performance_context,
        "source_files": source_files,
        "target_free_check": target_free_check,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "builds": len(enabled_builds),
            "components": len(component_metadata),
            "dynamic_links": enabled_document["metadata"]["dynamic_link_count"],
            "tags": len({int(block["tag"]) for block in enabled_document["blocks"]}),
            "blocks": len(enabled_document["blocks"]),
            "mapped_pes": summary["mapped_pe_count"],
            "outputs": len(actual_outputs),
            "functional_operations": functional["operations"],
            "maximum_absolute_error": max(errors),
            "maximum_boundary_absolute_error": max(
                max(values) for values in boundary_errors.values()
            ),
            "cycles": summary["cycles"],
            "boundary_events": summary["boundary_events_emitted"],
            "route_hops": summary["route_hops"],
            "skip_hops": summary["skip_hops"],
            "unit_hops": summary["unit_hops"],
            "enabled_disabled_timing_identical": all(execution_checks.values()),
            "complete_block_functional_complete": supported,
            "completed_operator_payloads": [
                "bsmm",
                "fft_cmp",
                "attention",
                "swa",
                "elementwise",
                "complete_transformer_block",
            ]
            if supported
            else ["bsmm", "fft_cmp", "attention", "swa", "elementwise"],
            "operator_payload_coverage": 6 if supported else 5,
            "required_operator_payloads": 6,
            "existing_joint_block_speedup_minimum": performance_context[
                "minimum_joint_speedup"
            ],
            "existing_joint_block_speedup_maximum": performance_context[
                "maximum_joint_speedup"
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
            "dynamic_link_checks",
            "static_checks",
            "numeric_checks",
            "boundary_checks",
            "transfer_route_checks",
            "timing_checks",
            "actual_boundaries",
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
