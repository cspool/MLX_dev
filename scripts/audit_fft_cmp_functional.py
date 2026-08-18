#!/usr/bin/env python3
"""Audit H157 same-input FFT-CMP functional execution."""

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
from scripts.compile_fft_cmp_functional import output_address, schedule_counts

DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fft_cmp_functional_v1.yaml"


def without_functional(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "functional"}


def numpy_reference(
    config: dict[str, Any],
) -> tuple[list[float], list[list[complex]], list[list[complex]]]:
    contract = config["operator_contract"]
    input_length = int(contract["chunk_length"])
    output_length = int(contract["compressed_length"])
    expected: list[float] = []
    retained: list[list[complex]] = []
    full_spectra: list[list[complex]] = []
    for values in contract["inputs"]:
        vector = np.asarray(values, dtype=np.float64)
        full = np.fft.fft(vector)
        spectrum = np.fft.rfft(vector)
        resized = spectrum[: output_length // 2 + 1].copy()
        if output_length % 2 == 0:
            resized[-1] *= 2.0
        output = np.fft.irfft(resized, n=output_length)
        output *= output_length / input_length
        expected.extend(output.tolist())
        retained.append([complex(full[index]) for index in contract["retained_frequency_bins"]])
        full_spectra.append([complex(value) for value in full])
    return expected, retained, full_spectra


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
    placement = config["operator_contract"]["placement"]
    stage0 = [block for block in document["blocks"] if block["stage"] == 0]
    stage1 = [block for block in document["blocks"] if block["stage"] == 1]
    final = [block for block in document["blocks"] if block["stage"] == 2]
    if len(stage0) != 2 or len(stage1) != 2 or len(final) != 1:
        return False
    stage0_events = set()
    for block in stage0:
        pair = int(block["pair_id"])
        xfers = [item for item in block["instructions"] if item["pipeline"] == "xfer"]
        if len(xfers) != 4:
            return False
        for item in xfers:
            identifier = item["id"]
            is_sum = "_sum_" in identifier
            component = 1 if identifier.endswith("_i") else 0
            destination_pair = 0 if is_sum else 1
            expected_register = pair * 2 + component
            if not (
                item["destination"] == placement["fft_stage1"][destination_pair]
                and int(item["destination_tag"]) == 2
                and int(item["destination_register"]) == expected_register
            ):
                return False
            stage0_events.add(item["emit_event"])
    stage1_waits = {event for block in stage1 for event in block["wait_events"]}
    stage1_events = set()
    for block in stage1:
        xfers = [item for item in block["instructions"] if item["pipeline"] == "xfer"]
        if len(xfers) != 2:
            return False
        expected_base = int(block["pair_id"]) * 2
        for component, item in enumerate(xfers):
            if not (
                item["destination"] == placement["compressed_irfft"]
                and int(item["destination_tag"]) == 3
                and int(item["destination_register"]) == expected_base + component
            ):
                return False
            stage1_events.add(item["emit_event"])
    return (
        len(stage0_events) == 8
        and stage0_events == stage1_waits
        and len(stage1_events) == 4
        and stage1_events == set(final[0]["wait_events"])
    )


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
    compile_path = output_root / "fft-cmp-functional-compile-manifest.json"
    run_path = output_root / "fft-cmp-functional-run-manifest.json"
    compiler = json.loads(compile_path.read_text())
    run = json.loads(run_path.read_text())
    generated_inputs = {
        "compile_manifest": qualify(compile_path),
        "run_manifest": qualify(run_path),
    }
    compile_checks = {
        "experiment": compiler["experiment_id"] == "H157",
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
        "family": enabled_document["metadata"]["operator_family"] == "fft_cmp",
        "semantic_basis": enabled_document["metadata"]["semantic_basis"]
        == "chunked_real_fft_low_frequency_truncation_short_irfft",
        "shape": int(contract["chunk_length"]) == 4
        and math.isclose(float(contract["compression_ratio"]), 0.5)
        and int(contract["compressed_length"]) == 2
        and int(contract["batch"]) == 2,
        "blocks": len(enabled_document["blocks"]) == int(config["acceptance"]["expected_blocks"]),
        "tags": [block["tag"] for block in enabled_document["blocks"]]
        == [1, 1, 2, 2, 3],
        "pes": {tuple(block["pe"]) for block in enabled_document["blocks"]}
        == {(0, 0), (0, 2), (2, 0), (2, 2), (3, 1)},
        "trips": all(block["trip_count"] == 2 for block in enabled_document["blocks"]),
        "memory_seeds": len(enabled_document["functional_execution"]["memory"]) == 8,
        "constant_seeds": len(enabled_document["functional_execution"]["registers"]) == 10
        and all(
            item["reg"] == 15 and float(item["value"]) == -1.0
            for item in enabled_document["functional_execution"]["registers"]
        ),
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
            name: counts["operations"][name] for name in ("add", "fma", "mul")
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
        "experiment": run["experiment_id"] == "H157",
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
    expected_outputs, retained_bins, full_spectra = numpy_reference(config)
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
    final_pe = contract["placement"]["compressed_irfft"]
    retained_checks = []
    actual_retained = []
    for iteration, bins in enumerate(retained_bins):
        values = []
        for frequency, expected in enumerate(bins):
            real = register_value(
                registers, pe=final_pe, tag=3, iteration=iteration, reg=frequency * 2
            )
            imag = register_value(
                registers,
                pe=final_pe,
                tag=3,
                iteration=iteration,
                reg=frequency * 2 + 1,
            )
            values.append({"real": real, "imag": imag})
            retained_checks.extend(
                (
                    real is not None
                    and math.isclose(real, expected.real, rel_tol=0.0, abs_tol=1e-12),
                    imag is not None
                    and math.isclose(imag, expected.imag, rel_tol=0.0, abs_tol=1e-12),
                )
            )
        actual_retained.append(values)
    transfer_route_checks = {
        "retained_complex_bins": len(retained_checks) == 8 and all(retained_checks),
        "events": summary["boundary_events_emitted"]
        == int(config["acceptance"]["expected_boundary_events"]),
        "route_hops": summary["route_hops"]
        == int(config["acceptance"]["expected_route_hops"]),
        "skip_hops": summary["skip_hops"]
        == int(config["acceptance"]["expected_skip_hops"]),
        "unit_hops": summary["unit_hops"]
        == int(config["acceptance"]["expected_unit_hops"]),
        "stores": set(output_addresses)
        == {
            output_address(batch, index)
            for batch in range(int(contract["batch"]))
            for index in range(int(contract["compressed_length"]))
        },
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
    fft_points = {
        key: value
        for key, value in parents["fft_multiport_performance"]["comparisons"].items()
        if key.startswith("fft-")
    }
    speedups = [float(item["cycle_speedup"]) for item in fft_points.values()]
    performance_context = {
        "qualified_rows": len(fft_points),
        "all_same_work_proxy": all(
            item["end_to_end_non_regression"] and item["overlay_non_regression"]
            for item in fft_points.values()
        ),
        "all_strictly_improved": all(
            item["strict_cycle_improvement"] for item in fft_points.values()
        ),
        "minimum_speedup": min(speedups),
        "maximum_speedup": max(speedups),
    }
    parent_semantic_checks = {
        "h156_chain": parents["bsmm_functional"]["summary"][
            "completed_operator_payloads"
        ]
        == ["bsmm"],
        "h120_fft_rows": performance_context["qualified_rows"] == 8
        and performance_context["all_same_work_proxy"]
        and performance_context["all_strictly_improved"]
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
        and transfer_route_checks["retained_complex_bins"],
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
        "numeric_evaluated": len(errors) == 4,
        "complex_evaluated": len(retained_checks) == 8,
        "timing_evaluated": all(timing_checks.values()),
        "source": target_free_check and all(item["pass"] for item in source_files.values()),
        "acceptance_evaluated": len(acceptance_gates) == 10
        and all(isinstance(value, bool) for value in acceptance_gates),
    }
    integrity = all(integrity_checks.values())
    supported = integrity and all(acceptance_gates)
    serializable_spectra = [
        [{"real": value.real, "imag": value.imag} for value in spectrum]
        for spectrum in full_spectra
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
        "paper_reproduction_claim": "fft_cmp_functional_plus_existing_multiport_gain_only",
        "functional_claim": "same_input_chunked_fft_truncate_short_irfft_on_timed_schedule",
        "semantic_status": contract["semantic_status"],
        "frozen_inputs": frozen,
        "generated_inputs": generated_inputs,
        "parent_checks": parent_checks,
        "compile_checks": compile_checks,
        "contract_checks": contract_checks,
        "static_checks": static_checks,
        "run_checks": run_checks,
        "execution_checks": execution_checks,
        "numeric_checks": numeric_checks,
        "transfer_route_checks": transfer_route_checks,
        "timing_checks": timing_checks,
        "parent_semantic_checks": parent_semantic_checks,
        "numpy_full_spectra": serializable_spectra,
        "actual_retained_bins": actual_retained,
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
            "skip_hops": summary["skip_hops"],
            "unit_hops": summary["unit_hops"],
            "enabled_disabled_timing_identical": all(execution_checks.values()),
            "fft_cmp_functional_complete": supported,
            "completed_operator_payloads": ["bsmm", "fft_cmp"] if supported else ["bsmm"],
            "operator_payload_coverage": 2 if supported else 1,
            "required_operator_payloads": 6,
            "existing_fft_speedup_minimum": performance_context["minimum_speedup"],
            "existing_fft_speedup_maximum": performance_context["maximum_speedup"],
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
            "transfer_route_checks",
            "timing_checks",
            "actual_retained_bins",
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
