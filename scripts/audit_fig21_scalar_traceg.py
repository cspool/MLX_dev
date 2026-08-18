#!/usr/bin/env python3
"""Audit H147 scalar traceg service models and Figure 21 family estimates."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from mlxsim.repeat_folding import fit_affine, relative_error
from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig21_xavier_scalar_traceg_v1.yaml"


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    parent_names = ("h56_xavier", "h91_contract", "h146_hmma")
    parents = {
        name: json.loads((PROJECT_ROOT / config["frozen_inputs"][name]["path"]).read_text())
        for name in parent_names
    }
    parent_checks = {
        name: parent["hypothesis_status"] == config["frozen_inputs"][name]["required_status"]
        and parent["audit_integrity"] is config["frozen_inputs"][name]["required_integrity"]
        for name, parent in parents.items()
    }
    opcode_text = (PROJECT_ROOT / config["frozen_inputs"]["volta_opcode"]["path"]).read_text()
    opcode_checks = {
        "sp": all(
            f'{{"{opcode}", OpcodeChar(OP_{opcode}, SP_OP)}}' in opcode_text
            for opcode in ("FADD", "FMUL", "FMNMX")
        ),
        "sfu": '{"MUFU", OpcodeChar(OP_MUFU, SFU_OP)}' in opcode_text,
        "alu": '{"SHFL", OpcodeChar(OP_SHFL, ALU_OP)}' in opcode_text,
    }
    output_root = PROJECT_ROOT / config["output_root"]
    compile_manifest_path = output_root / "scalar-traceg-compile-manifest.json"
    run_manifest_path = output_root / "scalar-traceg-run-manifest.json"
    compile_manifest = json.loads(compile_manifest_path.read_text())
    run_manifest = json.loads(run_manifest_path.read_text())
    generated_inputs = {
        "compile_manifest": qualify(compile_manifest_path),
        "run_manifest": qualify(run_manifest_path),
    }
    services = config["trace_contract"]["service_classes"]
    repeats_values = [
        *config["trace_contract"]["fit_repeats"],
        *config["trace_contract"]["holdout_repeats"],
    ]
    expected_keys = {f"{service}-r{repeat}" for service in services for repeat in repeats_values}
    compile_checks = {
        "experiment": compile_manifest["experiment_id"] == "H147",
        "target_free": compile_manifest["paper_performance_targets_consumed"] is False,
        "identity": compile_manifest["trace_identity"]
        == "source_derived_compute_only_scalar_service_microtrace",
        "keys": set(compile_manifest["outputs"]) == expected_keys,
        "passing": all(item["pass"] for item in compile_manifest["outputs"].values()),
    }
    ctas = int(config["trace_contract"]["grid"][0])
    scalar_lanes = int(config["trace_contract"]["scalar_operations_per_warp_instruction"])
    trace_checks = {}
    for service, spec in services.items():
        for repeat in repeats_values:
            key = f"{service}-r{repeat}"
            record = compile_manifest["outputs"][key]
            text = (PROJECT_ROOT / record["primary_trace"]["path"]).read_text()
            trace_checks[key] = (
                record["service"] == service
                and record["opcode"] == spec["opcode"]
                and record["ctas"] == ctas
                and record["warp_instructions"] == ctas * repeat
                and record["scalar_operations"] == ctas * repeat * scalar_lanes
                and text.count(f" {spec['opcode']} ") == ctas * repeat
                and all(token not in text for token in (" LD", " ST", "ATOM", "RED"))
                and record["primary_trace"]["sha256"] == record["replay_trace"]["sha256"]
            )
    run_checks = {
        "experiment": run_manifest["experiment_id"] == "H147",
        "target_free": run_manifest["paper_performance_targets_consumed"] is False,
        "identity": run_manifest["trace_identity"]
        == "source_derived_compute_only_scalar_service_microtrace",
        "keys": set(run_manifest["records"]) == expected_keys,
        "passing": all(item["pass"] for item in run_manifest["records"].values()),
    }
    replay_checks = {}
    for service in services:
        for repeat in repeats_values:
            key = f"{service}-r{repeat}"
            item = run_manifest["records"][key]
            measurement = json.loads((PROJECT_ROOT / item["path"]).read_text())
            replay_checks[key] = (
                item["cycles"] > 0
                and item["instructions"] == ctas * 32 * (repeat + 2)
                and item["ctas"] == ctas
                and measurement["scalar_operations"] == ctas * repeat * scalar_lanes
                and all(measurement["checks"].values())
            )
    service_models = {}
    holdouts = []
    fit_repeats = config["trace_contract"]["fit_repeats"]
    for service in services:
        fit_points = [
            (
                float(compile_manifest["outputs"][f"{service}-r{repeat}"]["scalar_operations"]),
                float(run_manifest["records"][f"{service}-r{repeat}"]["cycles"]),
            )
            for repeat in fit_repeats
        ]
        model = fit_affine(fit_points[0][0], fit_points[0][1], fit_points[1][0], fit_points[1][1])
        service_models[service] = {
            "intercept": model.intercept,
            "slope_cycles_per_scalar_operation": model.slope,
            "fit_points": fit_points,
        }
        for repeat in config["trace_contract"]["holdout_repeats"]:
            work = float(compile_manifest["outputs"][f"{service}-r{repeat}"]["scalar_operations"])
            actual = float(run_manifest["records"][f"{service}-r{repeat}"]["cycles"])
            prediction = model.predict(work)
            error = relative_error(prediction, actual)
            holdouts.append(
                {
                    "service": service,
                    "repeat": repeat,
                    "scalar_operations": work,
                    "actual_cycles": actual,
                    "predicted_cycles": prediction,
                    "relative_error": error,
                    "pass": error <= float(config["acceptance"]["holdout_relative_error_limit"]),
                }
            )
    model_checks = {
        service: item["slope_cycles_per_scalar_operation"] > 0
        and item["intercept"] + item["slope_cycles_per_scalar_operation"] > 0
        for service, item in service_models.items()
    }
    holdout_passes = sum(item["pass"] for item in holdouts)
    holdout_gate = holdout_passes == int(config["acceptance"]["required_holdouts"])

    def scalar_cycles(service: str, operations: int) -> float:
        model = service_models[service]
        return (
            float(model["intercept"])
            + float(model["slope_cycles_per_scalar_operation"]) * operations
        )

    h146_model = parents["h146_hmma"]["cycle_model"]

    def tensor_cycles(fma: int) -> float:
        return float(h146_model["intercept"]) + float(h146_model["slope_cycles_per_fma"]) * fma

    family_estimates = {}
    composition_checks = {}
    if holdout_gate and all(model_checks.values()):
        contracts = parents["h91_contract"]["contracts"]
        layers = int(config["composition_contract"]["dense_layers"])
        for sequence in config["composition_contract"]["sequence_lengths"]:
            contract = contracts[f"N{sequence}"]
            attention_counts = contract["dense_components"]["attention"]["fu_instruction_instances"]
            elementwise_counts = contract["elementwise"]["fu_instruction_instances"]
            attention_mapped = {
                service: layers * sum(int(attention_counts[operation]) for operation in operations)
                for service, operations in config["composition_contract"][
                    "attention_mapping"
                ].items()
            }
            elementwise_mapped = {
                service: layers
                * sum(int(elementwise_counts[operation]) for operation in operations)
                for service, operations in config["composition_contract"][
                    "elementwise_mapping"
                ].items()
            }
            attention_components = {
                "tensor": tensor_cycles(attention_mapped["tensor"]),
                "sp": scalar_cycles("sp", attention_mapped["sp"]),
                "sfu": scalar_cycles("sfu", attention_mapped["sfu"]),
            }
            elementwise_components = {
                service: scalar_cycles(service, operations)
                for service, operations in elementwise_mapped.items()
            }
            attention_cycles = sum(attention_components.values())
            elementwise_cycles = sum(elementwise_components.values())
            family_estimates[f"N{sequence}"] = {
                "dense_layers": layers,
                "attention_operation_counts": attention_mapped,
                "elementwise_operation_counts": elementwise_mapped,
                "attention_cycle_components": attention_components,
                "elementwise_cycle_components": elementwise_components,
                "dense_attention_cycles": attention_cycles,
                "dense_attention_seconds": attention_cycles
                / float(config["xavier_replay"]["clock_hz"]),
                "elementwise_cycles": elementwise_cycles,
                "elementwise_seconds": elementwise_cycles
                / float(config["xavier_replay"]["clock_hz"]),
                "mapping_claim": "source_derived_compute_only_service_traceg_proxy",
            }
            composition_checks[f"N{sequence}"] = (
                sum(attention_mapped.values())
                == layers * sum(int(value) for value in attention_counts.values())
                and sum(elementwise_mapped.values())
                == layers * sum(int(value) for value in elementwise_counts.values())
                and all(
                    math.isfinite(value) and value > 0
                    for value in (
                        attention_cycles,
                        elementwise_cycles,
                        family_estimates[f"N{sequence}"]["dense_attention_seconds"],
                        family_estimates[f"N{sequence}"]["elementwise_seconds"],
                    )
                )
            )
    source_files = {
        name: qualify(PROJECT_ROOT / path) for name, path in config["source_layout"].items()
    }
    source_text = "\n".join(
        (PROJECT_ROOT / path).read_text(errors="replace")
        for path in config["source_layout"].values()
    )
    forbidden = (
        "fig21-target" + "s-run094.json",
        "target" + "_efficiency_factor",
        "post_result" + "_service_selection",
    )
    target_free_check = (
        compile_manifest["paper_performance_targets_consumed"] is False
        and run_manifest["paper_performance_targets_consumed"] is False
        and config["acceptance"]["targets_consumed"] is False
        and not any(token in source_text for token in forbidden)
    )
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        all(opcode_checks.values()),
        all(compile_checks.values()) and all(trace_checks.values()),
        all(run_checks.values()) and all(replay_checks.values()),
        all(replay_checks.values()),
        all(model_checks.values()),
        holdout_gate,
        len(family_estimates) == int(config["acceptance"]["required_shapes"])
        and all(composition_checks.values()),
        target_free_check and all(item["pass"] for item in source_files.values()),
        all(
            item["mapping_claim"] == "source_derived_compute_only_service_traceg_proxy"
            for item in family_estimates.values()
        ),
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "opcodes": all(opcode_checks.values()),
        "compile": all(compile_checks.values()),
        "traces": all(trace_checks.values()),
        "runs": all(run_checks.values()) and all(replay_checks.values()),
        "models_evaluated": len(service_models) == 3 and len(holdouts) == 6,
        "composition_evaluated": len(composition_checks)
        in {0, int(config["acceptance"]["required_shapes"])},
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
        "paper_reproduction_claim": "none_target_free_synthetic_scalar_services",
        "trace_identity_claim": "source_derived_compute_only_not_captured_sass",
        "frozen_inputs": frozen,
        "generated_inputs": generated_inputs,
        "parent_checks": parent_checks,
        "opcode_checks": opcode_checks,
        "compile_checks": compile_checks,
        "trace_checks": trace_checks,
        "run_checks": run_checks,
        "replay_checks": replay_checks,
        "service_models": service_models,
        "holdouts": holdouts,
        "family_estimates": family_estimates,
        "composition_checks": composition_checks,
        "source_files": source_files,
        "target_free_check": target_free_check,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "generated_traces": len(compile_manifest["outputs"]),
            "successful_replays": sum(item["pass"] for item in run_manifest["records"].values()),
            "cycles_by_service_repeat": {
                key: item["cycles"] for key, item in run_manifest["records"].items()
            },
            "service_models": len(service_models),
            "holdout_passes": holdout_passes,
            "holdout_total": len(holdouts),
            "holdout_mape": sum(item["relative_error"] for item in holdouts) / len(holdouts),
            "holdout_max_error": max(item["relative_error"] for item in holdouts),
            "dense_attention_estimates": len(family_estimates),
            "elementwise_estimates": len(family_estimates),
            "figure21_dense_projection_complete": True,
            "figure21_dense_attention_complete": supported,
            "figure21_elementwise_complete": supported,
            "active_simulator_figures_reproduced": 3,
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
            "service_models",
            "holdouts",
            "family_estimates",
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
