#!/usr/bin/env python3
"""Audit H146 source-derived HMMA traceg folding and projection estimates."""

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

DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig21_xavier_hmma_traceg_v1.yaml"


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    parent_names = (
        "h56_xavier",
        "h91_contract",
        "h144_ptx_failure",
        "h145_nvbit_failure",
    )
    parents = {
        name: json.loads((PROJECT_ROOT / config["frozen_inputs"][name]["path"]).read_text())
        for name in parent_names
    }
    parent_checks = {
        name: parent["hypothesis_status"] == config["frozen_inputs"][name]["required_status"]
        and parent["audit_integrity"] is config["frozen_inputs"][name]["required_integrity"]
        for name, parent in parents.items()
    }
    source_semantic_checks = {
        "h144_failure": parents["h144_ptx_failure"]["failure_class"]
        == "gpgpusim_functional_ptx_wmma_post_enqueue_crash",
        "h145_failure": parents["h145_nvbit_failure"]["failure_class"]
        == "nvbit_1_7_3_cuda_error_not_supported_on_driver_595",
        "hmma_mapping": '{"HMMA", OpcodeChar(OP_HMMA, SPECIALIZED_UNIT_3_OP)}'
        in (PROJECT_ROOT / config["frozen_inputs"]["volta_opcode"]["path"]).read_text(),
        "traceg_parser": 'hasEnding(filePath, ".traceg")'
        in (PROJECT_ROOT / config["frozen_inputs"]["trace_parser"]["path"]).read_text(),
        "tensor_unit": "-specialized_unit_3 1,4,8,4,4,TENSOR"
        in (PROJECT_ROOT / config["frozen_inputs"]["trace_config"]["path"]).read_text(),
    }
    output_root = PROJECT_ROOT / config["output_root"]
    compile_manifest_path = output_root / "hmma-traceg-compile-manifest.json"
    run_manifest_path = output_root / "hmma-traceg-run-manifest.json"
    compile_manifest = json.loads(compile_manifest_path.read_text())
    run_manifest = json.loads(run_manifest_path.read_text())
    generated_inputs = {
        "compile_manifest": qualify(compile_manifest_path),
        "run_manifest": qualify(run_manifest_path),
    }
    repeats_values = [
        *config["trace_contract"]["fit_repeats"],
        *config["trace_contract"]["holdout_repeats"],
    ]
    expected_keys = {f"r{repeat}" for repeat in repeats_values}
    compile_checks = {
        "experiment": compile_manifest["experiment_id"] == "H146",
        "target_free": compile_manifest["paper_performance_targets_consumed"] is False,
        "identity": compile_manifest["trace_identity"]
        == "source_derived_compute_only_HMMA_microtrace",
        "keys": set(compile_manifest["outputs"]) == expected_keys,
        "passing": all(item["pass"] for item in compile_manifest["outputs"].values()),
    }
    trace_checks = {}
    ctas = int(config["trace_contract"]["grid"][0])
    fma_per_hmma = int(config["trace_contract"]["fma_equivalents_per_hmma"])
    for repeat in repeats_values:
        key = f"r{repeat}"
        record = compile_manifest["outputs"][key]
        text = (PROJECT_ROOT / record["primary_trace"]["path"]).read_text()
        trace_checks[key] = (
            record["ctas"] == ctas
            and record["hmma_instructions"] == ctas * repeat
            and record["fma_equivalents"] == ctas * repeat * fma_per_hmma
            and text.count("#BEGIN_TB") == ctas
            and text.count(" HMMA ") == ctas * repeat
            and text.count(" MOV ") == ctas
            and text.count(" EXIT ") == ctas
            and all(token not in text for token in (" LD", " ST", "ATOM", "RED"))
            and record["primary_trace"]["sha256"] == record["replay_trace"]["sha256"]
        )
    run_checks = {
        "experiment": run_manifest["experiment_id"] == "H146",
        "target_free": run_manifest["paper_performance_targets_consumed"] is False,
        "identity": run_manifest["trace_identity"] == "source_derived_compute_only_HMMA_microtrace",
        "keys": set(run_manifest["records"]) == expected_keys,
        "passing": all(item["pass"] for item in run_manifest["records"].values()),
    }
    replay_checks = {}
    for repeat in repeats_values:
        key = f"r{repeat}"
        item = run_manifest["records"][key]
        measurement = json.loads((PROJECT_ROOT / item["path"]).read_text())
        replay_checks[key] = (
            item["cycles"] > 0
            and item["instructions"] == ctas * 32 * (repeat + 2)
            and item["ctas"] == ctas
            and measurement["hmma_instructions"] == ctas * repeat
            and measurement["fma_equivalents"] == ctas * repeat * fma_per_hmma
            and all(measurement["checks"].values())
        )
    fit_repeats = config["trace_contract"]["fit_repeats"]
    fit_points = [
        (
            float(compile_manifest["outputs"][f"r{repeat}"]["fma_equivalents"]),
            float(run_manifest["records"][f"r{repeat}"]["cycles"]),
        )
        for repeat in fit_repeats
    ]
    model = fit_affine(fit_points[0][0], fit_points[0][1], fit_points[1][0], fit_points[1][1])
    holdouts = []
    for repeat in config["trace_contract"]["holdout_repeats"]:
        work = float(compile_manifest["outputs"][f"r{repeat}"]["fma_equivalents"])
        actual = float(run_manifest["records"][f"r{repeat}"]["cycles"])
        prediction = model.predict(work)
        error = relative_error(prediction, actual)
        holdouts.append(
            {
                "repeat": repeat,
                "fma_equivalents": work,
                "actual_cycles": actual,
                "predicted_cycles": prediction,
                "relative_error": error,
                "pass": error <= float(config["acceptance"]["holdout_relative_error_limit"]),
            }
        )
    model_check = model.slope > 0 and model.predict(1.0) > 0
    holdout_passes = sum(item["pass"] for item in holdouts)
    holdout_gate = holdout_passes == int(config["acceptance"]["required_holdouts"])
    projection_estimates = {}
    if holdout_gate:
        contracts = parents["h91_contract"]["contracts"]
        for sequence in config["projection_contract"]["sequence_lengths"]:
            dense_components = contracts[f"N{sequence}"]["dense_components"]
            component_fma = {
                component: int(dense_components[component]["fma_equivalents"])
                for component in config["projection_contract"]["components"]
            }
            total_fma = int(config["projection_contract"]["dense_layers"]) * sum(
                component_fma.values()
            )
            cycles = model.predict(float(total_fma))
            projection_estimates[f"N{sequence}"] = {
                "component_fma_equivalents_per_layer": component_fma,
                "dense_layers": int(config["projection_contract"]["dense_layers"]),
                "total_fma_equivalents": total_fma,
                "xavier_cycles": cycles,
                "xavier_seconds": cycles / float(config["xavier_replay"]["clock_hz"]),
                "mapping_claim": "source_derived_compute_only_HMMA_traceg_proxy",
            }
    estimate_checks = {
        key: all(
            math.isfinite(float(item[field])) and float(item[field]) > 0
            for field in ("total_fma_equivalents", "xavier_cycles", "xavier_seconds")
        )
        and len(item["component_fma_equivalents_per_layer"]) == 4
        for key, item in projection_estimates.items()
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
        "target" + "_efficiency_factor",
        "post_result" + "_trace_selection",
    )
    target_free_check = (
        compile_manifest["paper_performance_targets_consumed"] is False
        and run_manifest["paper_performance_targets_consumed"] is False
        and config["acceptance"]["targets_consumed"] is False
        and not any(token in source_text for token in forbidden)
    )
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        all(source_semantic_checks.values()),
        all(compile_checks.values()) and all(trace_checks.values()),
        all(trace_checks.values()),
        all(run_checks.values()) and all(replay_checks.values()),
        model_check,
        holdout_gate,
        len(projection_estimates) == int(config["acceptance"]["required_projection_shapes"])
        and all(estimate_checks.values()),
        target_free_check and all(item["pass"] for item in source_files.values()),
        all(
            item["mapping_claim"] == "source_derived_compute_only_HMMA_traceg_proxy"
            for item in projection_estimates.values()
        ),
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "source_semantics": all(source_semantic_checks.values()),
        "compile": all(compile_checks.values()),
        "traces": all(trace_checks.values()),
        "runs": all(run_checks.values()) and all(replay_checks.values()),
        "model_evaluated": model_check and len(holdouts) == 2,
        "estimates_evaluated": len(estimate_checks)
        in {0, int(config["acceptance"]["required_projection_shapes"])},
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
        "paper_reproduction_claim": "none_target_free_synthetic_hmma_projection",
        "trace_identity_claim": "source_derived_compute_only_not_captured_sass",
        "frozen_inputs": frozen,
        "generated_inputs": generated_inputs,
        "parent_checks": parent_checks,
        "source_semantic_checks": source_semantic_checks,
        "compile_checks": compile_checks,
        "trace_checks": trace_checks,
        "run_checks": run_checks,
        "replay_checks": replay_checks,
        "cycle_model": {
            "intercept": model.intercept,
            "slope_cycles_per_fma": model.slope,
            "fit_points": fit_points,
        },
        "holdouts": holdouts,
        "projection_estimates": projection_estimates,
        "estimate_checks": estimate_checks,
        "source_files": source_files,
        "target_free_check": target_free_check,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "generated_traces": len(compile_manifest["outputs"]),
            "successful_replays": sum(item["pass"] for item in run_manifest["records"].values()),
            "cycles_by_repeat": {
                str(repeat): run_manifest["records"][f"r{repeat}"]["cycles"]
                for repeat in repeats_values
            },
            "holdout_passes": holdout_passes,
            "holdout_total": len(holdouts),
            "holdout_mape": sum(item["relative_error"] for item in holdouts) / len(holdouts),
            "holdout_max_error": max(item["relative_error"] for item in holdouts),
            "projection_estimates": len(projection_estimates),
            "figure21_dense_projection_complete": supported,
            "figure21_dense_attention_complete": False,
            "figure21_elementwise_complete": False,
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
            "cycle_model",
            "holdouts",
            "projection_estimates",
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
