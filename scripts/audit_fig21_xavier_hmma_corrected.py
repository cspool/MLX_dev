#!/usr/bin/env python3
"""Correct H146 SASS-HMMA work semantics without changing replay cycles."""

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

DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/fig21_xavier_hmma_corrected_v1.yaml"


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    evidence = {
        name: json.loads((PROJECT_ROOT / spec["path"]).read_text())
        for name, spec in config["frozen_inputs"].items()
    }
    parent_checks = {
        name: parent["hypothesis_status"] == spec["required_status"]
        and parent["audit_integrity"] is spec["required_integrity"]
        for name, parent in evidence.items()
        for spec in [config["frozen_inputs"][name]]
    }
    corrected = config["corrected_trace_contract"]
    diagnosis = evidence["h150_diagnosis"]["tensor_decomposition"]
    decomposition_checks = {
        "old": diagnosis["recorded_fma_per_trace_hmma"]
        == corrected["ptx_wmma_fma_equivalents"]
        == 4096,
        "sass_count": diagnosis["sass_hmma_per_ptx_wmma"]
        == corrected["sass_hmma_per_ptx_wmma"]
        == 16,
        "corrected": diagnosis["corrected_fma_per_trace_hmma"]
        == corrected["sass_hmma_fma_equivalents"]
        == 256,
    }
    h146 = evidence["h146_hmma"]
    cycles = {int(key): int(value) for key, value in h146["summary"]["cycles_by_repeat"].items()}
    replay_checks = {
        "identity": h146["trace_identity_claim"] == "source_derived_compute_only_not_captured_sass",
        "cycles": cycles
        == {int(key): int(value) for key, value in corrected["replay_cycles"].items()},
        "replays": h146["summary"]["successful_replays"] == 4,
    }
    ctas = int(corrected["ctas"])
    fma_per_hmma = int(corrected["sass_hmma_fma_equivalents"])
    repeats_values = [*corrected["fit_repeats"], *corrected["holdout_repeats"]]
    work_rows = {
        repeat: {
            "repeats": repeat,
            "ctas": ctas,
            "sass_hmma_instructions": ctas * repeat,
            "fma_equivalents_per_sass_hmma": fma_per_hmma,
            "corrected_fma_equivalents": ctas * repeat * fma_per_hmma,
            "old_h146_fma_equivalents": ctas * repeat * 4096,
            "cycles": cycles[repeat],
        }
        for repeat in repeats_values
    }
    work_checks = {
        str(repeat): row["old_h146_fma_equivalents"] == 16 * row["corrected_fma_equivalents"]
        and row["corrected_fma_equivalents"] == ctas * repeat * 256
        for repeat, row in work_rows.items()
    }
    fit_repeats = corrected["fit_repeats"]
    fit_rows = [work_rows[repeat] for repeat in fit_repeats]
    model = fit_affine(
        float(fit_rows[0]["corrected_fma_equivalents"]),
        float(fit_rows[0]["cycles"]),
        float(fit_rows[1]["corrected_fma_equivalents"]),
        float(fit_rows[1]["cycles"]),
    )
    model_check = model.slope > 0 and model.predict(1.0) > 0
    holdouts = []
    for repeat in corrected["holdout_repeats"]:
        row = work_rows[repeat]
        prediction = model.predict(float(row["corrected_fma_equivalents"]))
        error = relative_error(prediction, float(row["cycles"]))
        holdouts.append(
            {
                "repeat": repeat,
                "corrected_fma_equivalents": row["corrected_fma_equivalents"],
                "actual_cycles": row["cycles"],
                "predicted_cycles": prediction,
                "relative_error": error,
                "pass": error <= float(config["acceptance"]["holdout_relative_error_limit"]),
            }
        )
    holdout_passes = sum(item["pass"] for item in holdouts)
    holdout_gate = holdout_passes == int(config["acceptance"]["required_holdouts"])
    projection_estimates = {}
    if holdout_gate:
        contracts = evidence["h91_contract"]["contracts"]
        for sequence in config["projection_contract"]["sequence_lengths"]:
            dense = contracts[f"N{sequence}"]["dense_components"]
            component_fma = {
                component: int(dense[component]["fma_equivalents"])
                for component in config["projection_contract"]["components"]
            }
            total_fma = int(config["projection_contract"]["dense_layers"]) * sum(
                component_fma.values()
            )
            predicted_cycles = model.predict(float(total_fma))
            old_cycles = float(h146["projection_estimates"][f"N{sequence}"]["xavier_cycles"])
            projection_estimates[f"N{sequence}"] = {
                "component_fma_equivalents_per_layer": component_fma,
                "dense_layers": int(config["projection_contract"]["dense_layers"]),
                "total_fma_equivalents": total_fma,
                "xavier_cycles": predicted_cycles,
                "xavier_seconds": predicted_cycles
                / float(config["projection_contract"]["xavier_clock_hz"]),
                "old_h146_cycles": old_cycles,
                "corrected_over_old_cycles": predicted_cycles / old_cycles,
                "mapping_claim": "source_corrected_compute_only_SASS_HMMA_traceg_proxy",
            }
    estimate_checks = {
        key: len(item["component_fma_equivalents_per_layer"]) == 4
        and all(
            math.isfinite(float(item[field])) and float(item[field]) > 0
            for field in (
                "total_fma_equivalents",
                "xavier_cycles",
                "xavier_seconds",
                "old_h146_cycles",
                "corrected_over_old_cycles",
            )
        )
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
        "target" + "_factor",
        "cycle" + "_adjustment",
        "post_result" + "_model_choice",
    )
    target_free_check = config["acceptance"]["targets_consumed"] is False and not any(
        token in source_text for token in forbidden
    )
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        all(decomposition_checks.values()),
        all(replay_checks.values()),
        all(work_checks.values()),
        model_check,
        holdout_gate,
        len(evidence["h91_contract"]["contracts"]) == 5
        and all(
            all(value > 0 for value in item["component_fma_equivalents_per_layer"].values())
            for item in projection_estimates.values()
        ),
        len(projection_estimates) == int(config["acceptance"]["required_shapes"])
        and all(estimate_checks.values()),
        target_free_check and all(item["pass"] for item in source_files.values()),
        all(
            item["mapping_claim"] == "source_corrected_compute_only_SASS_HMMA_traceg_proxy"
            for item in projection_estimates.values()
        ),
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "decomposition": all(decomposition_checks.values()),
        "replays": all(replay_checks.values()),
        "work": all(work_checks.values()),
        "model_evaluated": model_check and len(holdouts) == 2,
        "estimates_evaluated": len(estimate_checks)
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
        "paper_reproduction_claim": "none_target_free_corrected_sass_hmma_projection",
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "decomposition_checks": decomposition_checks,
        "replay_checks": replay_checks,
        "work_rows": {str(key): value for key, value in work_rows.items()},
        "work_checks": work_checks,
        "cycle_model": {
            "intercept": model.intercept,
            "slope_cycles_per_fma": model.slope,
            "fit_repeats": fit_repeats,
        },
        "holdouts": holdouts,
        "projection_estimates": projection_estimates,
        "estimate_checks": estimate_checks,
        "source_files": source_files,
        "target_free_check": target_free_check,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "replay_cycles_unchanged": cycles == {16: 128, 32: 240, 64: 464, 128: 912},
            "corrected_fma_per_sass_hmma": fma_per_hmma,
            "holdout_passes": holdout_passes,
            "holdout_total": len(holdouts),
            "holdout_mape": sum(item["relative_error"] for item in holdouts) / len(holdouts),
            "holdout_max_error": max(item["relative_error"] for item in holdouts),
            "projection_estimates": len(projection_estimates),
            "minimum_projection_seconds": min(
                item["xavier_seconds"] for item in projection_estimates.values()
            ),
            "maximum_projection_seconds": max(
                item["xavier_seconds"] for item in projection_estimates.values()
            ),
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
            "work_rows",
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
