#!/usr/bin/env python3
"""Build the H180 paper-informed bounded Figure 18 performance estimate."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/fig18_bounded_estimate_v1.yaml"


WORKLOAD_FIELD_MAP = {
    "model_or_block_architecture": ("unit", "Figure18+H141"),
    "batch_size": ("batch", "H141"),
    "structured_vs_dense_component_graph": ("structured_component_graph", "H141"),
    "structured_vs_dense_layer_mix": ("structured_vs_dense_layer_mix", "H141"),
    "block_size_B": ("block_size_B", "H141"),
    "fft_chunk_length_L": ("fft_chunk_length_L", "H141"),
    "per_component_compression_ratio": ("per_component_compression", "Figure18+H141"),
    "ffn_dimension": ("ffn_dimension", "H141"),
    "attention_heads_and_layout": ("attention_heads", "H141"),
    "elementwise_components": ("elementwise_components", "H141"),
    "memory_residency_and_boundaries": ("memory_residency", "H175"),
    "launch_and_measurement_interval": ("measurement_interval", "H141"),
}


def _all_close(left: list[Any], right: list[Any]) -> bool:
    return len(left) == len(right) and all(
        math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-12)
        for a, b in zip(left, right, strict=True)
    )


def _inside(value: float, lower: float, upper: float) -> bool:
    return lower <= value <= upper


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    specs = config["frozen_inputs"]
    identity = json.loads((PROJECT_ROOT / specs["identity_gap"]["path"]).read_text())
    scaling = json.loads(
        (PROJECT_ROOT / specs["complete_block_scaling"]["path"]).read_text()
    )
    mechanism = json.loads(
        (PROJECT_ROOT / specs["data_ready_mechanism"]["path"]).read_text()
    )
    legacy_reference = json.loads(
        (PROJECT_ROOT / specs["figure18_reference"]["path"]).read_text()
    )
    target_file = yaml.safe_load(
        (PROJECT_ROOT / specs["figure18_energy_reference"]["path"]).read_text()
    )
    targets = target_file["fig18_prior_accelerators"]

    parent_checks = {
        "identity_gap": identity["hypothesis_status"]
        == specs["identity_gap"]["required_status"]
        and identity["audit_integrity"] is specs["identity_gap"]["required_integrity"],
        "complete_block_scaling": scaling["hypothesis_status"]
        == specs["complete_block_scaling"]["required_status"]
        and scaling["audit_integrity"]
        is specs["complete_block_scaling"]["required_integrity"],
        "data_ready_mechanism": mechanism["hypothesis_status"]
        == specs["data_ready_mechanism"]["required_status"]
        and mechanism["audit_integrity"]
        is specs["data_ready_mechanism"]["required_integrity"],
    }

    missing_workload_fields = [
        name
        for name, item in identity["workload_evidence"].items()
        if item["status"] == "not_reported" and item["figure18_specific_value"] is None
    ]
    missing_provenance_fields = [
        name
        for name, item in identity["provenance_evidence"].items()
        if item["status"] == "not_reported" and item["figure18_specific_value"] is None
    ]
    identity_gap_checks = {
        "workload_fields_12_of_12_missing": len(missing_workload_fields) == 12,
        "provenance_fields_6_of_6_missing": len(missing_provenance_fields) == 6,
        "exact_workload_still_unidentified": identity["exact_workload_identified"] is False,
        "measurement_provenance_still_unidentified": (
            identity["exact_performance_provenance_identified"] is False
        ),
    }

    representative = config["representative_workload"]
    inferred_workload = {
        field: {
            "value": representative[config_key],
            "provenance": provenance,
            "disclosure": "cross_figure_inference_not_author_reported",
        }
        for field, (config_key, provenance) in WORKLOAD_FIELD_MAP.items()
    }
    workload_checks = {
        "exact_missing_field_set": set(inferred_workload)
        == set(identity["workload_evidence"]),
        "all_values_populated": all(
            item["value"] is not None and item["value"] != ""
            for item in inferred_workload.values()
        ),
        "all_inference_labeled": all(
            item["disclosure"] == "cross_figure_inference_not_author_reported"
            for item in inferred_workload.values()
        ),
        "global_disclosure": representative["disclosure"]
        == "cross_figure_inference_not_Figure18_author_manifest",
    }

    hardware = list(targets["hardware"])
    latency = [float(value) for value in targets["latency_speedup"]]
    energy = [float(value) for value in targets["normalized_energy_saving"]]
    affinity = [float(value) for value in targets["algorithm_normalized_speedup"]]
    flop_saving = [
        float(value) for value in targets["algorithm_flop_saving_from_table4"]
    ]
    calculated_affinity = [
        speedup / (saving / float(config["estimate"]["spatten_flop_saving"]))
        for speedup, saving in zip(latency, flop_saving, strict=True)
    ]
    reference_checks = {
        "hardware_matches_legacy": hardware == legacy_reference["hardware"],
        "latency_matches_legacy": _all_close(
            latency, legacy_reference["inputs"]["latency_speedup"]
        ),
        "flop_saving_matches_legacy": _all_close(
            flop_saving,
            legacy_reference["inputs"]["algorithm_flop_saving_from_table4"],
        ),
        "affinity_target_matches_legacy": _all_close(
            affinity, legacy_reference["target"]["algorithm_normalized_speedup"]
        ),
        "legacy_formula_recomputed": _all_close(
            calculated_affinity,
            legacy_reference["actual"]["algorithm_normalized_speedup"],
        ),
        "seven_complete_rows": all(
            len(values) == 7 for values in (hardware, latency, energy, affinity, flop_saving)
        ),
        "reported_values_positive": all(
            math.isfinite(value) and value > 0
            for values in (latency, energy, affinity, flop_saving)
            for value in values
        ),
    }

    lower_affinity = float(mechanism["summary"]["complete_block_speedup"])
    upper_components = [
        float(scaling["speedups"]["N1024-w2"]["simd32_4x4"]),
        float(scaling["speedups"]["N1024-w4"]["simd32_4x4"]),
    ]
    upper_affinity = sum(upper_components) / len(upper_components)
    point_affinity = (lower_affinity + upper_affinity) / 2.0
    bound_checks = {
        "finite_positive": all(
            math.isfinite(value) and value > 0
            for value in (lower_affinity, upper_affinity, point_affinity)
        ),
        "ordered": lower_affinity < point_affinity < upper_affinity,
        "lower_source": config["estimate"]["lower_affinity_source"]
        == "H172_complete_block_data_ready_speedup",
        "upper_source": config["estimate"]["upper_affinity_source"]
        == "mean_H141_N1024_SIMD32_speedup_windows2_4",
        "point_rule": config["estimate"]["point_affinity"] == "arithmetic_midpoint",
    }

    external_rows = [
        {
            "hardware": hardware[index],
            "row_role": "reported_external_reference_only",
            "reported_latency_speedup": latency[index],
            "reported_normalized_energy_saving": energy[index],
            "reported_affinity": affinity[index],
            "reported_flop_saving": flop_saving[index],
            "estimated_latency_speedup": None,
            "estimated_normalized_energy_saving": None,
        }
        for index in range(5)
    ]
    settings = [float(value) for value in config["estimate"]["settings"]]
    configured_savings = [float(value) for value in config["estimate"]["mlx_flop_savings"]]
    maximum_error = float(config["acceptance"]["maximum_point_latency_relative_error"])
    minimum_clear_speedup = float(config["acceptance"]["minimum_clear_speedup"])
    mlx_rows = []
    for offset, (setting, configured_saving) in enumerate(
        zip(settings, configured_savings, strict=True), start=5
    ):
        reported_saving = flop_saving[offset]
        factor = reported_saving / float(config["estimate"]["spatten_flop_saving"])
        latency_lower = lower_affinity * factor
        latency_upper = upper_affinity * factor
        estimated_latency = point_affinity * factor
        relative_error = abs(estimated_latency - latency[offset]) / latency[offset]
        mlx_rows.append(
            {
                "hardware": hardware[offset],
                "compression_setting": setting,
                "row_role": "paper_informed_bounded_MLX_estimate",
                "reported_latency_speedup": latency[offset],
                "reported_normalized_energy_saving": energy[offset],
                "reported_affinity": affinity[offset],
                "reported_flop_saving": reported_saving,
                "configured_flop_saving": configured_saving,
                "affinity_lower": lower_affinity,
                "affinity_point": point_affinity,
                "affinity_upper": upper_affinity,
                "reported_affinity_inside_bounds": _inside(
                    affinity[offset], lower_affinity, upper_affinity
                ),
                "latency_speedup_lower": latency_lower,
                "estimated_latency_speedup": estimated_latency,
                "latency_speedup_upper": latency_upper,
                "reported_latency_inside_bounds": _inside(
                    latency[offset], latency_lower, latency_upper
                ),
                "point_latency_relative_error": relative_error,
                "point_latency_within_limit": relative_error <= maximum_error,
                "clear_improvement": estimated_latency >= minimum_clear_speedup,
                "estimated_normalized_energy_saving": None,
                "energy_role": config["acceptance"]["energy_series_role"],
            }
        )

    row_checks = {
        "external_rows": len(external_rows)
        == int(config["acceptance"]["required_external_rows"]),
        "mlx_rows": len(mlx_rows) == int(config["acceptance"]["required_mlx_rows"]),
        "settings": settings == [0.75, 0.5],
        "configured_savings_match_reported": all(
            math.isclose(
                row["configured_flop_saving"],
                row["reported_flop_saving"],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for row in mlx_rows
        ),
        "energy_reference_only": all(
            row["estimated_normalized_energy_saving"] is None for row in external_rows
        )
        and all(
            row["estimated_normalized_energy_saving"] is None
            and row["energy_role"] == "reported_reference_only"
            for row in mlx_rows
        ),
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    claim_checks = {
        "validation_disabled": config["validation_eligible"] is False,
        "independent_validation_not_claimed": (
            config["acceptance"]["independent_validation_claimed"] is False
        ),
        "paper_targets_openly_consumed": True,
        "energy_not_estimated": row_checks["energy_reference_only"],
    }

    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        all(identity_gap_checks.values()),
        all(workload_checks.values()),
        all(reference_checks.values()),
        all(bound_checks.values()),
        all(row["reported_affinity_inside_bounds"] for row in mlx_rows),
        all(row["reported_latency_inside_bounds"] for row in mlx_rows),
        all(row["point_latency_within_limit"] for row in mlx_rows),
        all(row["clear_improvement"] for row in mlx_rows) and all(row_checks.values()),
        all(claim_checks.values()) and all(item["pass"] for item in source_files.values()),
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parent_schema": len(parent_checks) == 3,
        "identity_schema": len(identity_gap_checks) == 4,
        "workload_schema": len(inferred_workload) == 12 and len(workload_checks) == 4,
        "reference_schema": len(reference_checks) == 7,
        "bounds_finite": bound_checks["finite_positive"],
        "row_schema": len(external_rows) == 5 and len(mlx_rows) == 2,
        "source": all(item["pass"] for item in source_files.values()),
        "acceptance_evaluated": len(acceptance_gates) == 10
        and all(isinstance(value, bool) for value in acceptance_gates),
    }
    integrity = all(integrity_checks.values())
    supported = integrity and all(acceptance_gates)
    max_point_error = max(row["point_latency_relative_error"] for row in mlx_rows)
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
        "paper_performance_targets_consumed": True,
        "paper_reproduction_claim": (
            "figure18_bounded_paper_informed_exploration_not_independent_reproduction"
        ),
        "independent_validation_claimed": False,
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "identity_gap_checks": identity_gap_checks,
        "missing_workload_fields": missing_workload_fields,
        "missing_provenance_fields": missing_provenance_fields,
        "representative_workload": representative,
        "inferred_workload": inferred_workload,
        "unresolved_measurement_provenance": identity["provenance_evidence"],
        "reference_checks": reference_checks,
        "affinity_envelope": {
            "lower": lower_affinity,
            "upper_components": upper_components,
            "upper": upper_affinity,
            "point": point_affinity,
            "lower_provenance": config["estimate"]["lower_affinity_source"],
            "upper_provenance": config["estimate"]["upper_affinity_source"],
            "point_rule": config["estimate"]["point_affinity"],
        },
        "bound_checks": bound_checks,
        "external_rows": external_rows,
        "mlx_rows": mlx_rows,
        "row_checks": row_checks,
        "claim_checks": claim_checks,
        "source_files": source_files,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "external_reference_rows": len(external_rows),
            "mlx_estimate_rows": len(mlx_rows),
            "identity_workload_fields_missing": len(missing_workload_fields),
            "identity_provenance_fields_missing": len(missing_provenance_fields),
            "affinity_lower": lower_affinity,
            "affinity_point": point_affinity,
            "affinity_upper": upper_affinity,
            "paper_affinity_inside_bounds": sum(
                row["reported_affinity_inside_bounds"] for row in mlx_rows
            ),
            "paper_latency_inside_bounds": sum(
                row["reported_latency_inside_bounds"] for row in mlx_rows
            ),
            "point_latency_passes": sum(
                row["point_latency_within_limit"] for row in mlx_rows
            ),
            "point_latency_max_relative_error": max_point_error,
            "clear_improvement_rows": sum(row["clear_improvement"] for row in mlx_rows),
            "energy_estimated_rows": 0,
            "figure18_exploration_complete": supported,
            "figure18_independently_reproduced": False,
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
            "identity_gap_checks",
            "reference_checks",
            "affinity_envelope",
            "bound_checks",
            "external_rows",
            "mlx_rows",
            "row_checks",
            "claim_checks",
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
