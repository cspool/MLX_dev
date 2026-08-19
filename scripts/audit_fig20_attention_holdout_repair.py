#!/usr/bin/env python3
"""Audit H195 Figure20 Attention holdout repair."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from mlxsim.performance_service import CrossFittedLogNContrastService
from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/analysis/fig20_attention_holdout_repair_v1.yaml"
)


def prediction_map(predictions: dict[str, list[dict[str, Any]]]) -> dict[tuple[Any, ...], float]:
    result: dict[tuple[Any, ...], float] = {}
    for item in predictions["figure23"]:
        result[(23, int(item["sequence_length"]), item["series"])] = float(
            item["prediction"]
        )
    for item in predictions["figure19"]:
        result[(19, int(item["sequence_length"]), item["series"])] = float(
            item["prediction"]
        )
    for item in predictions["figure20"]:
        result[
            (
                20,
                int(item["sequence_length"]),
                f"{item['panel']}:{item['operator']}",
            )
        ] = float(item["prediction"])
    return result


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    documents: dict[str, Any] = {}
    for name in (
        "holdout_predictions",
        "holdout_result",
        "endpoint_trace",
        "selected_model",
        "joint_certificate",
    ):
        documents[name] = json.loads(
            (PROJECT_ROOT / config["frozen_inputs"][name]["path"]).read_text()
        )
    parent_checks = {
        name: document["hypothesis_status"] == spec["required_status"]
        and document["audit_integrity"] is spec["required_integrity"]
        for name, document in documents.items()
        for spec in [config["frozen_inputs"][name]]
        if "required_status" in spec
    }

    manifest_path = PROJECT_ROOT / config["prediction_manifest"]
    generated_input = qualify(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    repair = config["repair"]
    sequences = [int(value) for value in repair["trace_sequences"]]
    holdouts = [int(value) for value in repair["holdout_sequences"]]
    base = documents["holdout_predictions"]
    result198 = documents["holdout_result"]
    selected = documents["selected_model"]

    expected_parameter_object = {
        figure: selected[figure]["parameters"]
        for figure in ("figure23", "figure19", "figure20")
    }
    parameter_checks = {
        "object": manifest["parameter_object"] == expected_parameter_object,
        "hash": manifest["parameter_sha256"] == base["parameter_sha256"],
        "attention_parameters": manifest["attention_parameters"]
        == {
            name: float(expected_parameter_object["figure20"][name])
            for name in repair["attention_parameters"]
        },
        "no_refit": manifest["attention_service"]["parameters"]
        == manifest["attention_parameters"],
    }

    components = {str(repair["dense_component"]), str(repair["sparse_component"])}
    expected_records = [
        record
        for document, field in ((documents["endpoint_trace"], "cases"), (base, "trace_cases"))
        for record in document[field]
        if int(record["figure"]) == 20
        and int(record["sequence_length"]) in sequences
        and record["component"] in components
    ]
    trace_checks = {
        "count": len(manifest["trace_records"]) == len(expected_records) == 10,
        "identity": sorted(manifest["trace_records"], key=lambda item: item["key"])
        == sorted(expected_records, key=lambda item: item["key"]),
        "shapes": sorted(int(key) for key in manifest["raw_contrasts"]) == sequences,
        "finite": all(
            math.isfinite(float(value)) for value in manifest["raw_contrasts"].values()
        ),
    }

    contrast_values = {
        int(key): float(value) for key, value in manifest["raw_contrasts"].items()
    }
    service = CrossFittedLogNContrastService(
        values_by_sequence=contrast_values,
        reference_sequence=int(repair["reference_sequence"]),
        model_name="fig20_attention_cross_fitted_log_n_contrast",
        target_informed=False,
        provenance="H182-endpoints+H193-target-free-traces",
    )
    fit_checks: dict[str, bool] = {}
    repair_map = {
        (int(item["sequence_length"]), item["panel"]): item
        for item in manifest["repairs"]
    }
    for sequence in holdouts:
        expected_fit = service.predict_excluding(sequence)
        for panel in repair["panels"]:
            item = repair_map[(sequence, panel)]
            fit_checks[f"N{sequence}:{panel}"] = (
                item["cross_fit"] == expected_fit
                and len(item["cross_fit"]["training_sequences"])
                == int(repair["required_cross_fit_support"])
                and sequence not in item["cross_fit"]["training_sequences"]
                and float(item["cross_fit"]["slope"]) > 0
                and math.isfinite(float(item["new_prediction"]))
                and float(item["new_prediction"]) > 0
            )

    old_predictions = prediction_map(base["predictions"])
    new_predictions = prediction_map(manifest["predictions"])
    changed = [
        key for key in old_predictions if old_predictions[key] != new_predictions[key]
    ]
    changed_expected = {
        (20, sequence, f"{panel}:attention")
        for sequence in holdouts
        for panel in repair["panels"]
    }
    identity_checks = {
        "same_keys": set(new_predictions) == set(old_predictions),
        "changed": set(changed) == changed_expected,
        "changed_count": len(changed) == int(config["acceptance"]["required_replaced_points"]),
        "unchanged_count": len(old_predictions) - len(changed)
        == int(config["acceptance"]["required_unchanged_points"]),
        "unchanged_exact": all(
            new_predictions[key] == old_predictions[key]
            for key in old_predictions
            if key not in changed_expected
        ),
    }

    points: list[dict[str, Any]] = []
    for old_point in result198["points"]:
        key = (
            int(old_point["figure"]),
            int(old_point["sequence_length"]),
            old_point["series"],
        )
        prediction = new_predictions[key]
        reference = float(old_point["reference"])
        direction = old_point["direction_match"]
        if key in changed_expected:
            direction = prediction >= 1.0 and reference >= 1.0
        points.append(
            {
                "figure": key[0],
                "sequence_length": key[1],
                "series": key[2],
                "prediction": prediction,
                "reference": reference,
                "relative_error": abs(prediction - reference) / reference,
                "direction_match": direction,
            }
        )
    limit = float(config["acceptance"]["maximum_relative_error"])
    attention_points = [
        point
        for point in points
        if point["figure"] == 20 and point["series"].endswith(":attention")
    ]
    n4096 = [point for point in attention_points if point["sequence_length"] == 4096]
    directional = [point for point in points if point["direction_match"] is not None]
    numerical_checks = {
        "point_count": len(points) == int(config["acceptance"]["required_total_points"]),
        "attention_count": len(attention_points)
        == int(config["acceptance"]["required_attention_points"]),
        "attention": all(point["relative_error"] <= limit for point in attention_points),
        "n4096_count": len(n4096)
        == int(config["acceptance"]["required_n4096_points"]),
        "n4096": all(point["relative_error"] <= limit for point in n4096),
        "all_points": all(point["relative_error"] <= limit for point in points),
        "directions": len(directional)
        == int(config["acceptance"]["required_direction_matches"])
        and all(point["direction_match"] for point in directional),
        "finite": all(
            math.isfinite(point["prediction"])
            and math.isfinite(point["reference"])
            and math.isfinite(point["relative_error"])
            for point in points
        ),
    }

    runner_text = (PROJECT_ROOT / config["source_layout"]["runner"]).read_text()
    separation_checks = {
        "manifest": manifest["generated_without_evaluation_access"] is True
        and manifest["paper_performance_targets_consumed"] is False,
        "no_auditor_input": all(
            token not in runner_text
            for token in ("holdout_result", "paper_figure", 'frozen["targets"]')
        ),
        "parameters_frozen": all(parameter_checks.values()),
    }
    limitations = config["limitations"]
    limitation_checks = {
        "not_paper_measurement": limitations["n4096_is_paper_measurement"] is False,
        "interpolated_reference": limitations["reference_kind"]
        == "post_prediction_two_endpoint_logN_interpolation",
        "device_mismatch": limitations["trace_device"] != limitations["paper_device"],
        "sparse_proxy_incomplete": set(limitations["sparse_proxy_omits"])
        == {"qk", "softmax", "sv"},
        "not_independent": config["acceptance"]["independent_validation_claimed"]
        is False,
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        all(trace_checks.values()),
        all(fit_checks.values()),
        all(parameter_checks.values()),
        generated_input["pass"] and all(manifest["checks"].values()),
        all(identity_checks.values()),
        numerical_checks["attention"],
        numerical_checks["n4096"],
        numerical_checks["all_points"] and numerical_checks["directions"],
        all(separation_checks.values())
        and all(limitation_checks.values())
        and all(item["pass"] for item in source_files.values()),
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": len(parent_checks) == 4,
        "trace": len(trace_checks) == 4,
        "fits": len(fit_checks) == 6,
        "parameters": len(parameter_checks) == 4,
        "identity": len(identity_checks) == 5,
        "numerical": len(numerical_checks) == 8,
        "separation": len(separation_checks) == 3,
        "limitations": len(limitation_checks) == 5,
        "source": all(item["pass"] for item in source_files.values()),
        "acceptance_evaluated": len(acceptance_gates) == 10
        and all(isinstance(value, bool) for value in acceptance_gates),
    }
    integrity = all(integrity_checks.values())
    supported = integrity and all(acceptance_gates)
    old_n4096 = {
        point["series"]: point["relative_error"]
        for point in result198["failure_points"]
    }
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
            "post_failure_proxy_feature_repair_against_synthetic_interpolation_"
            "not_independent_or_author_hardware_validation"
        ),
        "frozen_inputs": frozen,
        "generated_inputs": {"prediction_manifest": generated_input},
        "parent_checks": parent_checks,
        "trace_checks": trace_checks,
        "fit_checks": fit_checks,
        "parameter_checks": parameter_checks,
        "identity_checks": identity_checks,
        "separation_checks": separation_checks,
        "limitation_checks": limitation_checks,
        "raw_contrasts": manifest["raw_contrasts"],
        "repairs": manifest["repairs"],
        "points": points,
        "attention_points": attention_points,
        "n4096_points": n4096,
        "old_n4096_errors": old_n4096,
        "numerical_checks": numerical_checks,
        "source_files": source_files,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "total_points": len(points),
            "passing_points": sum(point["relative_error"] <= limit for point in points),
            "attention_points": len(attention_points),
            "attention_passing_points": sum(
                point["relative_error"] <= limit for point in attention_points
            ),
            "n4096_points": len(n4096),
            "n4096_passing_points": sum(
                point["relative_error"] <= limit for point in n4096
            ),
            "n4096_max_relative_error": max(point["relative_error"] for point in n4096),
            "max_relative_error": max(point["relative_error"] for point in points),
            "mape": sum(point["relative_error"] for point in points) / len(points),
            "direction_matches": sum(point["direction_match"] is True for point in directional),
            "parameters_refit": False,
            "changed_points": len(changed),
            "unchanged_points": len(points) - len(changed),
            "independent_validation_claimed": False,
            "fig20_attention_holdout_repair_complete": supported,
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
            "parent_checks",
            "trace_checks",
            "fit_checks",
            "parameter_checks",
            "identity_checks",
            "separation_checks",
            "limitation_checks",
            "raw_contrasts",
            "repairs",
            "points",
            "attention_points",
            "n4096_points",
            "old_n4096_errors",
            "numerical_checks",
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
