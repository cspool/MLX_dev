#!/usr/bin/env python3
"""Audit target-free H116 physical-resource counter folding."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from mlxsim.repeat_folding import fit_affine

try:
    from scripts.audit_compute_dma_overlap import git_commit, qualify
except ModuleNotFoundError:
    from audit_compute_dma_overlap import git_commit, qualify

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/simulators/coupled_resource_counter_folding_v1.yaml"
)


def counter_error(prediction: float, actual: float) -> float:
    if actual == 0:
        return 0.0 if prediction == 0 else math.inf
    return abs(prediction - actual) / abs(actual)


def extract_counters(summary: dict[str, Any]) -> dict[str, int]:
    overlay = summary["overlay"]
    pipelines = overlay["productive_pe_cycles_by_pipeline"]
    return {
        "productive_compute_pe_cycles": int(pipelines["compute"]),
        "productive_load_pe_cycles": int(pipelines["load"]),
        "productive_store_pe_cycles": int(pipelines["store"]),
        "productive_xfer_pe_cycles": int(pipelines["xfer"]),
        "productive_fma_pe_cycles": int(
            overlay["productive_pe_cycles_by_fu_class"].get("fma", 0)
        ),
    }


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    h114 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h114"]["path"]).read_text()
    )
    h71 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h71"]["path"]).read_text()
    )
    h61 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h61"]["path"]).read_text()
    )
    parent_checks = {
        "h114": h114["hypothesis_status"] == "supported"
        and h114["audit_integrity"] is True,
        "h114_complete": h114["summary"]["paths"] == 48
        and h114["summary"]["configs"] == 192
        and h114["summary"]["cycle_holdouts_passed"] == 96,
        "h71": h71["hypothesis_status"] == "supported"
        and h71["audit_integrity"] is True,
        "h71_fma_counter": "fma_utilization" in h71,
        "h61": h61["hypothesis_status"] == "rejected"
        and h61["audit_integrity"] is True,
        "h61_semantics": h61["primary_metric"]["numerator"]
        == "productive_pe_cycles_by_pipeline"
        and h61["primary_metric"]["denominator"]
        == "cycles * physical_pe_count",
    }
    run_manifest_path = PROJECT_ROOT / h114["run_manifest"]["path"]
    run_manifest_file = qualify(run_manifest_path, h114["run_manifest"])
    run = json.loads(run_manifest_path.read_text())
    optimized = {
        item["run_key"]: item
        for item in run["records"]
        if item["mode"] == "optimized" and int(item["replay"]) == 1
    }
    record_checks = {
        run_key: qualify(
            PROJECT_ROOT / record["summary_path"],
            {"sha256": record["summary_sha256"]},
        )["pass"]
        and record["summary"]["overlay"]["done"] is True
        and int(record["summary"]["overlay"]["physical_pe_count"])
        == int(config["hardware"]["physical_pes"])
        and int(record["summary"]["end_to_end_cycles"])
        == int(h114["measurements"][run_key]["cycles"])
        for run_key, record in optimized.items()
    }
    metrics = list(config["folding"]["metrics"])
    measurements = {
        run_key: {
            "cycles": int(record["summary"]["end_to_end_cycles"]),
            "counters": extract_counters(record["summary"]),
        }
        for run_key, record in optimized.items()
    }
    counter_checks = {
        run_key: sorted(item["counters"]) == sorted(metrics)
        and all(value >= 0 for value in item["counters"].values())
        for run_key, item in measurements.items()
    }

    fit_scales = [int(value) for value in config["folding"]["fit_scales"]]
    holdout_scales = [
        int(value) for value in config["folding"]["holdout_scales"]
    ]
    limit = float(config["folding"]["relative_error_limit"])
    models: dict[str, dict[str, Any]] = {}
    full_counters: dict[str, dict[str, Any]] = {}
    evaluated_errors = []
    zero_metric_slots = 0
    metric_slots = 0
    eligible_paths = 0
    for path_key, path_model in h114["models"].items():
        models[path_key] = {"family": path_model["family"], "metrics": {}}
        full_cycles = float(h114["full_estimates"][path_key]["cycles"])
        full_scale = int(path_model["full_scale"])
        path_eligible = True
        projected: dict[str, Any] = {}
        for metric in metrics:
            metric_slots += 1
            values = {
                scale: int(
                    measurements[f"{path_key}-q{scale}"]["counters"][metric]
                )
                for scale in [*fit_scales, *holdout_scales]
            }
            if all(value == 0 for value in values.values()):
                zero_metric_slots += 1
                models[path_key]["metrics"][metric] = {
                    "classification": "exact_zero",
                    "values": values,
                    "holdouts": [],
                    "eligible": True,
                }
                projected[metric] = {
                    "counter": 0.0,
                    "utilization": 0.0,
                    "classification": "exact_zero",
                }
                continue
            affine = fit_affine(
                fit_scales[0],
                values[fit_scales[0]],
                fit_scales[1],
                values[fit_scales[1]],
            )
            holdouts = []
            for scale in holdout_scales:
                prediction = affine.predict(scale)
                actual = values[scale]
                error = counter_error(prediction, actual)
                evaluated_errors.append(error)
                holdouts.append(
                    {
                        "scale": scale,
                        "actual": actual,
                        "prediction": prediction,
                        "relative_error": error,
                        "pass_5pct": math.isfinite(error) and error <= limit,
                    }
                )
            eligible = all(item["pass_5pct"] for item in holdouts)
            path_eligible = path_eligible and eligible
            full_counter = affine.predict(full_scale) if eligible else None
            models[path_key]["metrics"][metric] = {
                "classification": "affine_nonzero",
                "intercept": affine.intercept,
                "slope": affine.slope,
                "values": values,
                "holdouts": holdouts,
                "eligible": eligible,
            }
            projected[metric] = {
                "counter": full_counter,
                "utilization": (
                    full_counter
                    / (full_cycles * int(config["hardware"]["physical_pes"]))
                    if full_counter is not None
                    else None
                ),
                "classification": "affine_nonzero",
            }
        models[path_key]["eligible"] = path_eligible
        if path_eligible:
            eligible_paths += 1
        issue_utilization = float(
            h114["full_estimates"][path_key]["fma_issue_utilization"]
        )
        full_counters[path_key] = {
            "family": path_model["family"],
            "cycles": full_cycles,
            "metrics": projected,
            "fma_issue_utilization_completed_work": issue_utilization,
            "fma_residence_utilization": projected[
                "productive_fma_pe_cycles"
            ]["utilization"],
            "counter_semantics": {
                "issue": "completed_scalar_fma_div_cycles_times_16x32",
                "residence": "productive_fma_pe_cycles_div_cycles_times_16",
            },
            "eligible": path_eligible,
        }

    utilization_checks = {}
    for path_key, item in full_counters.items():
        values = [
            metric["utilization"]
            for metric in item["metrics"].values()
            if metric["utilization"] is not None
        ]
        utilization_checks[path_key] = (
            item["eligible"]
            and len(values) == len(metrics)
            and all(math.isfinite(value) and 0 <= value <= 1 for value in values)
            and item["counter_semantics"]["issue"]
            != item["counter_semantics"]["residence"]
        )
    family_counts = Counter(item["family"] for item in models.values())
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    serialized_config = json.dumps(config, sort_keys=True).lower()
    target_free = (
        "paper_targets" not in serialized_config
        and "fig22_resource_utilization" not in serialized_config
        and "fig25_roofline_utilization" not in serialized_config
        and "residual_scale" not in serialized_config
        and "family_correction" not in serialized_config
        and "counter_selection" not in serialized_config
    )
    counts = {
        "paths": len(models) == int(config["required"]["paths"]),
        "configs": len(optimized) == int(config["required"]["configs"]),
        "cycle_holdouts": h114["summary"]["cycle_holdouts_total"]
        == int(config["required"]["cycle_holdouts"]),
        "metric_slots": metric_slots == int(config["required"]["metric_slots"]),
        "families": dict(family_counts) == {"fft": 8, "qkv_bsmm": 24, "swa": 16},
    }
    all_counter_holdouts = all(
        item["eligible"]
        for path in models.values()
        for item in path["metrics"].values()
    )
    acceptance_gates = [
        all(item["pass"] for item in frozen.values())
        and all(parent_checks.values()),
        run_manifest_file["pass"]
        and len(optimized) == int(config["required"]["configs"])
        and all(record_checks.values()),
        all(counter_checks.values()),
        all(counts.values()) and set(metrics) == set(config["folding"]["metrics"]),
        all(
            item["classification"] != "exact_zero"
            or all(value == 0 for value in item["values"].values())
            for path in models.values()
            for item in path["metrics"].values()
        ),
        all_counter_holdouts
        and all(math.isfinite(error) and error <= limit for error in evaluated_errors),
        h114["summary"]["cycle_holdouts_passed"]
        == h114["summary"]["cycle_holdouts_total"]
        and all(item["cycles"] > 0 for item in full_counters.values()),
        all(utilization_checks.values()),
        all(
            item["counter_semantics"]["issue"]
            != item["counter_semantics"]["residence"]
            for item in full_counters.values()
        ),
        eligible_paths == int(config["required"]["paths"]),
        target_free and all(item["pass"] for item in source_files.values()),
        config["classification"] == "target_free_coupled_physical_counter_folding"
        and config["validation_eligible"] is False,
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "run_manifest": run_manifest_file["pass"],
        "records": all(record_checks.values()),
        "counters": all(counter_checks.values()),
        "counts": all(counts.values()),
        "models_evaluated": metric_slots == int(config["required"]["metric_slots"]),
        "source_files": all(item["pass"] for item in source_files.values()),
        "target_free": target_free,
        "acceptance_evaluated": len(acceptance_gates) == 12
        and all(isinstance(value, bool) for value in acceptance_gates),
    }
    integrity = all(integrity_checks.values())
    supported = integrity and all(acceptance_gates)
    finite_errors = [error for error in evaluated_errors if math.isfinite(error)]
    fma_residence = [
        item["fma_residence_utilization"]
        for item in full_counters.values()
        if item["fma_residence_utilization"] is not None
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
        "paper_reproduction_claim": "none_target_free_counter_folding_only",
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "run_manifest": run_manifest_file,
        "record_checks": record_checks,
        "counter_checks": counter_checks,
        "models": models,
        "full_counters": full_counters,
        "utilization_checks": utilization_checks,
        "counts": counts,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "paths": len(models),
            "configs": len(optimized),
            "metric_slots": metric_slots,
            "zero_metric_slots": zero_metric_slots,
            "modeled_metric_slots": metric_slots - zero_metric_slots,
            "counter_holdouts": len(evaluated_errors),
            "counter_holdouts_passed": sum(
                math.isfinite(error) and error <= limit
                for error in evaluated_errors
            ),
            "counter_mape": sum(finite_errors) / len(finite_errors),
            "counter_max_error": max(finite_errors),
            "eligible_full_paths": eligible_paths,
            "fma_residence_utilization_min": min(fma_residence),
            "fma_residence_utilization_max": max(fma_residence),
            "acceptance_gates_passed": sum(acceptance_gates),
            "acceptance_gates_total": len(acceptance_gates),
            "active_simulator_figures_reproduced": 0,
            "active_simulator_figures_total": 8,
        },
        "source_files": source_files,
        "integrity_checks": integrity_checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report = build_audit(config)
    output = PROJECT_ROOT / config["result_path"]
    if args.verify_existing:
        existing = json.loads(output.read_text())
        keys = (
            "hypothesis_status",
            "audit_integrity",
            "models",
            "full_counters",
            "acceptance_gates",
            "summary",
            "integrity_checks",
        )
        matches = all(existing.get(key) == report.get(key) for key in keys)
        print(json.dumps({"existing_matches": matches, **report}, indent=2))
        return 0 if matches else 1
    if args.preflight_only:
        print(json.dumps(report, indent=2))
        return 0 if report["audit_integrity"] and not output.exists() else 1
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"status": report["hypothesis_status"], **report["summary"]},
            indent=2,
        )
    )
    return 0 if report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
