#!/usr/bin/env python3
"""Audit H117 FFT q64/q128 coupled counter steady state."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from mlxsim.coupled_full_mesh_paths import compile_coupled_path
from mlxsim.dsagen_overlay import canonical_json
from mlxsim.repeat_folding import fit_affine
from scripts.audit_coupled_resource_counter_folding import (
    counter_error,
    extract_counters,
)

try:
    from scripts.audit_compute_dma_overlap import git_commit, qualify
except ModuleNotFoundError:
    from audit_compute_dma_overlap import git_commit, qualify

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/simulators/fft_coupled_counter_steady_state_v1.yaml"
)


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    h116 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h116"]["path"]).read_text()
    )
    h114 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h114"]["path"]).read_text()
    )
    h107 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h107"]["path"]).read_text()
    )
    base_config = yaml.safe_load(
        (PROJECT_ROOT / config["frozen_inputs"]["h114_config"]["path"])
        .read_text()
    )
    parent_checks = {
        "h116": h116["hypothesis_status"] == "rejected"
        and h116["audit_integrity"] is True,
        "h116_fft_only": sum(
            not holdout["pass_5pct"]
            for path in h116["models"].values()
            if path["family"] == "fft"
            for metric in path["metrics"].values()
            for holdout in metric["holdouts"]
        )
        == 27
        and all(
            path["eligible"]
            for path in h116["models"].values()
            if path["family"] != "fft"
        ),
        "h114": h114["hypothesis_status"] == "supported"
        and h114["audit_integrity"] is True,
        "h107": h107["hypothesis_status"] == "supported"
        and h107["audit_integrity"] is True,
    }
    h110 = json.loads(
        (PROJECT_ROOT / h114["frozen_inputs"]["h110"]["path"]).read_text()
    )
    h110_compile = json.loads(
        (PROJECT_ROOT / h110["compile_manifest"]["path"]).read_text()
    )
    h114_run_path = PROJECT_ROOT / h114["run_manifest"]["path"]
    h114_run_file = qualify(h114_run_path, h114["run_manifest"])
    h114_run = json.loads(h114_run_path.read_text())
    parent_records = {
        item["run_key"]: item
        for item in h114_run["records"]
        if item["mode"] == "optimized"
        and int(item["replay"]) == 1
        and int(item["run_key"].rsplit("-q", 1)[1]) in config["scales"]["fit"]
        and item["run_key"].startswith("fft_cmp--")
    }

    output_root = PROJECT_ROOT / config["output_root"]
    compile_path = output_root / "fft-coupled-counter-compile-manifest.json"
    run_path = output_root / "fft-coupled-counter-run-manifest.json"
    compile_file = qualify(compile_path)
    run_file = qualify(run_path)
    compiled = json.loads(compile_path.read_text())
    run = json.loads(run_path.read_text())
    child_records = {
        item["run_key"]: item
        for item in run["records"]
        if item["mode"] == "optimized" and int(item["replay"]) == 1
    }
    compile_checks = {}
    execution_checks = {}
    for run_key, item in compiled["outputs"].items():
        path_key, scale_text = run_key.rsplit("-q", 1)
        overlay, memory, metadata, _ = compile_coupled_path(
            run_key=run_key,
            contract=h110_compile["path_contracts"][path_key],
            path=h107["path_results"][path_key],
            scale=int(scale_text),
            config=base_config,
        )
        overlay_path = PROJECT_ROOT / item["overlay"]["path"]
        memory_path = PROJECT_ROOT / item["memory"]["path"]
        compile_checks[run_key] = (
            qualify(overlay_path, item["overlay"])["pass"]
            and qualify(memory_path, item["memory"])["pass"]
            and overlay_path.read_text() == canonical_json(overlay)
            and memory_path.read_text() == canonical_json(memory)
            and item["metadata"] == metadata
            and all(metadata["checks"].values())
        )
        record = child_records[run_key]
        summary = record["summary"]
        execution_checks[run_key] = (
            qualify(
                PROJECT_ROOT / record["summary_path"],
                {"sha256": record["summary_sha256"]},
            )["pass"]
            and record["pass"] is True
            and summary["overlay"]["done"] is True
            and summary["memory"]["idle"] is True
            and summary["overlay"]["instructions_completed"]
            == sum(metadata["pipeline_counts"].values())
            and summary["overlay"]["external_memory_requests"]
            == summary["overlay"]["external_memory_completions"]
            == metadata["memory_requests"]
            and summary["memory"]["responses"] == metadata["memory_requests"]
            and summary["memory"]["offchip_read_bytes"]
            == metadata["scaled_read_bytes"]
            and summary["memory"]["offchip_write_bytes"]
            == metadata["scaled_write_bytes"]
            and summary["memory"]["released_tiles"]
            == summary["memory"]["drained_tiles"]
            == metadata["tile_count"]
            and summary["memory"]["ownership_violations"] == 0
        )

    metrics = list(config["metrics"])
    all_records = {**parent_records, **child_records}
    measurements = {}
    record_checks = {}
    for run_key, record in all_records.items():
        summary = record["summary"]
        counters = extract_counters(summary)
        measurements[run_key] = {
            "cycles": int(summary["end_to_end_cycles"]),
            **counters,
        }
        record_checks[run_key] = (
            qualify(
                PROJECT_ROOT / record["summary_path"],
                {"sha256": record["summary_sha256"]},
            )["pass"]
            and sorted(counters)
            == sorted(metric for metric in metrics if metric != "cycles")
            and all(value >= 0 for value in counters.values())
        )

    fit_scales = [int(value) for value in config["scales"]["fit"]]
    holdout_scales = [int(value) for value in config["scales"]["holdout"]]
    limit = float(config["scales"]["relative_error_limit"])
    path_keys = sorted(
        key for key, item in h107["path_results"].items() if item["family"] == "fft"
    )
    models = {}
    full_estimates = {}
    errors = []
    failure_counts = Counter()
    for path_key in path_keys:
        path_models = {}
        path_eligible = True
        full_scale = int(h114["models"][path_key]["full_scale"])
        for metric in metrics:
            values = {
                scale: int(measurements[f"{path_key}-q{scale}"][metric])
                for scale in [*fit_scales, *holdout_scales]
            }
            if all(value == 0 for value in values.values()):
                path_models[metric] = {
                    "classification": "exact_zero",
                    "values": values,
                    "holdouts": [],
                    "eligible": True,
                    "full": 0.0,
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
                passed = math.isfinite(error) and error <= limit
                errors.append(error)
                if not passed:
                    failure_counts[metric] += 1
                holdouts.append(
                    {
                        "scale": scale,
                        "actual": actual,
                        "prediction": prediction,
                        "relative_error": error,
                        "pass_5pct": passed,
                    }
                )
            eligible = all(item["pass_5pct"] for item in holdouts)
            path_eligible = path_eligible and eligible
            path_models[metric] = {
                "classification": "affine_nonzero",
                "intercept": affine.intercept,
                "slope": affine.slope,
                "values": values,
                "holdouts": holdouts,
                "eligible": eligible,
                "full": affine.predict(full_scale) if eligible else None,
            }
        models[path_key] = {
            "family": "fft",
            "full_scale": full_scale,
            "metrics": path_models,
            "eligible": path_eligible,
        }
        full_cycles = path_models["cycles"]["full"]
        projected = {}
        for metric in metrics:
            full_value = path_models[metric]["full"]
            projected[metric] = full_value
            if metric != "cycles":
                projected[f"{metric}_utilization"] = (
                    full_value
                    / (full_cycles * int(base_config["hardware"]["physical_pes"]))
                    if full_value is not None and full_cycles is not None
                    else None
                )
        full_estimates[path_key] = {
            "eligible": path_eligible,
            "metrics": projected,
            "fma_issue_utilization_completed_work": h114["full_estimates"][path_key][
                "fma_issue_utilization"
            ],
            "fma_residence_semantics": (
                "productive_fma_pe_cycles_div_cycles_times_physical_pes"
            ),
        }

    utilization_checks = {
        path_key: item["eligible"]
        and all(
            value is None or math.isfinite(value) and 0 <= value <= 1
            for key, value in item["metrics"].items()
            if key.endswith("_utilization")
        )
        for path_key, item in full_estimates.items()
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    serialized = json.dumps(config, sort_keys=True).lower()
    target_free = (
        "paper_targets" not in serialized
        and "fig22" not in serialized
        and "fig25" not in serialized
        and "residual_scale" not in serialized
        and "family_correction" not in serialized
    )
    counts = {
        "paths": len(path_keys) == int(config["execution"]["required_paths"]),
        "configs": len(compiled["outputs"])
        == int(config["execution"]["required_configs"]),
        "records": len(run["records"])
        == int(config["execution"]["required_executions"]),
        "parent_records": len(parent_records) == 16,
        "combined_measurements": len(measurements) == 32,
    }
    all_holdouts = all(
        holdout["pass_5pct"]
        for path in models.values()
        for metric in path["metrics"].values()
        for holdout in metric["holdouts"]
    )
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        all(compile_checks.values()) and all(counts.values()),
        all(run["checks"].values())
        and all(run["replay_checks"].values())
        and all(run["sanitizer_checks"].values()),
        all(execution_checks.values()),
        h114_run_file["pass"] and all(record_checks.values()),
        all(
            metric["classification"] != "exact_zero"
            or all(value == 0 for value in metric["values"].values())
            for path in models.values()
            for metric in path["metrics"].values()
        ),
        all(
            holdout["pass_5pct"]
            for path in models.values()
            for metric_name, metric in path["metrics"].items()
            if metric_name == "cycles"
            for holdout in metric["holdouts"]
        ),
        all_holdouts,
        all(utilization_checks.values())
        and all(item["eligible"] for item in full_estimates.values()),
        all(
            item["fma_residence_semantics"]
            != "completed_fma_issue_throughput"
            for item in full_estimates.values()
        ),
        target_free and all(item["pass"] for item in source_files.values()),
        config["classification"] == "target_free_fft_coupled_counter_steady_state"
        and config["validation_eligible"] is False,
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "compile_manifest": compile_file["pass"],
        "run_manifest": run_file["pass"] and all(run["checks"].values()),
        "compile": all(compile_checks.values()),
        "execution": all(execution_checks.values()),
        "records": all(record_checks.values()),
        "counts": all(counts.values()),
        "models_evaluated": len(models) == 8,
        "source_files": all(item["pass"] for item in source_files.values()),
        "target_free": target_free,
        "acceptance_evaluated": len(acceptance_gates) == 12
        and all(isinstance(value, bool) for value in acceptance_gates),
    }
    integrity = all(integrity_checks.values())
    supported = integrity and all(acceptance_gates)
    finite_errors = [error for error in errors if math.isfinite(error)]
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
        "paper_reproduction_claim": "none_target_free_fft_counter_extension_only",
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "compile_manifest": compile_file,
        "run_manifest": run_file,
        "compile_checks": compile_checks,
        "execution_checks": execution_checks,
        "record_checks": record_checks,
        "models": models,
        "full_estimates": full_estimates,
        "utilization_checks": utilization_checks,
        "failure_counts": dict(failure_counts),
        "counts": counts,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "paths": len(path_keys),
            "new_configs": len(compiled["outputs"]),
            "executions": len(run["records"]),
            "sanitizer_executions": sum(
                item["mode"] in {"asan", "ubsan"} for item in run["records"]
            ),
            "metric_holdouts": len(errors),
            "metric_holdouts_passed": sum(
                math.isfinite(error) and error <= limit for error in errors
            ),
            "metric_mape": sum(finite_errors) / len(finite_errors),
            "metric_max_error": max(finite_errors),
            "eligible_full_paths": sum(item["eligible"] for item in models.values()),
            "failure_counts": dict(failure_counts),
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
            "full_estimates",
            "failure_counts",
            "acceptance_gates",
            "summary",
            "integrity_checks",
        )
        matches = all(
            json.dumps(existing.get(key), sort_keys=True)
            == json.dumps(json.loads(json.dumps(report.get(key))), sort_keys=True)
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
