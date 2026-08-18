#!/usr/bin/env python3
"""Audit H126 q32/q64 to q128 post-cache QKV Orin folding."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from mlxsim.repeat_folding import fit_affine
from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify
from scripts.audit_fig24_qkv_orin_steady_state import audited_measurement

DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig24_qkv_orin_postcache_v1.yaml"


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    h125 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h125"]["path"]).read_text()
    )
    parent_manifest = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h125_manifest"]["path"]).read_text()
    )
    h125_config = yaml.safe_load(
        (PROJECT_ROOT / config["frozen_inputs"]["h125_config"]["path"]).read_text()
    )
    h124_config = yaml.safe_load(
        (PROJECT_ROOT / h125_config["frozen_inputs"]["h124_config"]["path"]).read_text()
    )
    parent_checks = {
        "status": h125["hypothesis_status"] == "rejected"
        and h125["audit_integrity"] is True,
        "q32_failures": sum(
            not holdout["pass_5pct"]
            for model in h125["models"].values()
            for holdout in model["holdouts"]
        )
        == 3,
        "q16_passes": all(
            holdout["pass_5pct"]
            for model in h125["models"].values()
            for holdout in model["holdouts"]
            if holdout["scale"] == 16
        ),
    }
    output_root = PROJECT_ROOT / config["output_root"]
    manifest_path = output_root / "fig24-qkv-orin-postcache-run-manifest.json"
    manifest_file = qualify(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    parent_records = {item["key"]: item for item in parent_manifest["records"]}
    child_records = {item["key"]: item for item in manifest["records"]}
    parent_scale = int(config["folding"]["parent_anchor_scale"])
    new_anchor = int(config["folding"]["new_anchor_scale"])
    holdout_scale = int(config["folding"]["holdout_scale"])
    base_count = int(config["folding"]["base_element_count"])
    block_threads = int(config["folding"]["block_threads"])
    fma_per = int(config["folding"]["fma_per_element_stage"])
    checksum_limit = float(h124_config["acceptance"]["checksum_relative_error_limit"])
    limit = float(config["folding"]["relative_error_limit"])
    measurements: dict[str, Any] = {}
    run_checks: dict[str, bool] = {}
    for template, specification in config["templates"].items():
        stages = int(specification["stages"])
        key = f"{template}-q{parent_scale}"
        measurements[key], run_checks[key] = audited_measurement(
            parent_records[key],
            stages=stages,
            scale=parent_scale,
            base_count=base_count,
            block_threads=block_threads,
            fma_per_element_stage=fma_per,
            checksum_limit=checksum_limit,
        )
        for scale in (new_anchor, holdout_scale):
            key = f"{template}-q{scale}"
            measurements[key], run_checks[key] = audited_measurement(
                child_records[key],
                stages=stages,
                scale=scale,
                base_count=base_count,
                block_threads=block_threads,
                fma_per_element_stage=fma_per,
                checksum_limit=checksum_limit,
            )
    models: dict[str, Any] = {}
    errors = []
    for template in config["templates"]:
        model = fit_affine(
            parent_scale,
            measurements[f"{template}-q{parent_scale}"]["cycles"],
            new_anchor,
            measurements[f"{template}-q{new_anchor}"]["cycles"],
        )
        actual = measurements[f"{template}-q{holdout_scale}"]["cycles"]
        prediction = model.predict(holdout_scale)
        error = abs(prediction - actual) / actual
        errors.append(error)
        models[template] = {
            "intercept": model.intercept,
            "slope": model.slope,
            "holdout": {
                "scale": holdout_scale,
                "actual": actual,
                "prediction": prediction,
                "relative_error": error,
                "pass_5pct": error <= limit,
            },
            "eligible": error <= limit,
        }
    full_estimates: dict[str, Any] = {}
    full_checks: dict[str, bool] = {}
    for key, parent in h125["full_estimates"].items():
        template = parent["template"]
        full_q = int(parent["full_q"])
        model = models[template]
        cycles = model["intercept"] + model["slope"] * full_q
        fma_exact = full_q * int(parent["unit_scalar_fma"]) == int(
            parent["full_scalar_fma"]
        )
        full_checks[key] = (
            model["eligible"]
            and full_q > 0
            and fma_exact
            and math.isfinite(cycles)
            and cycles > 0
        )
        full_estimates[key] = {
            **parent,
            "cycles": cycles if full_checks[key] else None,
            "seconds": (
                cycles / int(h124_config["folding"]["orin_clock_hz"])
                if full_checks[key]
                else None
            ),
            "fit_scales": [parent_scale, new_anchor],
        }
    count_checks = {
        "parent_anchors": sum(key.endswith(f"q{parent_scale}") for key in measurements)
        == int(config["folding"]["required_parent_anchors"]),
        "new_runs": len(child_records) == int(config["folding"]["required_new_runs"]),
        "measurements": len(measurements) == 9,
        "holdouts": len(errors) == int(config["folding"]["required_holdouts"]),
        "full": len(full_estimates) == int(config["folding"]["required_full_estimates"]),
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    source_text = "\n".join(
        (PROJECT_ROOT / path).read_text(errors="replace")
        for path in config["source_layout"].values()
    )
    target_free_checks = {
        "config": config["execution"]["targets_consumed"] is False,
        "manifest": manifest["paper_performance_targets_consumed"] is False,
        "no_target": "fig24_structured" + "_sweep" not in source_text,
        "no_mlx": "coupled-full-mesh" + "-paths-run119" not in source_text,
        "no_residual": "residual" + "_factor" not in source_text,
        "no_blend": "precache" + "_blend" not in source_text,
    }
    all_holdouts_pass = all(model["holdout"]["pass_5pct"] for model in models.values())
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        count_checks["parent_anchors"]
        and all(run_checks[f"{t}-q{parent_scale}"] for t in config["templates"]),
        count_checks["new_runs"]
        and all(run_checks[f"{t}-q{s}"] for t in config["templates"] for s in (new_anchor, holdout_scale)),
        manifest_file["pass"] and all(manifest["checks"].values()),
        all(run_checks.values()),
        all_holdouts_pass,
        len(full_checks) == 21 and all(full_checks.values()),
        len(full_estimates) == 21 and all(item["cycles"] is not None for item in full_estimates.values()),
        all(target_free_checks.values()) and all(item["pass"] for item in source_files.values()),
        config["validation_eligible"] is False,
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "manifest": manifest_file["pass"] and all(manifest["checks"].values()),
        "runs": all(run_checks.values()),
        "models_evaluated": len(models) == 3,
        "full_evaluated": len(full_estimates) == 21,
        "counts": all(count_checks.values()),
        "source": all(target_free_checks.values())
        and all(item["pass"] for item in source_files.values()),
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
        "paper_reproduction_claim": "none_target_free_qkv_orin_postcache_only",
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "run_manifest": manifest_file,
        "run_checks": run_checks,
        "measurements": measurements,
        "models": models,
        "full_checks": full_checks,
        "full_estimates": full_estimates,
        "count_checks": count_checks,
        "target_free_checks": target_free_checks,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "templates": len(models),
            "parent_anchors": 3,
            "new_runs": len(child_records),
            "holdouts": len(errors),
            "holdouts_passed": sum(error <= limit for error in errors),
            "holdout_mape": sum(errors) / len(errors),
            "holdout_max_error": max(errors),
            "full_estimates": len(full_estimates),
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
    config = yaml.safe_load(args.config.read_text())
    report = build_audit(config)
    output = PROJECT_ROOT / config["result_path"]
    if args.verify_existing:
        existing = json.loads(output.read_text())
        keys = (
            "hypothesis_status",
            "audit_integrity",
            "measurements",
            "models",
            "full_estimates",
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
