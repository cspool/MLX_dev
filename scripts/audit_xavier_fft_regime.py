#!/usr/bin/env python3
"""Audit H133 regime-aware stable Xavier FFT-CMP folding."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from mlxsim.repeat_folding import fit_affine
from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/xavier_fft_regime_v1.yaml"


def measurement(record: dict[str, Any], job: dict[str, Any], checksum_limit: float) -> tuple[dict[str, Any], bool]:
    path = PROJECT_ROOT / record["path"]
    artifact = qualify(path, {"sha256": record["sha256"]})
    value = json.loads(path.read_text())
    checksum = float(value["run"]["summary"]["relative_error"])
    checks = {
        "artifact": artifact["pass"],
        "job": value["job"] == job,
        "cycles": value["cycles"] == record["cycles"] > 0,
        "instructions": int(value["instructions"]) > 0,
        "ctas": int(value["ctas"]) > 0,
        "pass": value["pass"] is True,
        "checksum": checksum <= checksum_limit,
        "detailed": value["checks"]["detailed"],
        "exit": value["checks"]["exit"],
    }
    return (
        {
            "artifact": artifact,
            "cycles": int(value["cycles"]),
            "instructions": int(value["instructions"]),
            "ctas": int(value["ctas"]),
            "checksum_relative_error": checksum,
            "checks": checks,
        },
        all(checks.values()),
    )


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    h87 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h87"]["path"]).read_text()
    )
    h126 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h126_regime_evidence"]["path"]).read_text()
    )
    parent_manifest = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h87_manifest"]["path"]).read_text()
    )
    parent_checks = {
        "h87": h87["hypothesis_status"] == "rejected"
        and h87["audit_integrity"] is True,
        "h87_fft_failures": all(
            not h87["models"][name]["holdout"]["pass_5pct"]
            for name in ("N256_fftcmp", "N8192_fftcmp")
        ),
        "h126_regime": h126["hypothesis_status"] == "supported"
        and h126["audit_integrity"] is True,
    }
    output_root = PROJECT_ROOT / config["output_root"]
    run_path = output_root / "xavier-fft-regime-run-manifest.json"
    run_file = qualify(run_path)
    run = json.loads(run_path.read_text())
    jobs = {job["name"]: job for job in config["jobs"]}
    measurements: dict[str, Any] = {}
    run_checks: dict[str, bool] = {}
    checksum_limit = float(config["checksum_relative_error_limit"])
    for name, job in jobs.items():
        measurements[name], run_checks[name] = measurement(
            run["records"][name], job, checksum_limit
        )
    parent_jobs = {
        job["name"]: job
        for job in yaml.safe_load(
            (PROJECT_ROOT / config["frozen_inputs"]["h87_config"]["path"]).read_text()
        )["jobs"]
    }
    parent_measurements: dict[str, Any] = {}
    parent_run_checks: dict[str, bool] = {}
    for model_spec in config["models"].values():
        name = model_spec["parent_anchor"]
        parent_measurements[name], parent_run_checks[name] = measurement(
            parent_manifest["records"][name], parent_jobs[name], checksum_limit
        )
    models: dict[str, Any] = {}
    errors = []
    full_estimates: dict[str, Any] = {}
    limit = float(config["cycle_relative_error_limit"])
    for name, specification in config["models"].items():
        parent_name = specification["parent_anchor"]
        anchor_name = specification["new_anchor"]
        holdout_name = specification["holdout"]
        parent_count = int(parent_jobs[parent_name]["count"])
        anchor_count = int(jobs[anchor_name]["count"])
        holdout_count = int(jobs[holdout_name]["count"])
        model = fit_affine(
            parent_count,
            parent_measurements[parent_name]["cycles"],
            anchor_count,
            measurements[anchor_name]["cycles"],
        )
        prediction = model.predict(holdout_count)
        actual = measurements[holdout_name]["cycles"]
        error = abs(prediction - actual) / actual
        errors.append(error)
        eligible = error <= limit
        full_count = int(specification["full_count"])
        full_cycles = model.predict(full_count)
        models[name] = {
            "intercept": model.intercept,
            "slope": model.slope,
            "fit": [parent_name, anchor_name],
            "holdout": {
                "measurement": holdout_name,
                "count": holdout_count,
                "actual": actual,
                "prediction": prediction,
                "relative_error": error,
                "pass_5pct": eligible,
            },
            "eligible": eligible,
        }
        full_estimates[name] = {
            "full_count": full_count,
            "cycles": full_cycles if eligible else None,
            "seconds": (
                full_cycles / int(config["device_clock_hz"]) if eligible else None
            ),
            "eligible": eligible,
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
        "run": run["paper_performance_targets_consumed"] is False,
        "no_target": "fig20" + "_speedup" not in source_text,
        "no_mlx": "combined-attention" + "-memory-run088" not in source_text,
        "no_residual": "residual" + "_factor" not in source_text,
    }
    all_holdouts_pass = all(model["eligible"] for model in models.values())
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        len(jobs) == 4,
        all(run_checks.values()),
        run_file["pass"] and len(run["binaries"]) == 1,
        all(parent_run_checks.values()),
        all_holdouts_pass,
        {item["full_count"] for item in full_estimates.values()}
        == {1_572_864, 50_331_648},
        all(item["cycles"] is not None and item["cycles"] > 0 for item in full_estimates.values()),
        all(target_free_checks.values()) and all(item["pass"] for item in source_files.values()),
        config["validation_eligible"] is False,
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "run_manifest": run_file["pass"]
        and run["paper_performance_targets_consumed"] is False,
        "runs": all(run_checks.values()),
        "parent_runs": all(parent_run_checks.values()),
        "models_evaluated": len(models) == 2,
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
        "paper_reproduction_claim": "none_target_free_xavier_fft_regime_only",
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "run_manifest": run_file,
        "run_checks": run_checks,
        "parent_run_checks": parent_run_checks,
        "measurements": measurements,
        "parent_measurements": parent_measurements,
        "models": models,
        "full_estimates": full_estimates,
        "target_free_checks": target_free_checks,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "new_runs": len(measurements),
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
