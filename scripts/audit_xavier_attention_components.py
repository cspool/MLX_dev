#!/usr/bin/env python3
"""Audit H134 regime-aware Xavier QK/SV and direct softmax components."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from mlxsim.repeat_folding import fit_affine
from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/xavier_attention_components_v1.yaml"


def read_measurement(
    record: dict[str, Any], expected_job: dict[str, Any], checksum_limit: float
) -> tuple[dict[str, Any], bool]:
    path = PROJECT_ROOT / record["path"]
    artifact = qualify(path, {"sha256": record["sha256"]})
    value = json.loads(path.read_text())
    checksum = float(value["run"]["summary"]["relative_error"])
    checks = {
        "artifact": artifact["pass"],
        "job": value["job"] == expected_job,
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
            "job": expected_job,
            "cycles": int(value["cycles"]),
            "instructions": int(value["instructions"]),
            "ctas": int(value["ctas"]),
            "checksum_relative_error": checksum,
            "checks": checks,
        },
        all(checks.values()),
    )


def job_from_record(record: dict[str, Any]) -> dict[str, Any]:
    return json.loads((PROJECT_ROOT / record["path"]).read_text())["job"]


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    h133 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h133"]["path"]).read_text()
    )
    h85 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h85"]["path"]).read_text()
    )
    h87 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h87"]["path"]).read_text()
    )
    h85_manifest = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h85_manifest"]["path"]).read_text()
    )
    h87_manifest = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h87_manifest"]["path"]).read_text()
    )
    parent_checks = {
        "h133": h133["hypothesis_status"] == "supported"
        and h133["audit_integrity"] is True,
        "h85_status_retained": h85["hypothesis_status"] == "rejected"
        and h85["audit_integrity"] is False,
        "h87": h87["hypothesis_status"] == "rejected"
        and h87["audit_integrity"] is True,
    }
    output_root = PROJECT_ROOT / config["output_root"]
    run_path = output_root / "xavier-attention-components-run-manifest.json"
    run_file = qualify(run_path)
    run = json.loads(run_path.read_text())
    jobs = {job["name"]: job for job in config["jobs"]}
    checksum_limit = float(config["checksum_relative_error_limit"])
    measurements: dict[str, Any] = {}
    run_checks: dict[str, bool] = {}
    for name, job in jobs.items():
        measurements[name], run_checks[name] = read_measurement(
            run["records"][name], job, checksum_limit
        )
    parent_records = {**h85_manifest["records"], **h87_manifest["records"]}
    parent_names = {
        specification["parent_anchor"] for specification in config["models"].values()
    } | {
        specification["measurement"]
        for specification in config["direct_softmax"].values()
    }
    parent_measurements: dict[str, Any] = {}
    parent_run_checks: dict[str, bool] = {}
    for name in parent_names:
        record = parent_records[name]
        job = job_from_record(record)
        parent_measurements[name], parent_run_checks[name] = read_measurement(
            record, job, checksum_limit
        )
    parent_shape_checks = {
        "shared_qk": parent_measurements["shared-qk-c4096"]["job"]["family"]
        == "qk"
        and parent_measurements["shared-qk-c4096"]["job"]["count"] == 4096,
        "N256_sv": parent_measurements["N256-sv-c16384"]["job"]["family"]
        == "sv"
        and parent_measurements["N256-sv-c16384"]["job"]["count"] == 16384,
        "N8192_sv": parent_measurements["N8192-sv-c4096"]["job"]["family"]
        == "sv"
        and parent_measurements["N8192-sv-c4096"]["job"]["count"] == 4096,
        "softmax": parent_measurements["N256-softmax-c128"]["job"]["count"]
        == 128
        and parent_measurements["N8192-softmax-c4096"]["job"]["count"] == 4096,
    }
    models: dict[str, Any] = {}
    errors = []
    component_estimates: dict[str, dict[str, Any]] = {"N256": {}, "N8192": {}}
    limit = float(config["cycle_relative_error_limit"])
    for name, specification in config["models"].items():
        parent_name = specification["parent_anchor"]
        anchor_name = specification["new_anchor"]
        holdout_name = specification["holdout"]
        parent_count = int(parent_measurements[parent_name]["job"]["count"])
        anchor_count = int(measurements[anchor_name]["job"]["count"])
        holdout_count = int(measurements[holdout_name]["job"]["count"])
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
        models[name] = {
            "intercept": model.intercept,
            "slope": model.slope,
            "fit": [parent_name, anchor_name],
            "holdout": {
                "measurement": holdout_name,
                "actual": actual,
                "prediction": prediction,
                "relative_error": error,
                "pass_5pct": eligible,
            },
            "eligible": eligible,
        }
        for shape, full_count_value in specification["full_counts"].items():
            full_count = int(full_count_value)
            cycles = model.predict(full_count)
            component = "qk" if name == "shared_qk" else "sv"
            component_estimates[shape][component] = {
                "full_count": full_count,
                "cycles": cycles if eligible else None,
                "seconds": (
                    cycles / int(config["device_clock_hz"]) if eligible else None
                ),
                "eligible": eligible,
            }
    for shape, specification in config["direct_softmax"].items():
        name = specification["measurement"]
        component_estimates[shape]["softmax"] = {
            "full_count": int(specification["full_count"]),
            "cycles": parent_measurements[name]["cycles"],
            "seconds": parent_measurements[name]["cycles"]
            / int(config["device_clock_hz"]),
            "eligible": parent_run_checks[name],
            "direct_measurement": name,
        }
    full_count_checks = {
        "N256": component_estimates["N256"]["qk"]["full_count"] == 16_384
        and component_estimates["N256"]["sv"]["full_count"] == 524_288
        and component_estimates["N256"]["softmax"]["full_count"] == 128,
        "N8192": component_estimates["N8192"]["qk"]["full_count"] == 16_777_216
        and component_estimates["N8192"]["sv"]["full_count"] == 16_777_216
        and component_estimates["N8192"]["softmax"]["full_count"] == 4096,
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
    all_components_eligible = all(
        item["eligible"]
        for shape in component_estimates.values()
        for item in shape.values()
    )
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        len(jobs) == 6,
        all(run_checks.values()),
        run_file["pass"] and len(run["binaries"]) == 1,
        all(parent_run_checks.values()) and all(parent_shape_checks.values()),
        all_holdouts_pass,
        all(full_count_checks.values()),
        all_components_eligible,
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
        "parent_shapes": all(parent_shape_checks.values()),
        "models_evaluated": len(models) == 3,
        "components_evaluated": all(
            set(shape) == {"qk", "sv", "softmax"}
            for shape in component_estimates.values()
        ),
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
        "paper_reproduction_claim": "none_target_free_xavier_components_only",
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "run_manifest": run_file,
        "run_checks": run_checks,
        "parent_run_checks": parent_run_checks,
        "parent_shape_checks": parent_shape_checks,
        "measurements": measurements,
        "parent_measurements": parent_measurements,
        "models": models,
        "component_estimates": component_estimates,
        "full_count_checks": full_count_checks,
        "target_free_checks": target_free_checks,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "new_runs": len(measurements),
            "parent_records": len(parent_measurements),
            "holdouts": len(errors),
            "holdouts_passed": sum(error <= limit for error in errors),
            "holdout_mape": sum(errors) / len(errors),
            "holdout_max_error": max(errors),
            "eligible_components": sum(
                item["eligible"]
                for shape in component_estimates.values()
                for item in shape.values()
            ),
            "required_components": 6,
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
            "component_estimates",
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
