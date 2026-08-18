#!/usr/bin/env python3
"""Audit H124 QKV GPGPU-Sim Orin repeat folding."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from mlxsim.repeat_folding import fit_affine
from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig24_qkv_orin_folding_v1.yaml"


def last_integer(text: str, name: str) -> int | None:
    matches = re.findall(rf"^{re.escape(name)} = ([0-9]+)$", text, re.MULTILINE)
    return int(matches[-1]) if matches else None


def parse_run(path: Path) -> dict[str, Any]:
    text = path.read_text(errors="replace")
    matches = re.findall(r"^MLX_FIG24_SCHEDULE_SUMMARY (\{.*\})$", text, re.MULTILINE)
    return {
        "summary": json.loads(matches[-1]) if matches else None,
        "cycles": last_integer(text, "gpu_tot_sim_cycle"),
        "instructions": last_integer(text, "gpu_tot_sim_insn"),
        "ctas": last_integer(text, "gpu_tot_issued_cta"),
        "detailed": "GPGPU-Sim uArch: performance model initialization complete."
        in text,
        "normal_exit": "GPGPU-Sim: *** exit detected ***" in text,
    }


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    h101 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h101"]["path"]).read_text()
    )
    h123 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h123"]["path"]).read_text()
    )
    h101_manifest = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h101_manifest"]["path"]).read_text()
    )
    parent_checks = {
        "h101": h101["hypothesis_status"] == "supported"
        and h101["audit_integrity"] is True,
        "h123": h123["hypothesis_status"] == "supported"
        and h123["audit_integrity"] is True
        and h123["schedule_sensitive"] is True,
    }
    output_root = PROJECT_ROOT / config["output_root"]
    manifest_path = output_root / "fig24-qkv-orin-run-manifest.json"
    manifest_file = qualify(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    records = {item["key"]: item for item in manifest["records"]}
    fit_scales = [int(value) for value in config["folding"]["fit_scales"]]
    holdout_scales = [int(value) for value in config["folding"]["holdout_scales"]]
    all_scales = [*fit_scales, *holdout_scales]
    limit = float(config["folding"]["relative_error_limit"])
    run_checks: dict[str, bool] = {}
    measurements: dict[str, Any] = {}
    for template, specification in config["templates"].items():
        stages = int(specification["stages"])
        for scale in all_scales:
            key = f"{template}-q{scale}"
            record = records[key]
            path = PROJECT_ROOT / record["artifact"]["path"]
            parsed = parse_run(path)
            summary = parsed["summary"] or {}
            count = int(config["folding"]["base_element_count"]) * scale
            expected_ctas = stages * math.ceil(
                count / int(config["folding"]["block_threads"])
            )
            expected_fma = (
                count
                * stages
                * int(config["folding"]["fma_per_element_stage"])
            )
            checks = {
                "artifact": qualify(path, record["artifact"])["pass"],
                "record": record["pass"] is True and record["returncode"] == 0,
                "shape": summary.get("count") == count
                and summary.get("stages") == stages
                and summary.get("block_threads")
                == int(config["folding"]["block_threads"]),
                "work": summary.get("scalar_fma") == expected_fma,
                "ctas": summary.get("total_ctas") == parsed["ctas"] == expected_ctas,
                "checksum": summary.get("relative_error", math.inf)
                <= config["acceptance"]["checksum_relative_error_limit"],
                "cycles": isinstance(parsed["cycles"], int) and parsed["cycles"] > 0,
                "instructions": isinstance(parsed["instructions"], int)
                and parsed["instructions"] > 0,
                "detailed": parsed["detailed"],
                "exit": parsed["normal_exit"],
            }
            run_checks[key] = all(checks.values())
            measurements[key] = {
                "template": template,
                "stages": stages,
                "scale": scale,
                "count": count,
                "scalar_fma": expected_fma,
                "cycles": parsed["cycles"],
                "instructions": parsed["instructions"],
                "ctas": parsed["ctas"],
                "checksum_relative_error": summary.get("relative_error"),
                "checks": checks,
            }
    models: dict[str, Any] = {}
    holdout_errors: list[float] = []
    for template in config["templates"]:
        model = fit_affine(
            fit_scales[0],
            measurements[f"{template}-q{fit_scales[0]}"]["cycles"],
            fit_scales[1],
            measurements[f"{template}-q{fit_scales[1]}"]["cycles"],
        )
        holdouts = []
        for scale in holdout_scales:
            actual = measurements[f"{template}-q{scale}"]["cycles"]
            prediction = model.predict(scale)
            error = abs(prediction - actual) / actual
            holdout_errors.append(error)
            holdouts.append(
                {
                    "scale": scale,
                    "actual": actual,
                    "prediction": prediction,
                    "relative_error": error,
                    "pass_5pct": error <= limit,
                }
            )
        models[template] = {
            "intercept": model.intercept,
            "slope": model.slope,
            "holdouts": holdouts,
            "eligible": all(item["pass_5pct"] for item in holdouts),
        }
    full_estimates: dict[str, Any] = {}
    full_checks: dict[str, bool] = {}
    cases = set(config["folding"]["figure24_cases"])
    for key, contract in h101_manifest["path_contracts"].items():
        if contract["family"] != "qkv_bsmm" or contract["case"]["name"] not in cases:
            continue
        operator = contract["operator"]["name"]
        stages = int(contract["actual"]["stage_count"])
        full_fma = int(contract["actual"]["fu"]["fma"])
        unit_fma = (
            int(config["folding"]["base_element_count"])
            * stages
            * int(config["folding"]["fma_per_element_stage"])
        )
        integral = full_fma % unit_fma == 0
        full_q = full_fma // unit_fma if integral else None
        model = models[operator]
        cycles = model["intercept"] + model["slope"] * full_q if integral else None
        full_checks[key] = (
            integral
            and full_q is not None
            and full_q > 0
            and model["eligible"]
            and math.isfinite(cycles)
            and cycles > 0
            and full_q * unit_fma == full_fma
        )
        full_estimates[key] = {
            "template": operator,
            "case": contract["case"],
            "stages": stages,
            "full_scalar_fma": full_fma,
            "unit_scalar_fma": unit_fma,
            "full_q": full_q,
            "cycles": cycles if full_checks[key] else None,
            "seconds": (
                cycles / int(config["folding"]["orin_clock_hz"])
                if full_checks[key]
                else None
            ),
            "cta_mapping": config["acceptance"]["cta_mapping"],
        }
    count_checks = {
        "templates": len(models) == int(config["folding"]["required_templates"]),
        "runs": len(measurements) == int(config["folding"]["required_runs"]),
        "holdouts": len(holdout_errors)
        == int(config["folding"]["required_holdouts"]),
        "full": len(full_estimates)
        == int(config["folding"]["required_full_estimates"]),
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
        "config": config["acceptance"]["targets_consumed"] is False,
        "manifest": manifest["paper_performance_targets_consumed"] is False,
        "no_target": "fig24_structured" + "_sweep" not in source_text,
        "no_mlx_cycles": "coupled-full-mesh" + "-paths-run119" not in source_text,
        "no_residual": "residual" + "_factor" not in source_text,
        "proxy_label": config["acceptance"]["author_cuda_mapping_claimed"] is False,
    }
    all_holdouts_pass = all(
        holdout["pass_5pct"]
        for model in models.values()
        for holdout in model["holdouts"]
    )
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        all(count_checks.values()),
        all(run_checks.values()),
        all(
            item["checksum_relative_error"]
            <= config["acceptance"]["checksum_relative_error_limit"]
            for item in measurements.values()
        ),
        manifest_file["pass"] and all(manifest["checks"].values()),
        all_holdouts_pass,
        len(full_checks) == 21 and all(full_checks.values()),
        len(full_estimates) == 21
        and all(item["cycles"] is not None for item in full_estimates.values()),
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
        "paper_reproduction_claim": "none_target_free_qkv_orin_folding_only",
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
            "runs": len(measurements),
            "holdouts": len(holdout_errors),
            "holdouts_passed": sum(error <= limit for error in holdout_errors),
            "holdout_mape": sum(holdout_errors) / len(holdout_errors),
            "holdout_max_error": max(holdout_errors),
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
