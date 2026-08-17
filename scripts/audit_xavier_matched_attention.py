#!/usr/bin/env python3
"""Audit H84 execution-driven Xavier Attention component folding."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from compile_xavier_matched_attention import work

from mlxsim.repeat_folding import fit_affine, relative_error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/xavier_matched_attention_v1.yaml"


def qualify(path: Path, expected: dict[str, Any] | None = None) -> dict[str, Any]:
    exists = path.is_file()
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if exists else None
    checks = {"is_file": exists}
    if expected and "sha256" in expected:
        checks["sha256"] = digest == expected["sha256"]
    try:
        display = str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        display = str(path)
    return {
        "path": display,
        "bytes": path.stat().st_size if exists else None,
        "sha256": digest,
        "checks": checks,
        "pass": all(checks.values()),
    }


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={PROJECT_ROOT}", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _parent_check(report: dict[str, Any], spec: dict[str, Any]) -> bool:
    return (
        report.get("hypothesis_status") == spec["required_status"]
        and report.get("audit_integrity") is spec["required_integrity"]
    )


def _full_work(shape: dict[str, Any]) -> dict[str, dict[str, int]]:
    return {
        family: work(family, int(shape["full_counts"][family]), shape)
        for family in ("fftcmp", "qk", "softmax", "sv")
    }


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    files = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    parents = {}
    parent_checks = {}
    for name in ("mlx", "xavier", "signature"):
        spec = config["frozen_inputs"][name]
        report = json.loads(
            (PROJECT_ROOT / spec["path"]).read_text(encoding="utf-8")
        )
        parents[name] = report
        parent_checks[name] = _parent_check(report, spec)

    output_root = PROJECT_ROOT / config["output_root"]
    compile_path = output_root / "xavier-attention-compile-manifest.json"
    run_path = output_root / "xavier-attention-run-manifest.json"
    compile_file = qualify(compile_path)
    run_file = qualify(run_path)
    compiler = json.loads(compile_path.read_text(encoding="utf-8"))
    run = json.loads(run_path.read_text(encoding="utf-8"))
    jobs = {job["name"]: job for job in compiler["jobs"]}

    run_checks = {}
    measurements = {}
    for name, job in jobs.items():
        record = run["records"][name]
        measurement_path = PROJECT_ROOT / record["path"]
        artifact = qualify(measurement_path, {"sha256": record["sha256"]})
        measurement = json.loads(measurement_path.read_text(encoding="utf-8"))
        expected_work = work(job["family"], int(job["count"]), config["shapes"][job["shape"]])
        checks = {
            "artifact": artifact["pass"],
            "pass": measurement["pass"] is True and all(measurement["checks"].values()),
            "job": measurement["job"] == job,
            "cycles": measurement["cycles"] == record["cycles"] > 0,
            "work": job["work"] == expected_work,
            "checksum": measurement["run"]["summary"]["relative_error"]
            <= float(config["checksum_relative_error_limit"]),
        }
        run_checks[name] = all(checks.values())
        measurements[name] = {
            "artifact": artifact,
            "cycles": int(measurement["cycles"]),
            "instructions": int(measurement["instructions"]),
            "ctas": int(measurement["ctas"]),
            "summary": measurement["run"]["summary"],
            "checks": checks,
        }

    signature_checks = {}
    full_work = {}
    for shape_name, shape in config["shapes"].items():
        derived = _full_work(shape)
        signature = parents["signature"]["signatures"][shape_name]
        fft = signature["fft_compression"]["fu_instruction_instances"]
        attention = signature["compressed_attention"]["fu_instruction_instances"]
        checks = {
            "fft_fma": derived["fftcmp"]["fma"] == fft["fma"],
            "fft_add": derived["fftcmp"]["add"] == fft["alu_add"],
            "fft_shuffle": derived["fftcmp"]["shuffle"] == fft["shuffle"],
            "qk_fma": derived["qk"]["fma"] * 2 == attention["fma"],
            "softmax_fmax": derived["softmax"]["fmax"] == attention["fmax"],
            "softmax_fexp": derived["softmax"]["fexp"] == attention["fexp"],
            "softmax_add": derived["softmax"]["add"] == attention["alu_add"],
            "sv_fma": derived["sv"]["fma"] * 2 == attention["fma"],
            "sv_fdiv": derived["sv"]["fdiv"] == attention["fdiv"],
        }
        signature_checks[shape_name] = all(checks.values())
        full_work[shape_name] = {"families": derived, "checks": checks}

    models = {}
    errors = []
    limit = float(config["cycle_relative_error_limit"])
    for shape_name, shape in config["shapes"].items():
        models[shape_name] = {}
        for family, ranges in config["families"].items():
            fit_counts = [int(value) for value in ranges["fit_counts"]]
            holdout_counts = [int(value) for value in ranges["holdout_counts"]]
            model = fit_affine(
                fit_counts[0],
                measurements[f"{shape_name}-{family}-c{fit_counts[0]}"]["cycles"],
                fit_counts[1],
                measurements[f"{shape_name}-{family}-c{fit_counts[1]}"]["cycles"],
            )
            holdouts = []
            for count in holdout_counts:
                actual = measurements[f"{shape_name}-{family}-c{count}"]["cycles"]
                predicted = model.predict(count)
                error = relative_error(predicted, actual)
                errors.append(error)
                holdouts.append(
                    {
                        "count": count,
                        "actual_cycles": actual,
                        "predicted_cycles": predicted,
                        "relative_error": error,
                        "pass_5pct": error <= limit,
                    }
                )
            full_count = int(shape["full_counts"][family])
            models[shape_name][family] = {
                "intercept": model.intercept,
                "slope_cycles_per_count": model.slope,
                "holdouts": holdouts,
                "full_count": full_count,
                "full_predicted_cycles": model.predict(full_count),
            }

    numerical = {
        "passing_holdouts": sum(error <= limit for error in errors),
        "total_holdouts": len(errors),
        "mape": sum(errors) / len(errors),
        "max_error": max(errors),
        "all_holdouts_pass": all(error <= limit for error in errors),
    }
    full_estimates = {}
    for shape_name, family_models in models.items():
        components = {
            family: item["full_predicted_cycles"]
            for family, item in family_models.items()
        }
        full_estimates[shape_name] = {
            "components": components,
            "total_cycles": sum(components.values()),
            "eligible": numerical["all_holdouts_pass"],
        }

    gpgpu_root = PROJECT_ROOT / "third_party/accel-sim-framework/gpu-simulator/gpgpu-sim"
    revision = subprocess.run(
        ["git", "-c", f"safe.directory={gpgpu_root}", "rev-parse", "HEAD"],
        cwd=gpgpu_root,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    binary = qualify(PROJECT_ROOT / run["binary"]["path"], run["binary"])
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    cuda_text = (
        PROJECT_ROOT / config["source_layout"]["cuda_source"]
    ).read_text(encoding="utf-8")
    source_checks = {
        "fft_pair": all(
            token in cuda_text
            for token in ("fft_pair_stage", "truncate_half", "run_fftcmp")
        ),
        "attention_components": all(
            token in cuda_text
            for token in ("qk_scores", "softmax_stats", "sv_outputs")
        ),
        "fu_mix": all(token in cuda_text for token in ("fmaf", "fmaxf", "expf")),
        "summary": "MLX_GPU_PROXY_SUMMARY" in cuda_text,
    }
    implementation_text = "\n".join(
        (PROJECT_ROOT / path).read_text(encoding="utf-8")
        for name, path in config["source_layout"].items()
        if name != "auditor"
    ).lower()
    integrity_checks = {
        "frozen_files": all(item["pass"] for item in files.values()),
        "parents": all(parent_checks.values()),
        "revision": revision == config["gpgpu_sim_revision"],
        "compile_manifest": compile_file["pass"]
        and compiler["paper_performance_targets_consumed"] is False
        and len(jobs) == 32,
        "run_manifest": run_file["pass"]
        and run["paper_performance_targets_consumed"] is False
        and len(run["records"]) == 32,
        "runs": all(run_checks.values()),
        "signature": all(signature_checks.values()),
        "binary": binary["pass"],
        "source_files": all(item["pass"] for item in source_files.values()),
        "source_contract": all(source_checks.values()),
        "targets_absent": "paper_targets" not in implementation_text,
    }
    integrity = all(integrity_checks.values())
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "classification": config["classification"],
        "validation_eligible": config["validation_eligible"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_commit": git_commit(),
        "hypothesis_status": (
            "supported" if integrity and numerical["all_holdouts_pass"] else "rejected"
        ),
        "audit_integrity": integrity,
        "frozen_inputs": files,
        "parent_checks": parent_checks,
        "gpgpu_sim_revision": revision,
        "compile_manifest": compile_file,
        "run_manifest": run_file,
        "binary": binary,
        "run_checks": run_checks,
        "measurements": measurements,
        "full_work": full_work,
        "signature_checks": signature_checks,
        "models": models,
        "numerical": numerical,
        "full_estimates": full_estimates,
        "source_files": source_files,
        "source_checks": source_checks,
        "integrity_checks": integrity_checks,
        "paper_performance_targets_consumed": False,
        "stopping_rule_applied": True,
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
        existing = json.loads(output.read_text(encoding="utf-8"))
        keys = (
            "hypothesis_status",
            "audit_integrity",
            "run_checks",
            "full_work",
            "models",
            "numerical",
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
            {
                "hypothesis_status": report["hypothesis_status"],
                "audit_integrity": report["audit_integrity"],
                "numerical": report["numerical"],
            },
            indent=2,
        )
    )
    return 0 if report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
