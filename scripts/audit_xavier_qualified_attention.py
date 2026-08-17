#!/usr/bin/env python3
"""Audit H86 stable-FFT and short-SV Xavier qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from mlxsim.repeat_folding import fit_affine, relative_error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/xavier_qualified_attention_v1.yaml"


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


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    files = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    h85_spec = config["frozen_inputs"]["h85"]
    mlx_spec = config["frozen_inputs"]["mlx"]
    h85 = json.loads((PROJECT_ROOT / h85_spec["path"]).read_text(encoding="utf-8"))
    mlx = json.loads((PROJECT_ROOT / mlx_spec["path"]).read_text(encoding="utf-8"))
    parent_checks = {
        "h85": h85["hypothesis_status"] == h85_spec["required_status"]
        and h85["audit_integrity"] is h85_spec["required_integrity"],
        "mlx": mlx["hypothesis_status"] == mlx_spec["required_status"]
        and mlx["audit_integrity"] is mlx_spec["required_integrity"],
    }
    output_root = PROJECT_ROOT / config["output_root"]
    run_path = output_root / "xavier-qualified-run-manifest.json"
    run_file = qualify(run_path)
    run = json.loads(run_path.read_text(encoding="utf-8"))
    jobs = {job["name"]: job for job in config["jobs"]}
    run_checks = {}
    measurements = {}
    for name, job in jobs.items():
        record = run["records"][name]
        path = PROJECT_ROOT / record["path"]
        artifact = qualify(path, {"sha256": record["sha256"]})
        measurement = json.loads(path.read_text(encoding="utf-8"))
        checksum = float(measurement["run"]["summary"]["relative_error"])
        checks = {
            "artifact": artifact["pass"],
            "job": measurement["job"] == job,
            "cycles": measurement["cycles"] == record["cycles"] > 0,
            "pass": measurement["pass"] is True,
            "checksum": checksum <= float(config["checksum_relative_error_limit"]),
            "detailed": measurement["checks"]["detailed"],
            "exit": measurement["checks"]["exit"],
        }
        run_checks[name] = all(checks.values())
        measurements[name] = {
            "artifact": artifact,
            "cycles": int(measurement["cycles"]),
            "instructions": int(measurement["instructions"]),
            "ctas": int(measurement["ctas"]),
            "checksum_relative_error": checksum,
            "checks": checks,
        }

    def cycles(name: str) -> int:
        if name in measurements:
            return measurements[name]["cycles"]
        return int(h85["measurements"][name]["cycles"])

    specs = {
        "N256_fftcmp": ((2048, "N256-stablefft-c2048"), (4096, "N256-stablefft-c4096"), (8192, "N256-stablefft-c8192")),
        "N8192_fftcmp": ((2048, "N8192-stablefft-c2048"), (4096, "N8192-stablefft-c4096"), (8192, "N8192-stablefft-c8192")),
        "N256_sv": ((2048, "N256-sv-c2048"), (4096, "N256-sv-c4096"), (8192, "N256-sv-c8192")),
    }
    models = {}
    errors = []
    limit = float(config["cycle_relative_error_limit"])
    for name, (first, second, holdout) in specs.items():
        model = fit_affine(first[0], cycles(first[1]), second[0], cycles(second[1]))
        actual = cycles(holdout[1])
        predicted = model.predict(holdout[0])
        error = relative_error(predicted, actual)
        errors.append(error)
        models[name] = {
            "fit": [
                {"count": first[0], "measurement": first[1], "cycles": cycles(first[1])},
                {"count": second[0], "measurement": second[1], "cycles": cycles(second[1])},
            ],
            "intercept": model.intercept,
            "slope_cycles_per_count": model.slope,
            "holdout": {
                "count": holdout[0],
                "measurement": holdout[1],
                "actual_cycles": actual,
                "predicted_cycles": predicted,
                "relative_error": error,
                "pass_5pct": error <= limit,
            },
        }
    numerical = {
        "passing_holdouts": sum(error <= limit for error in errors),
        "total_holdouts": len(errors),
        "mape": sum(errors) / len(errors),
        "max_error": max(errors),
        "all_holdouts_pass": all(error <= limit for error in errors),
    }
    reuse_checks = {
        "shared_qk": h85["models"]["shared_qk"]["holdout"]["pass_5pct"],
        "N8192_sv": h85["models"]["N8192_sv"]["holdout"]["pass_5pct"],
        "N256_softmax_direct": "N256-softmax-c128" in h85["measurements"],
        "N8192_softmax_direct": "N8192-softmax-c4096" in h85["measurements"],
    }
    full_estimates = {
        "N256": {
            "fftcmp": models["N256_fftcmp"]["intercept"]
            + models["N256_fftcmp"]["slope_cycles_per_count"]
            * int(config["full_counts"]["N256"]["fftcmp"]),
            "qk": h85["models"]["shared_qk"]["intercept"]
            + h85["models"]["shared_qk"]["slope_cycles_per_count"]
            * int(config["full_counts"]["N256"]["qk"]),
            "softmax": int(h85["measurements"]["N256-softmax-c128"]["cycles"]),
            "sv": models["N256_sv"]["intercept"]
            + models["N256_sv"]["slope_cycles_per_count"]
            * int(config["full_counts"]["N256"]["sv"]),
        },
        "N8192": {
            "fftcmp": models["N8192_fftcmp"]["intercept"]
            + models["N8192_fftcmp"]["slope_cycles_per_count"]
            * int(config["full_counts"]["N8192"]["fftcmp"]),
            "qk": h85["models"]["shared_qk"]["intercept"]
            + h85["models"]["shared_qk"]["slope_cycles_per_count"]
            * int(config["full_counts"]["N8192"]["qk"]),
            "softmax": int(h85["measurements"]["N8192-softmax-c4096"]["cycles"]),
            "sv": h85["models"]["N8192_sv"]["intercept"]
            + h85["models"]["N8192_sv"]["slope_cycles_per_count"]
            * int(config["full_counts"]["N8192"]["sv"]),
        },
    }
    for estimate in full_estimates.values():
        estimate["total_cycles"] = sum(estimate.values())
        estimate["eligible"] = numerical["all_holdouts_pass"] and all(
            reuse_checks.values()
        )

    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    stable_text = (
        PROJECT_ROOT / config["source_layout"]["stable_fft_source"]
    ).read_text(encoding="utf-8")
    source_checks = {
        "stable_coefficients": "stage_cosine" in stable_text
        and "stage_sine" in stable_text,
        "no_transcendental": "sinf(" not in stable_text and "cosf(" not in stable_text,
        "four_fma": all(token in stable_text for token in ("p0 = fmaf", "p3 = fmaf")),
        "summary": "MLX_GPU_PROXY_SUMMARY" in stable_text,
    }
    implementation_text = "\n".join(
        (PROJECT_ROOT / path).read_text(encoding="utf-8")
        for name, path in config["source_layout"].items()
        if name != "auditor"
    ).lower()
    integrity_checks = {
        "frozen_files": all(item["pass"] for item in files.values()),
        "parents": all(parent_checks.values()),
        "run_manifest": run_file["pass"]
        and run["paper_performance_targets_consumed"] is False
        and len(run["records"]) == len(jobs) == 7,
        "runs": all(run_checks.values()),
        "reuse": all(reuse_checks.values()),
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
        "run_manifest": run_file,
        "run_checks": run_checks,
        "measurements": measurements,
        "models": models,
        "numerical": numerical,
        "reuse_checks": reuse_checks,
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
            "models",
            "numerical",
            "reuse_checks",
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
