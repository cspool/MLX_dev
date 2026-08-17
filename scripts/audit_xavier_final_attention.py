#!/usr/bin/env python3
"""Audit H87 final Xavier Attention folding gate."""

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
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/xavier_final_attention_v1.yaml"


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
    parent_spec = config["frozen_inputs"]["h86"]
    parent = json.loads(
        (PROJECT_ROOT / parent_spec["path"]).read_text(encoding="utf-8")
    )
    parent_check = (
        parent["hypothesis_status"] == parent_spec["required_status"]
        and parent["audit_integrity"] is parent_spec["required_integrity"]
        and parent["numerical"]["all_holdouts_pass"] is False
    )
    output_root = PROJECT_ROOT / config["output_root"]
    run_path = output_root / "xavier-final-run-manifest.json"
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

    def parent_cycle(name: str) -> int:
        if name in parent["measurements"]:
            return int(parent["measurements"][name]["cycles"])
        if name == "N256-sv-c4096":
            return int(parent["models"]["N256_sv"]["fit"][1]["cycles"])
        raise KeyError(name)

    specs = {
        "N256_fftcmp": ((4096, "N256-stablefft-c4096"), (8192, "N256-stablefft-c8192"), (16384, "N256-stablefft-c16384")),
        "N8192_fftcmp": ((4096, "N8192-stablefft-c4096"), (8192, "N8192-stablefft-c8192"), (16384, "N8192-stablefft-c16384")),
        "N256_sv": ((4096, "N256-sv-c4096"), (8192, "N256-sv-c8192"), (16384, "N256-sv-c16384")),
    }
    models = {}
    errors = []
    limit = float(config["cycle_relative_error_limit"])
    for name, (first, second, holdout) in specs.items():
        model = fit_affine(
            first[0], parent_cycle(first[1]), second[0], parent_cycle(second[1])
        )
        actual = measurements[holdout[1]]["cycles"]
        predicted = model.predict(holdout[0])
        error = relative_error(predicted, actual)
        errors.append(error)
        models[name] = {
            "fit": [
                {"count": first[0], "measurement": first[1], "cycles": parent_cycle(first[1])},
                {"count": second[0], "measurement": second[1], "cycles": parent_cycle(second[1])},
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
        "shared_qk": parent["reuse_checks"]["shared_qk"],
        "N8192_sv": parent["reuse_checks"]["N8192_sv"],
        "N256_softmax_direct": parent["reuse_checks"]["N256_softmax_direct"],
        "N8192_softmax_direct": parent["reuse_checks"]["N8192_softmax_direct"],
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    implementation_text = "\n".join(
        (PROJECT_ROOT / path).read_text(encoding="utf-8")
        for name, path in config["source_layout"].items()
        if name != "auditor"
    ).lower()
    integrity_checks = {
        "frozen_files": all(item["pass"] for item in files.values()),
        "parent": parent_check,
        "run_manifest": run_file["pass"]
        and run["paper_performance_targets_consumed"] is False
        and len(run["records"]) == len(jobs) == 3,
        "runs": all(run_checks.values()),
        "reuse": all(reuse_checks.values()),
        "source_files": all(item["pass"] for item in source_files.values()),
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
        "parent_check": parent_check,
        "run_manifest": run_file,
        "run_checks": run_checks,
        "measurements": measurements,
        "models": models,
        "numerical": numerical,
        "reuse_checks": reuse_checks,
        "integrity_checks": integrity_checks,
        "paper_performance_targets_consumed": False,
        "stopping_rule_applied": True,
        "conclusion": "final affine Xavier folding remains above the 5% gate",
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
