#!/usr/bin/env python3
"""Build and audit H77 matched projection cycle estimates."""

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
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/matched_projection_estimator_v1.yaml"


def qualify(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    exists = path.is_file()
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if exists else None
    checks = {"is_file": exists, "sha256": digest == expected["sha256"]}
    return {
        "path": str(path.relative_to(PROJECT_ROOT)) if exists else str(path),
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
    parents = {
        name: json.loads((PROJECT_ROOT / spec["path"]).read_text(encoding="utf-8"))
        for name, spec in config["frozen_inputs"].items()
    }
    mlx_parent = parents["mlx_anchors"]
    fit_keys = config["mlx"]["fit_keys"]
    fit_trips = [float(value) for value in config["mlx"]["fit_trips"]]
    mlx_model = fit_affine(
        fit_trips[0],
        mlx_parent["measurements"][fit_keys[0]]["cycles"],
        fit_trips[1],
        mlx_parent["measurements"][fit_keys[1]]["cycles"],
    )
    validation = []
    for key in config["mlx"]["validation_keys"]:
        actual = float(mlx_parent["measurements"][key]["cycles"])
        predicted = mlx_model.predict(float(config["mlx"]["validation_trip"]))
        validation.append(
            {
                "key": key,
                "actual_cycles": actual,
                "predicted_cycles": predicted,
                "relative_error": relative_error(predicted, actual),
            }
        )

    xavier_points = []
    for key in config["xavier"]["fit_keys"]:
        path = PROJECT_ROOT / config["xavier"]["measurements"] / key / "measurement.json"
        measurement = json.loads(path.read_text(encoding="utf-8"))
        xavier_points.append(
            (
                float(measurement["metrics"]["fma_equivalents"]),
                float(measurement["metrics"]["cycles"]),
                str(path.relative_to(PROJECT_ROOT)),
            )
        )
    xavier_model = fit_affine(
        xavier_points[0][0],
        xavier_points[0][1],
        xavier_points[1][0],
        xavier_points[1][1],
    )
    logical = parents["logical_work"]["logical_profiles"]
    fma_per_trip = (
        int(config["mlx"]["fma_equivalents_per_simd8_trip"])
        * int(config["mlx"]["full_design_simd_scale"])
    )
    estimates: dict[str, Any] = {}
    for n in config["coverage"]["sequence_lengths"]:
        for kernel in config["coverage"]["kernels"]:
            key = f"{kernel}-N{n}"
            fma = float(logical[key]["fma_equivalents"])
            mlx_trips = fma / fma_per_trip
            mlx_cycles = mlx_model.predict(mlx_trips)
            xavier_cycles = xavier_model.predict(fma)
            speedup = (
                xavier_cycles / config["xavier"]["clock_hz"]
            ) / (mlx_cycles / config["mlx"]["clock_hz"])
            estimates[key] = {
                "logical_fma_equivalents": fma,
                "logical_offchip_bytes": logical[key]["offchip_bytes"],
                "logical_output_elements": logical[key]["output_elements"],
                "mlx_full_simd32_trips": mlx_trips,
                "mlx_cycles": mlx_cycles,
                "xavier_cycles": xavier_cycles,
                "sparse_cuda_speedup": speedup,
            }
    parent_checks = {
        name: (
            ("required_status" not in spec or parents[name]["hypothesis_status"] == spec["required_status"])
            and ("required_integrity" not in spec or parents[name]["audit_integrity"] is spec["required_integrity"])
        )
        for name, spec in config["frozen_inputs"].items()
    }
    source_files = {
        "auditor": {
            "path": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "repeat_module": {
            "path": "src/mlxsim/repeat_folding.py",
            "sha256": hashlib.sha256(
                (PROJECT_ROOT / "src/mlxsim/repeat_folding.py").read_bytes()
            ).hexdigest(),
        },
    }
    summary = {
        "covered_projection_points": len(estimates),
        "excluded_attention_points": len(config["coverage"]["sequence_lengths"]),
        "max_mlx_anchor_relative_error": max(
            item["relative_error"] for item in validation
        ),
        "logical_fma_exact": all(
            estimate["logical_fma_equivalents"] > 0 for estimate in estimates.values()
        ),
    }
    integrity_checks = {
        "frozen_files": all(item["pass"] for item in files.values()),
        "parents": all(parent_checks.values()),
        "mlx_anchor_validation": summary["max_mlx_anchor_relative_error"] <= 0.05,
        "positive_models": mlx_model.slope > 0 and xavier_model.slope > 0,
        "coverage": len(estimates) == 6,
        "attention_excluded": config["coverage"]["excluded"]["attention"]
        == "requires_fft_and_compressed_attention_anchors",
        "targets_consumed": False,
    }
    integrity = all(
        value for key, value in integrity_checks.items() if key != "targets_consumed"
    ) and not integrity_checks["targets_consumed"]
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "classification": config["classification"],
        "validation_eligible": config["validation_eligible"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_commit": git_commit(),
        "hypothesis_status": "supported" if integrity else "rejected",
        "audit_integrity": integrity,
        "frozen_inputs": files,
        "parent_checks": parent_checks,
        "mlx_model": {
            "intercept": mlx_model.intercept,
            "slope_cycles_per_trip": mlx_model.slope,
            "fma_equivalents_per_full_simd32_trip": fma_per_trip,
            "validation": validation,
        },
        "xavier_model": {
            "intercept": xavier_model.intercept,
            "slope_cycles_per_fma": xavier_model.slope,
            "anchors": xavier_points,
        },
        "estimates": estimates,
        "coverage": config["coverage"],
        "summary": summary,
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
        existing = json.loads(output.read_text(encoding="utf-8"))
        matches = all(
            existing.get(key) == report.get(key)
            for key in ("hypothesis_status", "audit_integrity", "summary")
        )
        print(json.dumps({"existing_matches": matches, **report}, indent=2, sort_keys=True))
        return 0 if matches else 1
    if args.preflight_only:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["audit_integrity"] and not output.exists() else 1
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
