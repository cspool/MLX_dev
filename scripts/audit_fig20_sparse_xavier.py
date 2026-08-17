#!/usr/bin/env python3
"""Audit H57's partial Figure 20 sparse-CUDA transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from scripts.audit_dsagen_fig25_transfer import relative_error
from scripts.audit_dsagen_mlx_dma_memory import git_revision, load_yaml, qualify_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig20_sparse_xavier_v1.yaml"
ROOT = PROJECT_ROOT / "artifacts/environment/h57"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--verify-existing", action="store_true")
    return parser.parse_args()


def build_audit(config: dict) -> dict:
    parent_checks = {}
    parents = {}
    for name, spec in config["parents"].items():
        artifact = qualify_file(PROJECT_ROOT / spec["path"], spec)
        report = json.loads((PROJECT_ROOT / spec["path"]).read_text(encoding="utf-8"))
        parent_checks[name] = artifact["pass"] and report.get(
            "hypothesis_status"
        ) == spec["required_status"] and report.get("audit_integrity") is spec[
            "required_integrity"
        ]
        parents[name] = artifact
    manifest_path = ROOT / "fig20-sparse-xavier-compile-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fmas = {item["name"]: item["mlx_fma_equivalents"] for item in manifest["mlx_outputs"]}
    targets = yaml.safe_load(
        (PROJECT_ROOT / config["targets"]["path"]).read_text(encoding="utf-8")
    )["fig20_xavier_kernels"]["versus_sparse_cuda"]
    cells = []
    run_checks = {}
    measurements = {}
    errors = []
    index = 0
    for case in ("short", "long"):
        for kernel_name in ("QKV", "Attn", "FFN1", "FFN2"):
            proxy = config["kernels"][kernel_name]["proxy"]
            name = f"{proxy}--{case}"
            mlx_path = ROOT / f"runs/mlx/{name}/measurement.json"
            gpu_path = ROOT / f"runs/xavier/{name}/measurement.json"
            mlx = json.loads(mlx_path.read_text(encoding="utf-8"))
            gpu = json.loads(gpu_path.read_text(encoding="utf-8"))
            mlx_spf = mlx["metrics"]["cycles"] / config["normalization"][
                "mlx_clock_hz"
            ] / fmas[name]
            gpu_spf = gpu["metrics"]["seconds_per_fma"]
            speedup = gpu_spf / mlx_spf
            target = float(targets["speedup"][index])
            error = relative_error(speedup, target)
            errors.append(error)
            energy = speedup * config["normalization"]["xavier_power_w"] / config[
                "normalization"
            ]["mlx_power_w"]
            cells.append(
                {
                    "group": f"{kernel_name}_{config['cases'][case]['label']}",
                    "speedup": speedup,
                    "speedup_target": target,
                    "relative_error": error,
                    "pass_10pct": error <= 0.10,
                    "fixed_power_energy_saving": energy,
                    "energy_target": targets["energy_saving"][index],
                }
            )
            run_checks[name] = mlx.get("pass") is True and gpu.get("pass") is True
            measurements[name] = {
                "mlx": qualify_file(mlx_path),
                "xavier": qualify_file(gpu_path),
            }
            index += 1
    numerical = {
        "cells": cells,
        "cells_within_10pct": sum(error <= 0.10 for error in errors),
        "mape": sum(errors) / len(errors),
        "max_error": max(errors),
        "pass_10pct": all(error <= 0.10 for error in errors),
    }
    source = {
        key: qualify_file(PROJECT_ROOT / path)
        for key, path in config["source_layout"].items()
    }
    integrity_checks = {
        "parents": all(parent_checks.values()),
        "manifest": len(manifest["mlx_outputs"]) == len(manifest["gpu_jobs"]) == 4
        and manifest.get("paper_target_values_consumed") is False,
        "runs": len(run_checks) == 4 and all(run_checks.values()),
        "source": all(item["pass"] for item in source.values()),
        "targets_audit_only": config["targets"]["execution_visibility"] == "audit_only",
    }
    integrity = all(integrity_checks.values())
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "classification": config["classification"],
        "hypothesis_status": "supported" if numerical["pass_10pct"] else "rejected",
        "audit_integrity": integrity,
        "validation_eligible": False,
        "git_revision": git_revision(PROJECT_ROOT),
        "parents": {"artifacts": parents, "checks": parent_checks},
        "compiler": qualify_file(manifest_path),
        "source": source,
        "runs": {"measurements": measurements, "checks": run_checks},
        "sparse_cuda_speedup": numerical,
        "energy_diagnostic": {
            "eligible": False,
            "reason": "fixed device power is not per-kernel activity power",
        },
        "dense_tcu": {
            "available": False,
            "reason": "no execution-driven Tensor Core kernel in H56",
        },
        "integrity_checks": integrity_checks,
        "paper_target_values_consumed_during_execution": False,
        "stopping_rule_applied": True,
    }


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config.resolve())
    report = build_audit(config)
    output = PROJECT_ROOT / config["result_path"]
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.verify_existing:
        if not output.is_file() or output.read_text(encoding="utf-8") != rendered:
            raise SystemExit("existing H57 result does not match a fresh audit")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(json.dumps({"status": report["hypothesis_status"], **report["sparse_cuda_speedup"]}, indent=2))
    return 0 if report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
