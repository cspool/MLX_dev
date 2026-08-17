#!/usr/bin/env python3
"""Audit H55's no-fit Figure 24 DSAGEN/Orin cross-simulator transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_dsagen_fig25_transfer import relative_error
from scripts.audit_dsagen_mlx_dma_memory import git_revision, load_yaml, qualify_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig24_cross_simulator_v1.yaml"
EVIDENCE_ROOT = PROJECT_ROOT / "artifacts/environment/h55"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--verify-existing", action="store_true")
    return parser.parse_args()


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    parent_artifacts = {}
    parent_checks = {}
    for name, specification in config["parents"].items():
        artifact = qualify_file(PROJECT_ROOT / specification["path"], specification)
        report = json.loads(
            (PROJECT_ROOT / specification["path"]).read_text(encoding="utf-8")
        )
        parent_checks[name] = artifact["pass"] and report.get(
            "hypothesis_status"
        ) == specification["required_status"] and report.get("audit_integrity") is specification[
            "required_integrity"
        ]
        parent_artifacts[name] = artifact

    manifest_path = EVIDENCE_ROOT / "fig24-cross-simulator-compile-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mlx_outputs = {item["name"]: item for item in manifest["mlx_outputs"]}
    jobs = {item["name"]: item for item in manifest["orin_jobs"]}
    expected_names = {
        f"{operator['name']}--{case['name']}"
        for operator in config["operators"]
        for case in config["cases"]
    }
    compile_checks = {
        "count": manifest.get("output_count") == 42,
        "mlx_matrix": set(mlx_outputs) == expected_names,
        "orin_matrix": set(jobs) == expected_names,
        "target_independent": manifest.get("paper_target_values_consumed") is False,
        "paper_static": all(
            item["metadata"].get("pe_dependency_model") == "paper_static"
            for item in mlx_outputs.values()
        ),
        "fmas": all(
            int(item["mlx_fma_equivalents"]) > 0 for item in mlx_outputs.values()
        )
        and all(int(item["gpu_fma_equivalents"]) > 0 for item in jobs.values()),
    }

    targets = yaml.safe_load(
        (PROJECT_ROOT / config["targets"]["path"]).read_text(encoding="utf-8")
    )["fig24_structured_sweep"]["mlx_over_orin"]
    mlx_clock = int(config["normalization"]["mlx_clock_hz"])
    rows = []
    errors = []
    run_checks = {}
    measurements = {}
    for operator in config["operators"]:
        cells = []
        for case_index, case in enumerate(config["cases"]):
            name = f"{operator['name']}--{case['name']}"
            mlx_path = EVIDENCE_ROOT / f"runs/mlx/{name}/measurement.json"
            orin_path = EVIDENCE_ROOT / f"runs/orin/{name}/measurement.json"
            mlx = json.loads(mlx_path.read_text(encoding="utf-8"))
            orin = json.loads(orin_path.read_text(encoding="utf-8"))
            mlx_fmas = int(mlx_outputs[name]["mlx_fma_equivalents"])
            mlx_cycles = int(mlx["metrics"]["cycles"])
            mlx_seconds_per_fma = mlx_cycles / mlx_clock / mlx_fmas
            orin_seconds_per_fma = float(orin["metrics"]["seconds_per_fma"])
            ratio = orin_seconds_per_fma / mlx_seconds_per_fma
            target = float(targets[operator["name"]][case_index])
            error = relative_error(ratio, target)
            errors.append(error)
            hazard_stalls = {
                key
                for key in mlx["overlay"].get("stalls_by_reason", {})
                if key.startswith(("register_", "rf_"))
            }
            run_checks[name] = (
                mlx.get("pass") is True
                and orin.get("pass") is True
                and mlx["overlay"].get("pe_dependency_model") == "paper_static"
                and not hazard_stalls
                and orin["job"] == jobs[name]
                and mlx_fmas > 0
            )
            measurements[name] = {
                "mlx": qualify_file(mlx_path),
                "orin": qualify_file(orin_path),
                "mlx_cycles": mlx_cycles,
                "mlx_fma_equivalents": mlx_fmas,
                "mlx_seconds_per_fma": mlx_seconds_per_fma,
                "orin_cycles": orin["metrics"]["cycles"],
                "orin_fma_equivalents": orin["metrics"]["fma_equivalents"],
                "orin_seconds_per_fma": orin_seconds_per_fma,
                "ratio": ratio,
            }
            cells.append(
                {
                    "case": case["name"],
                    "ratio": ratio,
                    "target": target,
                    "relative_error": error,
                    "pass_10pct": error <= 0.10,
                }
            )
        rows.append(
            {
                "operator": operator["name"],
                "cells": cells,
                "pass_10pct": all(cell["pass_10pct"] for cell in cells),
            }
        )
    numerical = {
        "metric": config["normalization"]["ratio"],
        "exact_kernel_identity": config["normalization"]["exact_kernel_identity"],
        "validation_eligible": False,
        "rows": rows,
        "cell_count": len(errors),
        "cells_within_10pct": sum(error <= 0.10 for error in errors),
        "mape": sum(errors) / len(errors),
        "max_error": max(errors),
        "pass_10pct": all(error <= 0.10 for error in errors),
    }
    source_files = {
        key: qualify_file(PROJECT_ROOT / path)
        for key, path in config["source_layout"].items()
    }
    integrity_checks = {
        "parents": all(parent_checks.values()),
        "compiler": all(compile_checks.values()),
        "runs": len(run_checks) == 42 and all(run_checks.values()),
        "source": all(item["pass"] for item in source_files.values()),
        "target_visibility": config["targets"]["execution_visibility"] == "audit_only",
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
        "parents": {"artifacts": parent_artifacts, "checks": parent_checks},
        "compiler": {
            "manifest": qualify_file(manifest_path),
            "checks": compile_checks,
        },
        "source": source_files,
        "runs": {"measurements": measurements, "checks": run_checks},
        "numerical_audit": numerical,
        "integrity_checks": integrity_checks,
        "paper_target_values_consumed_during_execution": False,
        "old_calibration_coefficients_consumed": False,
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
            raise SystemExit("existing H55 result does not match a fresh audit")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            {
                "hypothesis_status": report["hypothesis_status"],
                "audit_integrity": report["audit_integrity"],
                "cells_within_10pct": report["numerical_audit"]["cells_within_10pct"],
                "mape": report["numerical_audit"]["mape"],
                "max_error": report["numerical_audit"]["max_error"],
            },
            indent=2,
        )
    )
    return 0 if report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
