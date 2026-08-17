#!/usr/bin/env python3
"""Audit H50's source-arithmetic-expanded Figure 25 transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_dsagen_fig25_transfer import relative_error
from scripts.audit_dsagen_mlx_dma_memory import git_revision, load_yaml, qualify_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/dsagen_mlx_fig25_arithmetic_v1.yaml"
EVIDENCE_ROOT = PROJECT_ROOT / "artifacts/environment/h50"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--verify-existing", action="store_true")
    return parser.parse_args()


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    parent_spec = config["parent"]
    parent_artifact = qualify_file(PROJECT_ROOT / parent_spec["path"], parent_spec)
    parent = json.loads((PROJECT_ROOT / parent_spec["path"]).read_text(encoding="utf-8"))
    mapping = load_yaml(PROJECT_ROOT / config["mapping_config"])
    manifest_path = EVIDENCE_ROOT / "fig25-arithmetic-compile-manifest.json"
    replay_path = EVIDENCE_ROOT / "compiler-replay-check.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    outputs = {(item["operator"], item["case"]): item for item in manifest["outputs"]}

    compile_checks: dict[str, bool] = {}
    for operator in mapping["operators"]:
        for case in mapping["cases"]:
            key = (operator["name"], case["name"])
            item = outputs[key]
            metadata = item["metadata"]
            operation_counts = metadata["operation_counts"]
            trip = int(case["trip_count"])
            if operator["family"] == "fft":
                expected_fma = 6 * 3 * 4 * 4 * trip
                expected_add = 6 * 3 * 4 * 6 * trip
                arithmetic = operation_counts.get("fma") == expected_fma and operation_counts.get(
                    "add"
                ) == expected_add
            elif operator["family"] == "qkv_bsmm":
                depth = int(operator["stages"])
                expected_fma = depth * 3 * 4 * 4 * trip
                expected_add = depth * 3 * 4 * 2 * trip
                arithmetic = operation_counts.get("fma") == expected_fma and operation_counts.get(
                    "add"
                ) == expected_add
            else:
                expansion = config["arithmetic_expansion"][operator["name"]]
                expected_fma = 4 * (
                    int(expansion["score_fma_groups"])
                    + int(expansion["sv_fma_groups"])
                ) * trip
                expected_loads = 2 * 4 * int(expansion["kv_load_waves"]) * trip
                arithmetic = operation_counts.get("fma") == expected_fma and metadata[
                    "pipeline_counts"
                ]["load"] == expected_loads
            compile_checks[f"{key[0]}--{key[1]}"] = (
                metadata.get("arithmetic_expanded") is True
                and metadata.get("paper_target_values_consumed") is False
                and arithmetic
            )

    run_items: dict[str, Any] = {}
    run_checks: dict[str, bool] = {}
    measured_rows = []
    targets = yaml.safe_load(
        (PROJECT_ROOT / config["targets"]["path"]).read_text(encoding="utf-8")
    )["fig25_roofline_utilization"]["heatmap"]["mlx"]
    all_errors = []
    for operator_index, operator in enumerate(mapping["operators"]):
        cells = []
        for case_index, case in enumerate(mapping["cases"]):
            name = f"{operator['name']}--{case['name']}"
            root = EVIDENCE_ROOT / "runs" / name
            measurement_path = root / "measurement.json"
            measurement = json.loads(measurement_path.read_text(encoding="utf-8"))
            artifacts = {
                "measurement": qualify_file(measurement_path),
                "log": qualify_file(root / "run.log"),
                "stats": qualify_file(root / "m5out/stats.txt"),
            }
            run_checks[name] = measurement.get("pass") is True and all(
                artifact["pass"] for artifact in artifacts.values()
            )
            value = measurement["metrics"]["compute_pipeline_occupancy"]
            target = float(targets[operator_index][case_index])
            error = relative_error(value, target)
            all_errors.append(error)
            cells.append(
                {
                    "case": case["name"],
                    "measured": value,
                    "target": target,
                    "relative_error": error,
                    "pass_10pct": error <= 0.10,
                }
            )
            run_items[name] = {"artifacts": artifacts, "measurement": measurement}
        measured_rows.append(
            {
                "operator": operator["name"],
                "cells": cells,
                "pass_10pct": all(cell["pass_10pct"] for cell in cells),
            }
        )

    numerical = {
        "metric": "global_compute_pipeline_busy_cycles_div_overlay_cycles",
        "paper_metric": "peak_or_bandwidth_roofline_normalized_fma_utilization",
        "metric_identity": False,
        "validation_eligible": False,
        "rows": measured_rows,
        "cell_count": len(all_errors),
        "cells_within_10pct": sum(error <= 0.10 for error in all_errors),
        "mape": sum(all_errors) / len(all_errors),
        "max_error": max(all_errors),
        "pass_10pct": all(error <= 0.10 for error in all_errors),
    }
    integrity_checks = {
        "parent": parent_artifact["pass"]
        and parent.get("hypothesis_status") == parent_spec["required_status"]
        and parent.get("audit_integrity") is parent_spec["required_integrity"],
        "manifest": manifest.get("output_count") == 24 == len(outputs),
        "compiler": all(compile_checks.values()),
        "replay": replay.get("all_identical") is True
        and len(replay.get("comparisons") or []) == 24,
        "runs": all(run_checks.values()),
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
        "parent": parent_artifact,
        "compiler": {
            "manifest": qualify_file(manifest_path),
            "replay": qualify_file(replay_path),
            "checks": compile_checks,
            "pass": all(compile_checks.values()),
        },
        "runs": {"items": run_items, "checks": run_checks, "pass": all(run_checks.values())},
        "numerical_audit": numerical,
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
            raise SystemExit("existing H50 result does not match a fresh audit")
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
