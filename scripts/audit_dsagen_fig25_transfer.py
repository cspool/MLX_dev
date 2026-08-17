#!/usr/bin/env python3
"""Audit H49's no-fit DSAGEN transfer to the MLX rows of Figure 25."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_dsagen_mlx_dma_memory import (
    git_revision,
    load_yaml,
    qualify_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/dsagen_mlx_fig25_transfer_v1.yaml"
EVIDENCE_ROOT = PROJECT_ROOT / "artifacts/environment/h49"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--verify-existing", action="store_true")
    return parser.parse_args()


def relative_error(measured: float, target: float) -> float:
    return abs(measured - target) / abs(target)


def source_audit(config: dict[str, Any]) -> dict[str, Any]:
    layout = config["source_layout"]
    token_map = {
        "compiler_core": [
            "compile_operator_proxy",
            "operator_stages",
            "paper_target_values_consumed",
        ],
        "compiler_cli": ["output_count", "reference_dir", "all_identical"],
        "runner": ["check_dsagen_fig25_run.py", "MLX_FIG25_OUTPUT_ROOT"],
        "auditor": ["fig25_roofline_utilization", "relative_error"],
    }
    files = {}
    for key, tokens in token_map.items():
        path = PROJECT_ROOT / layout[key]
        text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
        checks = {token: token in text for token in tokens}
        files[key] = {
            "path": layout[key],
            "tokens": checks,
            "pass": path.is_file() and all(checks.values()),
        }
    compiler_text = (PROJECT_ROOT / layout["compiler_core"]).read_text(encoding="utf-8")
    target_tokens = ["paper_targets.yaml", "fig25_roofline_utilization", "heatmap"]
    checks = {
        "files": all(item["pass"] for item in files.values()),
        "targets_absent_from_compiler": not any(token in compiler_text for token in target_tokens),
    }
    return {"files": files, "checks": checks, "pass": all(checks.values())}


def compiler_audit(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = EVIDENCE_ROOT / "fig25-transfer-compile-manifest.json"
    replay_path = EVIDENCE_ROOT / "compiler-replay-check.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    expected = {
        (operator["name"], case["name"]): (operator, case)
        for operator in config["operators"]
        for case in config["cases"]
    }
    outputs: dict[tuple[str, str], dict[str, Any]] = {}
    structural_checks: dict[str, bool] = {}
    for item in manifest["outputs"]:
        key = (item["operator"], item["case"])
        outputs[key] = item
        document = json.loads(Path(item["path"]).read_text(encoding="utf-8"))
        metadata = document["metadata"]
        operator, case = expected[key]
        waits = {
            event
            for block in document["blocks"]
            for event in block.get("wait_events") or []
        }
        emitters = {
            instruction["emit_event"]: block["tag"]
            for block in document["blocks"]
            for instruction in block["instructions"]
            if instruction.get("emit_event")
        }
        adjacent = all(
            event in emitters
            and block["tag"] == emitters[event] + 1
            for block in document["blocks"]
            for event in block.get("wait_events") or []
        )
        structural_checks[f"{key[0]}--{key[1]}"] = (
            metadata["operator"] == operator
            and metadata["case"] == case
            and metadata["stage_count"] == int(operator.get("stages", 4))
            and metadata["trip_count"] == int(case["trip_count"])
            and metadata["paper_target_values_consumed"] is False
            and document["memory_backend"] == "dsagen_dma"
            and document["active_window"] == 4
            and adjacent
            and waits.issubset(emitters)
        )
    replay_items = replay.get("comparisons") or []
    checks = {
        "manifest": manifest.get("output_count") == 24 == len(outputs),
        "expected_matrix": set(outputs) == set(expected),
        "structures": all(structural_checks.values()),
        "replay": replay.get("all_identical") is True
        and len(replay_items) == 24
        and all(item.get("identical") is True for item in replay_items),
        "no_targets": manifest.get("paper_target_values_consumed") is False,
    }
    return {
        "manifest": qualify_file(manifest_path),
        "replay": qualify_file(replay_path),
        "structural_checks": structural_checks,
        "checks": checks,
        "pass": all(checks.values()),
    }


def run_audit(config: dict[str, Any]) -> dict[str, Any]:
    runs: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    for operator in config["operators"]:
        for case in config["cases"]:
            name = f"{operator['name']}--{case['name']}"
            root = EVIDENCE_ROOT / "runs" / name
            measurement_path = root / "measurement.json"
            measurement = json.loads(measurement_path.read_text(encoding="utf-8"))
            item = {
                "measurement": qualify_file(measurement_path),
                "log": qualify_file(root / "run.log"),
                "stats": qualify_file(root / "m5out/stats.txt"),
                "result": measurement,
            }
            checks[name] = (
                measurement.get("pass") is True
                and measurement.get("operator") == operator["name"]
                and measurement.get("case") == case["name"]
                and 0 < measurement.get("metrics", {}).get("compute_pipeline_occupancy", 0) < 1
                and all(artifact["pass"] for artifact in (item["measurement"], item["log"], item["stats"]))
            )
            runs[name] = item
    return {"runs": runs, "checks": checks, "pass": all(checks.values())}


def numerical_audit(config: dict[str, Any], runs: dict[str, Any]) -> dict[str, Any]:
    targets_path = PROJECT_ROOT / config["frozen_inputs"]["targets"]["path"]
    targets = yaml.safe_load(targets_path.read_text(encoding="utf-8"))[
        "fig25_roofline_utilization"
    ]["heatmap"]["mlx"]
    rows = []
    all_errors = []
    for operator_index, operator in enumerate(config["operators"]):
        measured = []
        errors = []
        cells = []
        for case_index, case in enumerate(config["cases"]):
            name = f"{operator['name']}--{case['name']}"
            value = runs[name]["result"]["metrics"]["compute_pipeline_occupancy"]
            target = float(targets[operator_index][case_index])
            error = relative_error(value, target)
            measured.append(value)
            errors.append(error)
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
        rows.append(
            {
                "operator": operator["name"],
                "measured": measured,
                "targets": targets[operator_index],
                "cells": cells,
                "mape": sum(errors) / len(errors),
                "max_error": max(errors),
                "pass_10pct": all(error <= 0.10 for error in errors),
            }
        )
    return {
        "metric": "global_compute_pipeline_busy_cycles_div_overlay_cycles",
        "paper_metric": "peak_or_bandwidth_roofline_normalized_fma_utilization",
        "metric_identity": False,
        "validation_eligible": False,
        "rows": rows,
        "cell_count": len(all_errors),
        "cells_within_10pct": sum(error <= 0.10 for error in all_errors),
        "mape": sum(all_errors) / len(all_errors),
        "max_error": max(all_errors),
        "pass_10pct": all(error <= 0.10 for error in all_errors),
    }


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    parent_spec = config["frozen_inputs"]["full_block_result"]
    target_spec = config["frozen_inputs"]["targets"]
    parent_artifact = qualify_file(PROJECT_ROOT / parent_spec["path"], parent_spec)
    target_artifact = qualify_file(PROJECT_ROOT / target_spec["path"], target_spec)
    parent = json.loads((PROJECT_ROOT / parent_spec["path"]).read_text(encoding="utf-8"))
    source = source_audit(config)
    compiler = compiler_audit(config)
    runs = run_audit(config)
    numerical = numerical_audit(config, runs["runs"])
    integrity_checks = {
        "parent": parent_artifact["pass"]
        and parent.get("hypothesis_status") == parent_spec["required_status"]
        and parent.get("audit_integrity") is parent_spec["required_integrity"],
        "targets": target_artifact["pass"],
        "source": source["pass"],
        "compiler": compiler["pass"],
        "runs": runs["pass"],
        "post_run_target_access": target_spec["execution_visibility"] == "audit_only",
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
        "targets": target_artifact,
        "source": source,
        "compiler": compiler,
        "runs": runs,
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
            raise SystemExit("existing H49 result does not match a fresh audit")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            {
                "hypothesis_status": report["hypothesis_status"],
                "audit_integrity": report["audit_integrity"],
                "cells_within_10pct": report["numerical_audit"]["cells_within_10pct"],
                "cell_count": report["numerical_audit"]["cell_count"],
                "mape": report["numerical_audit"]["mape"],
                "max_error": report["numerical_audit"]["max_error"],
            },
            indent=2,
        )
    )
    return 0 if report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
