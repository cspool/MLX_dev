#!/usr/bin/env python3
"""Audit H53's Figure 25 transfer under paper-static PE semantics."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_dsagen_fig25_transfer import relative_error
from scripts.audit_dsagen_mlx_dma_memory import git_revision, load_yaml, qualify_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/dsagen_mlx_fig25_paper_static_v1.yaml"
EVIDENCE_ROOT = PROJECT_ROOT / "artifacts/environment/h53"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--verify-existing", action="store_true")
    return parser.parse_args()


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    parent_reports = {}
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
        parent_reports[name] = artifact

    manifest_path = EVIDENCE_ROOT / "fig25-paper-static-compile-manifest.json"
    replay_path = EVIDENCE_ROOT / "compiler-replay-check.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    transform_checks = {}
    for item in manifest["outputs"]:
        parent = json.loads(Path(item["parent"]["path"]).read_text(encoding="utf-8"))
        output = json.loads(Path(item["output"]["path"]).read_text(encoding="utf-8"))
        stripped = copy.deepcopy(output)
        model = stripped.pop("pe_dependency_model", None)
        metadata_model = stripped["metadata"].pop("pe_dependency_model", None)
        scoreboard_claim = stripped["metadata"].pop(
            "scoreboard_is_paper_semantics", None
        )
        transform_checks[item["name"]] = (
            stripped == parent
            and model == "paper_static"
            and metadata_model == "paper_static"
            and scoreboard_claim is False
        )

    mapping = load_yaml(PROJECT_ROOT / config["mapping_config"])
    targets = yaml.safe_load(
        (PROJECT_ROOT / config["targets"]["path"]).read_text(encoding="utf-8")
    )["fig25_roofline_utilization"]["heatmap"]["mlx"]
    runs = {}
    run_checks = {}
    rows = []
    errors = []
    hazard_prefixes = ("register_", "rf_")
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
            stalls = measurement["overlay"].get("stalls_by_reason", {})
            run_checks[name] = (
                measurement.get("pass") is True
                and measurement["overlay"].get("pe_dependency_model") == "paper_static"
                and not any(key.startswith(hazard_prefixes) for key in stalls)
                and all(artifact["pass"] for artifact in artifacts.values())
            )
            measured = measurement["metrics"]["compute_pipeline_occupancy"]
            target = float(targets[operator_index][case_index])
            error = relative_error(measured, target)
            errors.append(error)
            cells.append(
                {
                    "case": case["name"],
                    "measured": measured,
                    "target": target,
                    "relative_error": error,
                    "pass_10pct": error <= 0.10,
                }
            )
            runs[name] = {"artifacts": artifacts, "measurement": measurement}
        rows.append(
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
        "rows": rows,
        "cell_count": len(errors),
        "cells_within_10pct": sum(error <= 0.10 for error in errors),
        "mape": sum(errors) / len(errors),
        "max_error": max(errors),
        "pass_10pct": all(error <= 0.10 for error in errors),
    }
    integrity_checks = {
        "parents": all(parent_checks.values()),
        "manifest": manifest.get("output_count") == 24
        and manifest.get("pe_dependency_model") == "paper_static",
        "transform": len(transform_checks) == 24 and all(transform_checks.values()),
        "replay": replay.get("all_identical") is True
        and len(replay.get("comparisons") or []) == 24,
        "runs": len(run_checks) == 24 and all(run_checks.values()),
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
        "parents": {"artifacts": parent_reports, "checks": parent_checks},
        "compiler": {
            "manifest": qualify_file(manifest_path),
            "replay": qualify_file(replay_path),
            "transform_checks": transform_checks,
        },
        "runs": {"items": runs, "checks": run_checks, "pass": all(run_checks.values())},
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
            raise SystemExit("existing H53 result does not match a fresh audit")
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
