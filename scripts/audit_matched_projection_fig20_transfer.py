#!/usr/bin/env python3
"""Audit H77 matched projection estimates against six Figure 20 bars."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from mlxsim.repeat_folding import relative_error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/analysis/matched_projection_fig20_transfer_v1.yaml"
)


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


def nested(document: dict[str, Any], dotted: str) -> Any:
    value: Any = document
    for key in dotted.split("."):
        value = value[key]
    return value


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
    parent_spec = config["parent"]
    target_spec = config["targets"]
    parent_file = qualify(PROJECT_ROOT / parent_spec["path"], parent_spec)
    target_file = qualify(PROJECT_ROOT / target_spec["path"], target_spec)
    parent = json.loads(
        (PROJECT_ROOT / parent_spec["path"]).read_text(encoding="utf-8")
    )
    target_doc = yaml.safe_load(
        (PROJECT_ROOT / target_spec["path"]).read_text(encoding="utf-8")
    )
    targets = nested(target_doc, target_spec["section"])
    threshold = float(config["threshold_relative_error"])

    cells = []
    errors = []
    for mapping in config["cells"]:
        estimate = parent["estimates"][mapping["estimate"]]
        actual = float(estimate["sparse_cuda_speedup"])
        target = float(targets[int(mapping["target_index"])])
        error = relative_error(actual, target)
        errors.append(error)
        cells.append(
            {
                "paper_group": mapping["paper_group"],
                "estimate_key": mapping["estimate"],
                "target_index": int(mapping["target_index"]),
                "speedup": actual,
                "speedup_target": target,
                "absolute_relative_error": error,
                "pass_10pct": error <= threshold,
            }
        )

    covered_indices = [int(item["target_index"]) for item in config["cells"]]
    excluded_indices = [
        int(item["target_index"]) for item in config["excluded_cells"]
    ]
    summary = {
        "covered_cells": len(cells),
        "cells_within_10pct": sum(item["pass_10pct"] for item in cells),
        "mape": sum(errors) / len(errors),
        "max_error": max(errors),
        "pass_10pct": all(item["pass_10pct"] for item in cells),
    }
    parent_checks = {
        "status": parent.get("hypothesis_status") == parent_spec["required_status"],
        "integrity": parent.get("audit_integrity")
        is parent_spec["required_integrity"],
        "target_free": parent.get("integrity_checks", {}).get("targets_consumed")
        is False,
        "six_projection_estimates": len(parent.get("estimates", {})) == 6,
    }
    integrity_checks = {
        "parent_file": parent_file["pass"],
        "target_file": target_file["pass"],
        "parent": all(parent_checks.values()),
        "mapping": covered_indices == [0, 2, 3, 4, 6, 7],
        "excluded_attention": excluded_indices == [1, 5],
        "all_eight_indices_accounted": sorted(covered_indices + excluded_indices)
        == list(range(8)),
        "targets_audit_only": target_spec["execution_visibility"] == "audit_only",
        "threshold": threshold == 0.10,
    }
    integrity = all(integrity_checks.values())
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "classification": config["classification"],
        "validation_eligible": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_commit": git_commit(),
        "hypothesis_status": (
            "supported" if integrity and summary["pass_10pct"] else "rejected"
        ),
        "audit_integrity": integrity,
        "frozen_inputs": {"parent": parent_file, "targets": target_file},
        "parent_checks": parent_checks,
        "cells": cells,
        "excluded_cells": config["excluded_cells"],
        "summary": summary,
        "integrity_checks": integrity_checks,
        "paper_target_values_consumed_during_estimation": False,
        "paper_target_values_consumed_during_audit": True,
        "stopping_rule_applied": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
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
            "cells",
            "excluded_cells",
            "summary",
            "integrity_checks",
        )
        matches = all(existing.get(key) == report.get(key) for key in keys)
        print(json.dumps({"existing_matches": matches, **report}, indent=2))
        return 0 if matches else 1
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0 if report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
