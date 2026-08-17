#!/usr/bin/env python3
"""Close Figure 20 with six failures and two execution-incomplete cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/fig20_matched_evidence_closure_v1.yaml"


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
    files = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    parents = {
        name: json.loads((PROJECT_ROOT / spec["path"]).read_text(encoding="utf-8"))
        for name, spec in config["frozen_inputs"].items()
        if name != "targets"
    }
    parent_checks = {
        name: report["hypothesis_status"] == config["frozen_inputs"][name]["required_status"]
        and report["audit_integrity"]
        is config["frozen_inputs"][name]["required_integrity"]
        for name, report in parents.items()
    }
    target_doc = yaml.safe_load(
        (PROJECT_ROOT / config["frozen_inputs"]["targets"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    targets = nested(target_doc, config["target_section"])

    cells = []
    for cell in parents["projections"]["cells"]:
        cells.append(
            {
                "paper_group": cell["paper_group"],
                "target_index": int(cell["target_index"]),
                "target_speedup": float(cell["speedup_target"]),
                "estimated_speedup": float(cell["speedup"]),
                "relative_error": float(cell["absolute_relative_error"]),
                "pass_10pct": bool(cell["pass_10pct"]),
                "status": "reproduced" if cell["pass_10pct"] else "numerical_failure",
                "evidence": "matched_projection_estimator",
            }
        )
    for spec in config["attention_cells"]:
        shape = spec["mlx_shape"]
        target_index = int(spec["target_index"])
        cells.append(
            {
                "paper_group": spec["paper_group"],
                "target_index": target_index,
                "target_speedup": float(targets[target_index]),
                "estimated_speedup": None,
                "relative_error": None,
                "pass_10pct": None,
                "status": "execution_incomplete",
                "evidence": "mlx_complete_xavier_folding_rejected",
                "mlx_cycles": int(
                    parents["mlx_attention"]["models"][shape][
                        "full_work_predicted_cycles"
                    ]
                ),
                "xavier_cycles": None,
                "xavier_holdout_gate": parents["xavier_attention"]["numerical"],
            }
        )
    cells.sort(key=lambda item: item["target_index"])
    statuses = {
        status: sum(item["status"] == status for item in cells)
        for status in ("reproduced", "numerical_failure", "execution_incomplete")
    }
    summary = {
        "total_cells": len(cells),
        "status_counts": statuses,
        "numerically_compared_cells": sum(item["relative_error"] is not None for item in cells),
        "cells_within_10pct": sum(item["pass_10pct"] is True for item in cells),
        "all_eight_reproduced_within_10pct": all(
            item["pass_10pct"] is True for item in cells
        ),
    }
    integrity_checks = {
        "frozen_files": all(item["pass"] for item in files.values()),
        "parents": all(parent_checks.values()),
        "all_indices": [item["target_index"] for item in cells] == list(range(8)),
        "six_projection_failures": statuses["numerical_failure"] == 6,
        "two_attention_incomplete": statuses["execution_incomplete"] == 2,
        "no_attention_speedup": all(
            item["estimated_speedup"] is None
            for item in cells
            if item["status"] == "execution_incomplete"
        ),
        "global_false": summary["all_eight_reproduced_within_10pct"] is False,
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
        "hypothesis_status": "supported" if integrity else "rejected",
        "audit_integrity": integrity,
        "figure20_reproduced_within_10pct": False,
        "frozen_inputs": files,
        "parent_checks": parent_checks,
        "cells": cells,
        "summary": summary,
        "integrity_checks": integrity_checks,
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
        keys = ("hypothesis_status", "audit_integrity", "cells", "summary", "integrity_checks")
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
