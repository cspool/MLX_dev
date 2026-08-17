#!/usr/bin/env python3
"""Overlay latest source-integrated evidence on the H37 completion certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/full_paper_completion_update_v1.yaml"


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
    reports = {
        name: json.loads((PROJECT_ROOT / spec["path"]).read_text(encoding="utf-8"))
        for name, spec in config["frozen_inputs"].items()
    }
    parent_checks = {
        name: (
            "required_status" not in spec
            or reports[name]["hypothesis_status"] == spec["required_status"]
        )
        and (
            "required_integrity" not in spec
            or reports[name]["audit_integrity"] is spec["required_integrity"]
        )
        for name, spec in config["frozen_inputs"].items()
    }
    baseline = reports["baseline"]
    rows = []
    changed = []
    for item in baseline["items"]:
        identifier = item["id"]
        old_status = item["status"]
        status = config["overrides"].get(identifier, old_status)
        row = {
            "id": identifier,
            "paper_item": item["paper_item"],
            "baseline_status": old_status,
            "updated_status": status,
            "changed": status != old_status,
            "baseline_note": item["note"],
        }
        if row["changed"]:
            row["latest_evidence"] = config["frozen_inputs"][identifier]["path"]
            changed.append(identifier)
        rows.append(row)
    raw_counts = Counter(row["updated_status"] for row in rows)
    counts = {
        status: raw_counts.get(status, 0)
        for status in config["expected_status_counts"]
    }
    latest_checks = {
        "fig20": reports["fig20"]["figure20_reproduced_within_10pct"] is False
        and reports["fig20"]["summary"]["status_counts"]
        == {"reproduced": 0, "numerical_failure": 6, "execution_incomplete": 2},
        "fig22": reports["fig22"]["hypothesis_status"] == "rejected"
        and reports["fig22"]["audit_integrity"] is True,
        "fig23": reports["fig23"]["hypothesis_status"] == "rejected"
        and reports["fig23"]["audit_integrity"] is True,
        "fig24": reports["fig24"]["hypothesis_status"] == "rejected"
        and reports["fig24"]["audit_integrity"] is True,
        "fig25": reports["fig25"]["hypothesis_status"] == "rejected"
        and reports["fig25"]["audit_integrity"] is True,
    }
    summary = {
        "inventory_item_count": len(rows),
        "status_counts": counts,
        "reproduced_within_10pct_count": counts["reproduced_within_10pct"],
        "not_fully_reproduced_count": len(rows) - counts["reproduced_within_10pct"],
        "all_paper_experiments_reproduced_within_10pct": False,
        "changed_rows": changed,
    }
    integrity_checks = {
        "frozen_files": all(item["pass"] for item in files.values()),
        "parents": all(parent_checks.values()),
        "baseline": baseline["audit_integrity"] is True
        and baseline["summary"]["inventory_item_count"] == 18,
        "eighteen_rows": len(rows) == 18,
        "five_rows_reconsidered": list(config["overrides"]) == [
            "fig20",
            "fig22",
            "fig23",
            "fig24",
            "fig25",
        ],
        "four_status_changes": changed == ["fig22", "fig23", "fig24", "fig25"],
        "latest_evidence": all(latest_checks.values()),
        "status_counts": counts == config["expected_status_counts"],
        "zero_passes": summary["reproduced_within_10pct_count"] == 0,
        "global_false": summary["all_paper_experiments_reproduced_within_10pct"] is False,
    }
    integrity = all(integrity_checks.values())
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "classification": config["classification"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_commit": git_commit(),
        "hypothesis_status": "supported" if integrity else "rejected",
        "audit_integrity": integrity,
        "frozen_inputs": files,
        "parent_checks": parent_checks,
        "latest_evidence_checks": latest_checks,
        "rows": rows,
        "summary": summary,
        "integrity_checks": integrity_checks,
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
        keys = ("hypothesis_status", "audit_integrity", "rows", "summary", "integrity_checks")
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
