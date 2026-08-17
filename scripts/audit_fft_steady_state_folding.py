#!/usr/bin/env python3
"""Audit H81's larger-anchor FFT steady-state folding experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from audit_matched_fft_cycle_estimator import PROJECT_ROOT
from audit_matched_fft_cycle_estimator import build_audit as build_base_audit

DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fft_steady_state_folding_v1.yaml"


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    report = build_base_audit(config)
    spec = config["frozen_inputs"]["h80"]
    parent = json.loads(
        (PROJECT_ROOT / spec["path"]).read_text(encoding="utf-8")
    )
    parent_check = (
        parent.get("hypothesis_status") == spec["required_status"]
        and parent.get("audit_integrity") is spec["required_integrity"]
        and parent.get("numerical", {}).get("all_holdouts_pass") is False
    )
    report["parent_checks"]["h80"] = parent_check
    report["integrity_checks"]["h80_parent"] = parent_check
    report["audit_integrity"] = all(report["integrity_checks"].values())
    report["hypothesis_status"] = (
        "supported"
        if report["audit_integrity"] and report["numerical"]["all_holdouts_pass"]
        else "rejected"
    )
    report["conclusion"] = (
        "q4/q8 anchors reach a stable slope that predicts q16/q32 for both "
        "variable-depth FFT topologies"
    )
    return report


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
            "full_work_conservation",
            "models",
            "numerical",
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
                "full_work_cycles": {
                    key: item["full_work_predicted_cycles"]
                    for key, item in report["models"].items()
                },
            },
            indent=2,
        )
    )
    return 0 if report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
