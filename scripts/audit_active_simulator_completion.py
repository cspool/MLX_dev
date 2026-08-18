#!/usr/bin/env python3
"""Build the strict Figures 18-25 simulator-scope completion certificate."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/active_simulator_completion_v1.yaml"


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_evidence"].items()
    }
    evidence = {
        name: json.loads((PROJECT_ROOT / spec["path"]).read_text())
        for name, spec in config["frozen_evidence"].items()
    }
    rows = {
        18: {
            "status": "identity_or_provenance_incomplete",
            "evidence": "H131",
            "detail": {
                "missing_workload_fields": evidence["fig18"]["summary"][
                    "missing_workload_fields"
                ],
                "missing_provenance_fields": evidence["fig18"]["summary"][
                    "missing_provenance_fields"
                ],
            },
        },
        19: {
            "status": "numerical_rejection",
            "evidence": "H130",
            "detail": evidence["fig19"]["summary"],
        },
        20: {
            "status": "execution_incomplete",
            "evidence": "H88",
            "detail": evidence["fig20"]["summary"],
        },
        21: {
            "status": "execution_incomplete",
            "evidence": "H96",
            "detail": evidence["fig21"]["summary"],
        },
        22: {
            "status": "numerical_rejection",
            "evidence": "H121",
            "detail": evidence["fig22"]["summary"],
        },
        23: {
            "status": "identity_or_provenance_incomplete",
            "evidence": "H122",
            "detail": evidence["fig23"]["summary"],
        },
        24: {
            "status": "execution_incomplete",
            "evidence": "H127",
            "detail": {
                **evidence["fig24"]["summary"],
                "missing_operator_families": ["fft_cmp", "swa"],
            },
        },
        25: {
            "status": "numerical_rejection",
            "evidence": "H115",
            "detail": evidence["fig25"]["summary"],
        },
    }
    row_checks = {
        "fig18": rows[18]["detail"]
        == {"missing_workload_fields": 12, "missing_provenance_fields": 6},
        "fig19": rows[19]["detail"]["points"] == 12
        and rows[19]["detail"]["passing_points"] == 0,
        "fig20": rows[20]["detail"]["status_counts"]
        == {"reproduced": 0, "numerical_failure": 6, "execution_incomplete": 2},
        "fig21": rows[21]["detail"]["status_counts"]
        == {"reproduced": 9, "numerical_failure": 6, "execution_incomplete": 5}
        and rows[21]["detail"]["figure21_reproduced_within_10pct"] is False,
        "fig22": rows[22]["detail"]["points"] == 64
        and rows[22]["detail"]["passing_points"] == 4,
        "fig23": rows[23]["detail"]["missing_identity_fields"] == 13
        and rows[23]["detail"]["exact_workload_identified"] is False,
        "fig24": rows[24]["detail"]["points"] == 21
        and rows[24]["detail"]["passing_points"] == 0
        and rows[24]["detail"]["missing_operator_families"] == ["fft_cmp", "swa"],
        "fig25": rows[25]["detail"]["total_points"] == 24
        and rows[25]["detail"]["passing_points"] == 2,
    }
    status_counts = Counter(row["status"] for row in rows.values())
    for status in config["status_taxonomy"]:
        status_counts.setdefault(status, 0)
    reproduced = int(status_counts["reproduced"])
    all_complete = reproduced == int(config["scope"]["required_figures"])
    no_partial_promotion = all(
        not (
            figure == 21
            and row["detail"]["status_counts"]["reproduced"] == 9
            and row["status"] == "reproduced"
        )
        for figure, row in rows.items()
    ) and rows[24]["status"] != "reproduced"
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()),
        sorted(rows) == config["scope"]["figures"] and len(rows) == 8,
        row_checks["fig18"],
        row_checks["fig19"],
        row_checks["fig20"],
        row_checks["fig21"],
        row_checks["fig22"] and row_checks["fig23"],
        row_checks["fig24"] and row_checks["fig25"],
        no_partial_promotion,
        reproduced == 0 and all_complete is False,
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "scope": sorted(rows) == list(range(18, 26)),
        "rows": all(row_checks.values()),
        "taxonomy": set(status_counts) == set(config["status_taxonomy"]),
        "no_partial_promotion": no_partial_promotion,
        "acceptance_evaluated": len(acceptance_gates) == 10
        and all(isinstance(value, bool) for value in acceptance_gates),
    }
    integrity = all(integrity_checks.values())
    supported = integrity and all(acceptance_gates)
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "classification": config["classification"],
        "validation_eligible": config["validation_eligible"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_commit": git_commit(),
        "hypothesis_status": "supported" if supported else "rejected",
        "audit_integrity": integrity,
        "paper_performance_targets_consumed": False,
        "paper_reproduction_claim": "certificate_only_global_completion_false",
        "frozen_evidence": frozen,
        "rows": {str(key): value for key, value in rows.items()},
        "row_checks": row_checks,
        "status_counts": dict(status_counts),
        "no_partial_promotion": no_partial_promotion,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "active_figures": len(rows),
            "reproduced_full_figures": reproduced,
            "required_full_figures": int(config["scope"]["required_figures"]),
            "all_active_figures_reproduced_within_10pct": all_complete,
            "status_counts": dict(status_counts),
            "acceptance_gates_passed": sum(acceptance_gates),
            "acceptance_gates_total": len(acceptance_gates),
        },
        "integrity_checks": integrity_checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    report = build_audit(config)
    output = PROJECT_ROOT / config["result_path"]
    if args.verify_existing:
        existing = json.loads(output.read_text())
        keys = (
            "hypothesis_status",
            "audit_integrity",
            "rows",
            "status_counts",
            "acceptance_gates",
            "summary",
            "integrity_checks",
        )
        matches = all(
            json.dumps(existing.get(key), sort_keys=True)
            == json.dumps(report.get(key), sort_keys=True)
            for key in keys
        )
        print(json.dumps({"existing_matches": matches, **report}, indent=2))
        return 0 if matches else 1
    if args.preflight_only:
        print(json.dumps(report, indent=2))
        return 0 if report["audit_integrity"] and not output.exists() else 1
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["hypothesis_status"], **report["summary"]}, indent=2))
    return 0 if report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
