#!/usr/bin/env python3
"""Build the dual-criterion Figures 18-25 completion certificate."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/active_simulator_trend_completion_v1.yaml"


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
            "primary_status": "identity_or_provenance_incomplete",
            "strict_status": "identity_or_provenance_incomplete",
            "evidence": "H131",
            "detail": {
                "missing_workload_fields": evidence["fig18"]["summary"]["missing_workload_fields"],
                "missing_provenance_fields": evidence["fig18"]["summary"][
                    "missing_provenance_fields"
                ],
            },
        },
        19: {
            "primary_status": "trend_audit_pending",
            "strict_status": "strict_numerical_rejection",
            "evidence": "H130",
            "detail": evidence["fig19"]["summary"],
        },
        20: {
            "primary_status": "trend_reproduced",
            "strict_status": "strict_numerical_rejection",
            "evidence": "H136",
            "detail": evidence["fig20"]["summary"],
        },
        21: {
            "primary_status": "execution_incomplete",
            "strict_status": "execution_incomplete",
            "evidence": "H96",
            "detail": evidence["fig21"]["summary"],
        },
        22: {
            "primary_status": "trend_audit_pending",
            "strict_status": "strict_numerical_rejection",
            "evidence": "H121",
            "detail": evidence["fig22"]["summary"],
        },
        23: {
            "primary_status": "identity_or_provenance_incomplete",
            "strict_status": "identity_or_provenance_incomplete",
            "evidence": "H122",
            "detail": evidence["fig23"]["summary"],
        },
        24: {
            "primary_status": "execution_incomplete",
            "strict_status": "execution_incomplete",
            "evidence": "H127",
            "detail": {
                **evidence["fig24"]["summary"],
                "missing_operator_families": ["fft_cmp", "swa"],
            },
        },
        25: {
            "primary_status": "trend_audit_pending",
            "strict_status": "strict_numerical_rejection",
            "evidence": "H115",
            "detail": evidence["fig25"]["summary"],
        },
    }
    row_checks = {
        "fig18": rows[18]["detail"]
        == {"missing_workload_fields": 12, "missing_provenance_fields": 6},
        "fig19": rows[19]["detail"]["points"] == 12 and rows[19]["detail"]["passing_points"] == 0,
        "fig20": rows[20]["detail"]["trend_full_figure_passes"] == 8
        and rows[20]["detail"]["strict_full_figure_passes"] == 1
        and rows[20]["detail"]["trend_figure20_reproduced"] is True
        and rows[20]["detail"]["strict_figure20_reproduced"] is False,
        "fig21": rows[21]["detail"]["status_counts"]
        == {"reproduced": 9, "numerical_failure": 6, "execution_incomplete": 5},
        "fig22": rows[22]["detail"]["points"] == 64 and rows[22]["detail"]["passing_points"] == 4,
        "fig23": rows[23]["detail"]["missing_identity_fields"] == 13
        and rows[23]["detail"]["exact_workload_identified"] is False,
        "fig24": rows[24]["detail"]["points"] == 21
        and rows[24]["detail"]["passing_points"] == 0
        and rows[24]["detail"]["missing_operator_families"] == ["fft_cmp", "swa"],
        "fig25": rows[25]["detail"]["total_points"] == 24
        and rows[25]["detail"]["passing_points"] == 2,
    }
    primary_counts = Counter(row["primary_status"] for row in rows.values())
    strict_counts = Counter(row["strict_status"] for row in rows.values())
    for status in config["primary_status_taxonomy"]:
        primary_counts.setdefault(status, 0)
    for status in config["strict_status_taxonomy"]:
        strict_counts.setdefault(status, 0)
    primary_reproduced = [
        figure for figure, row in rows.items() if row["primary_status"] == "trend_reproduced"
    ]
    strict_reproduced = [
        figure for figure, row in rows.items() if row["strict_status"] == "strict_reproduced"
    ]
    trend_pending = [
        figure for figure, row in rows.items() if row["primary_status"] == "trend_audit_pending"
    ]
    expected_primary = config["scope"]["expected_primary_reproduced"]
    expected_strict = config["scope"]["expected_strict_reproduced"]
    expected_pending = config["scope"]["trend_audit_pending"]
    no_invalid_promotion = (
        primary_reproduced == expected_primary
        and strict_reproduced == expected_strict
        and trend_pending == expected_pending
    )
    source_files = {
        name: qualify(PROJECT_ROOT / path) for name, path in config["source_layout"].items()
    }
    required = int(config["scope"]["required_figures"])
    primary_complete = len(primary_reproduced) == required
    strict_complete = len(strict_reproduced) == required
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()),
        sorted(rows) == config["scope"]["figures"] and len(rows) == 8,
        row_checks["fig18"],
        row_checks["fig19"],
        row_checks["fig20"],
        row_checks["fig21"],
        row_checks["fig22"] and row_checks["fig23"],
        row_checks["fig24"] and row_checks["fig25"],
        no_invalid_promotion and all(item["pass"] for item in source_files.values()),
        len(primary_reproduced) == 1
        and len(strict_reproduced) == 0
        and not primary_complete
        and not strict_complete,
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "scope": sorted(rows) == list(range(18, 26)),
        "rows": all(row_checks.values()),
        "primary_taxonomy": set(primary_counts) == set(config["primary_status_taxonomy"]),
        "strict_taxonomy": set(strict_counts) == set(config["strict_status_taxonomy"]),
        "no_invalid_promotion": no_invalid_promotion,
        "source": all(item["pass"] for item in source_files.values()),
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
        "new_paper_performance_targets_consumed": False,
        "paper_reproduction_claim": "certificate_primary_1_of_8_strict_0_of_8",
        "trend_policy": config["trend_policy"],
        "frozen_evidence": frozen,
        "rows": {str(key): value for key, value in rows.items()},
        "row_checks": row_checks,
        "primary_status_counts": dict(primary_counts),
        "strict_status_counts": dict(strict_counts),
        "no_invalid_promotion": no_invalid_promotion,
        "source_files": source_files,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "active_figures": len(rows),
            "primary_reproduced_full_figures": len(primary_reproduced),
            "strict_reproduced_full_figures": len(strict_reproduced),
            "required_full_figures": required,
            "primary_reproduced_figures": primary_reproduced,
            "strict_reproduced_figures": strict_reproduced,
            "trend_audit_pending_figures": trend_pending,
            "all_active_figures_trend_reproduced": primary_complete,
            "all_active_figures_reproduced_within_10pct": strict_complete,
            "primary_status_counts": dict(primary_counts),
            "strict_status_counts": dict(strict_counts),
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
            "primary_status_counts",
            "strict_status_counts",
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
