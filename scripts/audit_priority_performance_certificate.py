#!/usr/bin/env python3
"""Audit H179 prioritized Figure 24/23/19/20 exploration completion."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/analysis/priority_performance_certificate_v1.yaml"
)


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    parents = {
        name: json.loads((PROJECT_ROOT / spec["path"]).read_text())
        for name, spec in config["frozen_inputs"].items()
    }
    parent_checks = {
        name: parent["hypothesis_status"]
        == config["frozen_inputs"][name]["required_status"]
        and parent["audit_integrity"]
        is config["frozen_inputs"][name]["required_integrity"]
        for name, parent in parents.items()
    }
    fig24 = parents["fig24"]["summary"]
    fig23 = parents["fig23"]["summary"]
    fig19 = parents["fig19"]["summary"]
    fig20 = parents["fig20"]["summary"]
    fig24_checks = {
        "rows": fig24["figure24_rows"] == int(config["completion"]["fig24_native_rows"]),
        "services": fig24["service_models"]
        == fig24["service_holdout_passes"]
        == int(config["completion"]["fig24_native_services"]),
        "gpu": fig24["native_gpu"] == "NVIDIA GeForce RTX 4090",
        "complete": fig24["figure24_rtx4090_complete"] is True,
        "target_free": parents["fig24"]["paper_performance_targets_consumed"]
        is False,
    }
    fig23_checks = {
        "trend": fig23["trend_passes"]
        == int(config["completion"]["fig23_trend_cells"]),
        "complete": fig23["figure23_trend_reproduced"] is True,
        "strict_retained": fig23["strict_passes"] == 23
        and fig23["figure23_strict_reproduced"] is False,
    }
    fig19_checks = {
        "curves": fig19["curve_passes"]
        == fig19["curve_total"]
        == int(config["completion"]["fig19_curves"]),
        "comparisons": fig19["comparison_passes"]
        == fig19["comparison_total"]
        == int(config["completion"]["fig19_comparisons"]),
        "trend": fig19["figure19_trend_reproduced"] is True,
        "strict_retained": fig19["figure19_strict_reproduced"] is False,
    }
    fig20_checks = {
        "cells": fig20["trend_full_figure_passes"]
        == int(config["completion"]["fig20_trend_cells"]),
        "trend": fig20["trend_figure20_reproduced"] is True,
        "strict_retained": fig20["strict_figure20_reproduced"] is False,
    }
    reference_checks = {
        "fig22_rejected": parents["fig22_reference"]["hypothesis_status"]
        == "rejected"
        and parents["fig22_reference"]["diagnosis"]
        == "resource_schema_insufficient_schedule_or_workload_next",
        "fig25_rejected": parents["fig25_reference"]["hypothesis_status"]
        == "rejected"
        and parents["fig25_reference"]["summary"]["figure25_trend_reproduced"]
        is False,
        "scope": config["completion"]["reference_only_figures"] == [22, 25],
    }
    scope_checks = {
        "completed": fig24_checks["complete"]
        and fig23_checks["complete"]
        and fig19_checks["trend"]
        and fig20_checks["trend"],
        "pending": int(config["completion"]["final_pending_figure"]) == 18,
        "no_strict_promotion": fig23_checks["strict_retained"]
        and fig19_checks["strict_retained"]
        and fig20_checks["strict_retained"],
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        fig24_checks["rows"] and fig24_checks["services"],
        fig24_checks["gpu"] and fig24_checks["complete"] and fig24_checks["target_free"],
        fig23_checks["trend"] and fig23_checks["complete"],
        fig23_checks["strict_retained"],
        fig19_checks["curves"] and fig19_checks["comparisons"] and fig19_checks["trend"],
        fig20_checks["cells"] and fig20_checks["trend"],
        all(reference_checks.values()),
        all(scope_checks.values()),
        all(item["pass"] for item in source_files.values()),
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "fig24": len(fig24_checks) == 5,
        "fig23": len(fig23_checks) == 3,
        "fig19": len(fig19_checks) == 4,
        "fig20": len(fig20_checks) == 3,
        "references": len(reference_checks) == 3,
        "scope": len(scope_checks) == 3,
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
        "paper_performance_targets_consumed": True,
        "paper_reproduction_claim": "prioritized_exploration_not_strict_full_figures",
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "fig24_checks": fig24_checks,
        "fig23_checks": fig23_checks,
        "fig19_checks": fig19_checks,
        "fig20_checks": fig20_checks,
        "reference_checks": reference_checks,
        "scope_checks": scope_checks,
        "source_files": source_files,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "fig24_native_rows": fig24["figure24_rows"],
            "fig24_native_services": fig24["service_models"],
            "fig23_trend_cells": fig23["trend_passes"],
            "fig19_curve_comparisons": fig19["curve_passes"]
            + fig19["comparison_passes"],
            "fig20_trend_cells": fig20["trend_full_figure_passes"],
            "reference_only_figures": [22, 25],
            "completed_priority_figures": [24, 23, 19, 20],
            "final_pending_figure": 18,
            "priority_stage_complete": supported,
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
            "fig24_checks",
            "fig23_checks",
            "fig19_checks",
            "fig20_checks",
            "reference_checks",
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
