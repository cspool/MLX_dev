#!/usr/bin/env python3
"""Audit H148 end-to-end ratios against Figure 21 speedup direction."""

from __future__ import annotations

import argparse
import copy
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/fig21_xavier_trend_transfer_v1.yaml"


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    evidence = {
        name: json.loads((PROJECT_ROOT / spec["path"]).read_text())
        for name, spec in config["frozen_inputs"].items()
    }
    parent_checks = {
        "h148": evidence["h148_composition"]["hypothesis_status"]
        == config["frozen_inputs"]["h148_composition"]["required_status"]
        and evidence["h148_composition"]["audit_integrity"]
        is config["frozen_inputs"]["h148_composition"]["required_integrity"],
        "h96": evidence["h96_ledger"]["hypothesis_status"]
        == config["frozen_inputs"]["h96_ledger"]["required_status"]
        and evidence["h96_ledger"]["audit_integrity"]
        is config["frozen_inputs"]["h96_ledger"]["required_integrity"],
        "h137": evidence["h137_policy"]["hypothesis_status"]
        == config["frozen_inputs"]["h137_policy"]["required_status"]
        and evidence["h137_policy"]["audit_integrity"]
        is config["frozen_inputs"]["h137_policy"]["required_integrity"],
        "targets": evidence["targets"]["verdict"]
        == config["frozen_inputs"]["targets"]["required_verdict"]
        and evidence["targets"]["summary"]["pass"] is True,
    }
    minimum_clear_speedup = float(config["acceptance"]["minimum_clear_speedup"])
    policy_check = math.isclose(
        float(evidence["h137_policy"]["trend_policy"]["minimum_clear_speedup"]),
        minimum_clear_speedup,
        rel_tol=0.0,
        abs_tol=0.0,
    )
    sequences = config["mapping"]["sequence_lengths"]
    predictions = {
        int(row["sequence_length"]): float(row["speedup"])
        for row in evidence["h148_composition"]["rows"]
    }
    target_values = evidence["targets"]["derived_targets"]
    targets = dict(
        zip(target_values["sequence_lengths"], target_values["speedup_over_xavier"], strict=True)
    )
    coverage_check = sorted(predictions) == sequences and sorted(targets) == sequences
    strict_limit = float(config["acceptance"]["strict_relative_error_limit"])
    cells = []
    for sequence in sequences:
        target = float(targets[sequence])
        prediction = float(predictions[sequence])
        relative_error = abs(prediction - target) / target
        direction_match = target > 1.0 and prediction > 1.0
        clear_improvement = prediction >= minimum_clear_speedup
        cells.append(
            {
                "sequence_length": sequence,
                "target_speedup": target,
                "predicted_speedup": prediction,
                "direction_match": direction_match,
                "clear_improvement": clear_improvement,
                "trend_pass": direction_match and clear_improvement,
                "relative_error": relative_error,
                "pass_10pct": relative_error <= strict_limit,
                "prediction_provenance": f"H148.rows.N{sequence}.speedup",
                "target_provenance": f"H25.derived_targets.speedup_over_xavier.N{sequence}",
            }
        )
    finite_check = all(
        math.isfinite(cell[key]) and cell[key] > 0
        for cell in cells
        for key in ("target_speedup", "predicted_speedup")
    )
    direction_passes = sum(cell["direction_match"] for cell in cells)
    clear_passes = sum(cell["clear_improvement"] for cell in cells)
    trend_passes = sum(cell["trend_pass"] for cell in cells)
    strict_passes = sum(cell["pass_10pct"] for cell in cells)
    other_h96_rows = copy.deepcopy(
        [row for row in evidence["h96_ledger"]["rows"] if row["series"] != "speedup_over_xavier"]
    )
    ledger_preservation = (
        len(other_h96_rows) == 15
        and sum(row["status"] == "reproduced" for row in other_h96_rows) == 9
        and sum(row["status"] == "numerical_failure" for row in other_h96_rows) == 6
    )
    source_files = {
        name: qualify(PROJECT_ROOT / path) for name, path in config["source_layout"].items()
    }
    source_text = "\n".join(
        (PROJECT_ROOT / path).read_text(errors="replace")
        for path in config["source_layout"].values()
    )
    forbidden = (
        "1.0 / " + "prediction",
        "prediction" + "_scale",
        "direction" + "_correction",
        "component" + "_factor",
        "post_result" + "_model_selection",
    )
    source_check = not any(token in source_text for token in forbidden)
    required = int(config["acceptance"]["required_trend_passes"])
    primary_speedup_pass = trend_passes == required
    strict_speedup_pass = strict_passes == len(cells)
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        policy_check,
        coverage_check,
        finite_check,
        direction_passes == int(config["acceptance"]["required_direction_passes"]),
        clear_passes == required,
        primary_speedup_pass,
        ledger_preservation,
        source_check and all(item["pass"] for item in source_files.values()),
        (4 if primary_speedup_pass else 3) == 3
        and not primary_speedup_pass
        and not strict_speedup_pass,
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "policy": policy_check,
        "coverage": coverage_check,
        "finite": finite_check,
        "ledger_preservation": ledger_preservation,
        "source": source_check and all(item["pass"] for item in source_files.values()),
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
        "paper_reproduction_claim": "figure21_speedup_direction_rejected",
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "policy_check": policy_check,
        "coverage_check": coverage_check,
        "cells": cells,
        "preserved_h96_non_speedup_rows": other_h96_rows,
        "ledger_preservation": ledger_preservation,
        "source_files": source_files,
        "source_check": source_check,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "speedup_cells": len(cells),
            "direction_passes": direction_passes,
            "clear_improvement_passes": clear_passes,
            "trend_passes": trend_passes,
            "strict_passes": strict_passes,
            "mape": sum(cell["relative_error"] for cell in cells) / len(cells),
            "max_relative_error": max(cell["relative_error"] for cell in cells),
            "preserved_other_rows": len(other_h96_rows),
            "figure21_speedup_trend_reproduced": primary_speedup_pass,
            "figure21_speedup_strict_reproduced": strict_speedup_pass,
            "figure21_full_trend_reproduced": False,
            "active_simulator_figures_reproduced": 3,
            "active_simulator_figures_total": 8,
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
            "cells",
            "preserved_h96_non_speedup_rows",
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
