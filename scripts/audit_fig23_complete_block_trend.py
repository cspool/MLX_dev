#!/usr/bin/env python3
"""Join H141 complete-block scaling to Figure 23 qualitative targets."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/fig23_complete_block_trend_v1.yaml"


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
        name: parent["hypothesis_status"] == spec["required_status"]
        and parent["audit_integrity"] is spec["required_integrity"]
        for name, parent in evidence.items()
        for spec in [config["frozen_inputs"][name]]
    }
    h141 = evidence["h141"]
    targets = evidence["h65_targets"]
    h137 = evidence["h137"]
    minimum_clear_speedup = float(config["acceptance"]["minimum_clear_speedup"])
    policy_check = math.isclose(
        float(h137["trend_policy"]["minimum_clear_speedup"]),
        minimum_clear_speedup,
        rel_tol=0.0,
        abs_tol=0.0,
    )
    sequences = config["mapping"]["sequence_lengths"]
    windows = config["mapping"]["active_windows"]
    series_names = config["mapping"]["speedup_series"]
    target_rows = {
        series: {int(point["sequence_length"]): point for point in targets["points"][series]}
        for series in series_names
    }
    target_coverage = (
        set(targets["points"]) == set(series_names)
        and all(sorted(rows) == sequences for rows in target_rows.values())
        and sum(len(rows) for rows in target_rows.values()) == 15
        and all(
            math.isfinite(float(point["target"])) and float(point["target"]) > 0
            for rows in target_rows.values()
            for point in rows.values()
        )
    )
    expected_h141_groups = {
        f"N{sequence}-w{window}" for window in windows for sequence in sequences
    }
    h141_coverage = (
        set(h141["speedups"]) == expected_h141_groups
        and all(set(values) == set(series_names) for values in h141["speedups"].values())
        and h141["summary"]["figure23_target_join_eligible"] is True
    )
    strict_limit = float(config["acceptance"]["strict_relative_error_limit"])
    cells = []
    for window in windows:
        for series in series_names:
            for sequence in sequences:
                target = float(target_rows[series][sequence]["target"])
                prediction = float(h141["speedups"][f"N{sequence}-w{window}"][series])
                relative_error = abs(prediction - target) / target
                direction_match = target > 1.0 and prediction > 1.0
                clear_improvement = prediction >= minimum_clear_speedup
                cells.append(
                    {
                        "active_window": window,
                        "series": series,
                        "sequence_length": sequence,
                        "target_speedup": target,
                        "predicted_speedup": prediction,
                        "direction_match": direction_match,
                        "clear_improvement": clear_improvement,
                        "trend_pass": direction_match and clear_improvement,
                        "relative_error": relative_error,
                        "pass_10pct": relative_error <= strict_limit,
                        "prediction_provenance": f"H141.speedups.N{sequence}-w{window}.{series}",
                        "target_provenance": f"H65.points.{series}.N{sequence}",
                    }
                )
    finite_predictions = len(cells) == 30 and all(
        math.isfinite(cell["predicted_speedup"])
        and cell["predicted_speedup"] > 0
        and math.isfinite(cell["relative_error"])
        and cell["relative_error"] >= 0
        for cell in cells
    )
    direction_passes = sum(cell["direction_match"] for cell in cells)
    clear_passes = sum(cell["clear_improvement"] for cell in cells)
    trend_passes = sum(cell["trend_pass"] for cell in cells)
    strict_passes = sum(cell["pass_10pct"] for cell in cells)
    required_trend_passes = int(config["acceptance"]["required_trend_passes"])
    primary_figure_pass = trend_passes == required_trend_passes
    strict_figure_pass = strict_passes == len(cells)
    window_counts = {
        str(window): {
            "trend_passes": sum(
                cell["trend_pass"] for cell in cells if cell["active_window"] == window
            ),
            "strict_passes": sum(
                cell["pass_10pct"] for cell in cells if cell["active_window"] == window
            ),
        }
        for window in windows
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path) for name, path in config["source_layout"].items()
    }
    source_text = "\n".join(
        (PROJECT_ROOT / path).read_text(errors="replace")
        for path in config["source_layout"].values()
    )
    forbidden = (
        "prediction" + " *",
        "prediction" + " +",
        "fit" + "_affine",
        "target" + "_factor",
        "exact_author" + "_schedule_reproduced",
    )
    source_check = not any(token in source_text for token in forbidden)
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        policy_check,
        target_coverage,
        h141_coverage,
        finite_predictions,
        direction_passes == required_trend_passes,
        clear_passes == required_trend_passes,
        trend_passes == required_trend_passes
        and all(item["trend_passes"] == 15 for item in window_counts.values()),
        source_check
        and all(item["pass"] for item in source_files.values())
        and h141["surrogate_identity_claim"]
        == "representative_complete_block_not_exact_author_schedule",
        primary_figure_pass and (3 if primary_figure_pass else 2) == 3,
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "policy": policy_check,
        "target_coverage": target_coverage,
        "h141_coverage": h141_coverage,
        "finite": finite_predictions,
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
        "paper_reproduction_claim": (
            "figure23_trend_complete_representative_block_strict_false"
            if supported and not strict_figure_pass
            else "figure23_trend_complete_strict_complete"
            if supported
            else "figure23_trend_rejected"
        ),
        "surrogate_identity_claim": h141["surrogate_identity_claim"],
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "policy_check": policy_check,
        "target_coverage": target_coverage,
        "h141_coverage": h141_coverage,
        "cells": cells,
        "window_counts": window_counts,
        "source_files": source_files,
        "source_check": source_check,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "cells": len(cells),
            "direction_passes": direction_passes,
            "clear_improvement_passes": clear_passes,
            "trend_passes": trend_passes,
            "strict_passes": strict_passes,
            "minimum_predicted_speedup": min(cell["predicted_speedup"] for cell in cells),
            "maximum_predicted_speedup": max(cell["predicted_speedup"] for cell in cells),
            "strict_mape": sum(cell["relative_error"] for cell in cells) / len(cells),
            "strict_max_relative_error": max(cell["relative_error"] for cell in cells),
            "figure23_trend_reproduced": supported,
            "figure23_strict_reproduced": strict_figure_pass,
            "active_simulator_figures_reproduced": 3 if supported else 2,
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
            "window_counts",
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
