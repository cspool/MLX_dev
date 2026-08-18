#!/usr/bin/env python3
"""Join H135 Attention speedups and refresh the dual-criterion Figure 20 ledger."""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/fig20_attention_completion_v1.yaml"


def trend_outcome(
    target_speedup: float, estimated_speedup: float, minimum_clear_speedup: float
) -> dict[str, bool]:
    """Evaluate the frozen user-directed speedup-trend criterion."""
    direction_match = target_speedup > 1.0 and estimated_speedup > 1.0
    clear_improvement = estimated_speedup >= minimum_clear_speedup
    return {
        "trend_direction_match": direction_match,
        "clear_improvement": clear_improvement,
        "trend_pass": direction_match and clear_improvement,
    }


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    h135 = json.loads((PROJECT_ROOT / config["frozen_inputs"]["h135"]["path"]).read_text())
    h88 = json.loads((PROJECT_ROOT / config["frozen_inputs"]["h88"]["path"]).read_text())
    parent_checks = {
        name: parent["hypothesis_status"] == spec["required_status"]
        and parent["audit_integrity"] is spec["required_integrity"]
        for name, parent in (("h135", h135), ("h88", h88))
        for spec in [config["frozen_inputs"][name]]
    }
    old_cells = {int(cell["target_index"]): cell for cell in h88["cells"]}
    old_projection_cells = {
        index: copy.deepcopy(cell)
        for index, cell in old_cells.items()
        if cell["status"] != "execution_incomplete"
    }
    old_incomplete = {
        index: cell for index, cell in old_cells.items() if cell["status"] == "execution_incomplete"
    }
    old_shape_checks = {
        "eight": len(old_cells) == 8,
        "projections": len(old_projection_cells) == 6
        and all(cell["status"] == "numerical_failure" for cell in old_projection_cells.values()),
        "attention": set(old_incomplete) == {1, 5}
        and all(cell["paper_group"].startswith("Attn_") for cell in old_incomplete.values()),
    }
    limit = float(config["acceptance"]["relative_error_limit"])
    attention_cells: dict[int, Any] = {}
    mapping_checks: dict[str, bool] = {}
    for shape, mapping in config["attention_mapping"].items():
        index = int(mapping["target_index"])
        old = old_cells[index]
        prediction = float(h135["compositions"][shape]["speedup"])
        target = float(old["target_speedup"])
        error = abs(prediction - target) / target
        passed = error <= limit
        mapping_checks[shape] = (
            old["paper_group"] == mapping["paper_group"]
            and h135["compositions"][shape]["speedup_definition"]
            == "xavier_total_seconds_div_mlx_total_seconds"
        )
        attention_cells[index] = {
            "paper_group": old["paper_group"],
            "target_index": index,
            "target_speedup": target,
            "estimated_speedup": prediction,
            "relative_error": error,
            "pass_10pct": passed,
            "status": "reproduced" if passed else "numerical_failure",
            "evidence": "H135_complete_xavier_attention_composition",
            "xavier_total_cycles": h135["compositions"][shape]["xavier_total_cycles"],
            "mlx_cycles": h135["compositions"][shape]["mlx_cycles"],
        }
    refreshed = {**old_projection_cells, **attention_cells}
    projection_unchanged = all(
        refreshed[index] == old_projection_cells[index] for index in old_projection_cells
    )
    minimum_clear_speedup = float(config["acceptance"]["minimum_clear_speedup"])
    for cell in refreshed.values():
        cell.update(
            trend_outcome(
                float(cell["target_speedup"]),
                float(cell["estimated_speedup"]),
                minimum_clear_speedup,
            )
        )
        cell["trend_status"] = "trend_reproduced" if cell["trend_pass"] else "trend_failure"
    cells = [refreshed[index] for index in sorted(refreshed)]
    finite_checks = {
        str(index): math.isfinite(cell["estimated_speedup"])
        and cell["estimated_speedup"] > 0
        and math.isfinite(cell["target_speedup"])
        and cell["target_speedup"] > 0
        and math.isfinite(cell["relative_error"])
        and cell["relative_error"] >= 0
        for index, cell in attention_cells.items()
    }
    status_counts = Counter(cell["status"] for cell in cells)
    trend_status_counts = Counter(cell["trend_status"] for cell in cells)
    strict_attention_passes = sum(cell["pass_10pct"] for cell in attention_cells.values())
    strict_full_passes = sum(cell.get("pass_10pct") is True for cell in cells)
    trend_attention_passes = sum(cell["trend_pass"] for cell in attention_cells.values())
    trend_full_passes = sum(cell["trend_pass"] for cell in cells)
    strict_attention_pass = strict_attention_passes == int(
        config["acceptance"]["required_attention_passes"]
    )
    strict_figure_pass = strict_full_passes == int(
        config["acceptance"]["required_full_figure_passes"]
    )
    attention_trend_pass = trend_attention_passes == int(
        config["acceptance"]["required_attention_trend_passes"]
    )
    full_figure_trend_pass = trend_full_passes == int(
        config["acceptance"]["required_full_figure_trend_passes"]
    )
    source_files = {
        name: qualify(PROJECT_ROOT / path) for name, path in config["source_layout"].items()
    }
    source_text = "\n".join(
        (PROJECT_ROOT / path).read_text(errors="replace")
        for path in config["source_layout"].values()
    )
    forbidden = (
        "fit" + "_affine",
        "component" + "_factor",
        "clock" + "_factor",
        "prediction" + " *",
        "prediction" + " +",
    )
    source_checks = {
        "no_fit": not any(token in source_text for token in forbidden),
        "no_projection_change": projection_unchanged,
    }
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        all(old_shape_checks.values()),
        all(mapping_checks.values()),
        all(finite_checks.values()),
        attention_trend_pass,
        projection_unchanged,
        len(cells) == 8
        and set(refreshed) == set(range(8))
        and status_counts["execution_incomplete"] == 0,
        full_figure_trend_pass,
        all(source_checks.values()) and all(item["pass"] for item in source_files.values()),
        (1 if full_figure_trend_pass else 0) == (1 if trend_full_passes == 8 else 0),
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "old_shape": all(old_shape_checks.values()),
        "mapping": all(mapping_checks.values()),
        "finite": all(finite_checks.values()),
        "projection_unchanged": projection_unchanged,
        "ledger": len(cells) == 8 and set(refreshed) == set(range(8)),
        "source": all(source_checks.values())
        and all(item["pass"] for item in source_files.values()),
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
            "figure20_trend_complete_strict_complete"
            if supported and strict_figure_pass
            else "figure20_trend_complete_strict_false"
            if supported
            else "figure20_trend_rejected"
        ),
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "old_shape_checks": old_shape_checks,
        "mapping_checks": mapping_checks,
        "finite_checks": finite_checks,
        "projection_unchanged": projection_unchanged,
        "attention_cells": {str(key): value for key, value in attention_cells.items()},
        "cells": cells,
        "status_counts": dict(status_counts),
        "trend_status_counts": dict(trend_status_counts),
        "source_checks": source_checks,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "total_cells": len(cells),
            "attention_cells": len(attention_cells),
            "primary_completion_criterion": config["acceptance"]["primary_completion_criterion"],
            "minimum_clear_speedup": minimum_clear_speedup,
            "strict_attention_passes": strict_attention_passes,
            "strict_full_figure_passes": strict_full_passes,
            "trend_attention_passes": trend_attention_passes,
            "trend_full_figure_passes": trend_full_passes,
            "status_counts": dict(status_counts),
            "trend_status_counts": dict(trend_status_counts),
            "strict_attention_reproduced": strict_attention_pass,
            "strict_figure20_reproduced": strict_figure_pass,
            "trend_attention_reproduced": attention_trend_pass,
            "trend_figure20_reproduced": supported,
            "figure20_reproduced": supported,
            "acceptance_gates_passed": sum(acceptance_gates),
            "acceptance_gates_total": len(acceptance_gates),
            "active_simulator_figures_reproduced": 1 if supported else 0,
            "active_simulator_figures_total": 8,
        },
        "source_files": source_files,
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
            "attention_cells",
            "cells",
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
