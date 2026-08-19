#!/usr/bin/env python3
"""Audit H184 trace-corrected Figure23 simulator executions."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig23_trace_corrected_v1.yaml"


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    documents = {
        name: json.loads((PROJECT_ROOT / spec["path"]).read_text())
        for name, spec in config["frozen_inputs"].items()
    }
    parent_checks = {
        name: document["hypothesis_status"] == spec["required_status"]
        and document["audit_integrity"] is spec["required_integrity"]
        for name, document in documents.items()
        for spec in [config["frozen_inputs"][name]]
        if "required_status" in spec
    }
    compile_path = PROJECT_ROOT / config["compile_manifest"]
    run_path = PROJECT_ROOT / config["run_manifest"]
    compile_file = qualify(compile_path)
    run_file = qualify(run_path)
    compiled = json.loads(compile_path.read_text())
    executed = json.loads(run_path.read_text())
    raw = documents["raw_simulator"]
    target_cells = documents["current_audit"]["cells"]
    compile_checks = {
        "count": len(compiled["outputs"]) == int(config["execution"]["expected_configs"]),
        "checks": all(compiled["checks"].values()),
        "parameters": compiled["parameters"] == documents["selected_model"]["figure23"][
            "parameters"
        ],
        "blocks": all(
            item["source_blocks_sha256"] == item["compiled_blocks_sha256"]
            for item in compiled["outputs"].values()
        ),
    }
    execution_checks = {
        "configs": len(executed["runs"]) == int(config["execution"]["expected_configs"]),
        "runs": sum(len(builds) for builds in executed["runs"].values())
        == int(config["execution"]["expected_runs"]),
        "checks": all(executed["checks"].values()),
        "builds": all(
            set(builds) == set(config["execution"]["builds"])
            for builds in executed["runs"].values()
        ),
    }
    raw_cycle_checks: dict[str, bool] = {}
    formula_checks: dict[str, bool] = {}
    work_checks: dict[str, bool] = {}
    for key, item in compiled["outputs"].items():
        summary = executed["runs"][key]["opt"]["summary"]
        metadata = item["metadata"]
        group = f"N{metadata['sequence_length']}-w{metadata['active_window']}"
        hardware = metadata["hardware_name"]
        raw_cycle_checks[key] = (
            summary["raw_cycles"] == raw["cycles"][group][hardware] == item["raw_cycles"]
        )
        formula_checks[key] = summary["cycles"] == (
            summary["raw_cycles"]
            - summary["latency_service"]["startup_credit_cycles"]
            + summary["latency_service"]["congestion_cycles"]
        ) == item["expected_cycles"]
        work_checks[key] = (
            summary["instructions_issued"]
            == summary["instructions_completed"]
            == metadata["work"]["instruction_instances"]
            and item["source_blocks_sha256"] == item["compiled_blocks_sha256"]
        )
    cells: list[dict[str, Any]] = []
    limit = float(config["acceptance"]["maximum_relative_error"])
    for target in target_cells:
        window = int(target["active_window"])
        sequence = int(target["sequence_length"])
        series = target["series"]
        baseline_key = f"N{sequence}-w{window}-baseline"
        series_key = f"N{sequence}-w{window}-{series}"
        baseline_cycles = float(executed["runs"][baseline_key]["opt"]["summary"]["cycles"])
        series_cycles = float(executed["runs"][series_key]["opt"]["summary"]["cycles"])
        prediction = baseline_cycles / series_cycles
        target_speedup = float(target["target_speedup"])
        error = abs(prediction - target_speedup) / target_speedup
        cells.append(
            {
                "active_window": window,
                "sequence_length": sequence,
                "series": series,
                "baseline_cycles": baseline_cycles,
                "series_cycles": series_cycles,
                "predicted_speedup": prediction,
                "target_speedup": target_speedup,
                "relative_error": error,
                "pass_15pct": error <= limit,
                "direction_match": prediction > 1.0 and target_speedup > 1.0,
                "is_holdout": sequence in (1024, 4096),
            }
        )
    holdouts = [cell for cell in cells if cell["is_holdout"]]
    numerical_checks = {
        "points": len(cells) == int(config["acceptance"]["required_points"]),
        "passes": sum(cell["pass_15pct"] for cell in cells)
        == int(config["acceptance"]["required_passing_points"]),
        "directions": sum(cell["direction_match"] for cell in cells)
        == int(config["acceptance"]["required_direction_matches"]),
        "holdouts": len(holdouts) == 12
        and max(cell["relative_error"] for cell in holdouts)
        <= float(config["acceptance"]["maximum_holdout_relative_error"]),
        "finite": all(
            math.isfinite(cell["predicted_speedup"])
            and math.isfinite(cell["relative_error"])
            and cell["predicted_speedup"] > 0
            for cell in cells
        ),
    }
    parameter_names = tuple(compiled["parameters"])
    parameter_checks = {
        "count": len(parameter_names) == 4,
        "not_point_keyed": not any(
            token in name
            for name in parameter_names
            for token in ("512", "1024", "2048", "4096", "8192", "target_index")
        ),
        "provenance": all(
            builds["opt"]["summary"]["latency_service"]["target_informed"] is True
            and builds["opt"]["summary"]["latency_service"]["provenance"]
            == "H183.figure23.parameters+H182.RTX4090.trace_features"
            for builds in executed["runs"].values()
        ),
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        compile_file["pass"] and all(compile_checks.values()),
        run_file["pass"] and all(execution_checks.values()),
        all(raw_cycle_checks.values()),
        all(work_checks.values()) and all(formula_checks.values()),
        all(executed["checks"].values()),
        numerical_checks["points"]
        and numerical_checks["passes"]
        and numerical_checks["directions"],
        numerical_checks["holdouts"] and numerical_checks["finite"],
        all(parameter_checks.values()),
        all(item["pass"] for item in source_files.values())
        and config["acceptance"]["independent_validation_claimed"] is False,
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": len(parent_checks) == 4,
        "compile": len(compile_checks) == 4,
        "execution": len(execution_checks) == 4,
        "raw_cycles": len(raw_cycle_checks) == 40,
        "formula": len(formula_checks) == 40,
        "work": len(work_checks) == 40,
        "numerical": len(numerical_checks) == 5,
        "parameters": len(parameter_checks) == 3,
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
        "paper_reproduction_claim": "figure23_trace_corrected_within_15pct_not_independent",
        "independent_validation_claimed": False,
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "compile_manifest": compile_file,
        "run_manifest": run_file,
        "compile_checks": compile_checks,
        "execution_checks": execution_checks,
        "raw_cycle_checks": raw_cycle_checks,
        "formula_checks": formula_checks,
        "work_checks": work_checks,
        "numerical_checks": numerical_checks,
        "parameter_checks": parameter_checks,
        "parameters": compiled["parameters"],
        "cells": cells,
        "source_files": source_files,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "configs": len(compiled["outputs"]),
            "executions": sum(len(builds) for builds in executed["runs"].values()),
            "raw_cycle_matches": sum(raw_cycle_checks.values()),
            "work_matches": sum(work_checks.values()),
            "points": len(cells),
            "passing_points": sum(cell["pass_15pct"] for cell in cells),
            "mape": sum(cell["relative_error"] for cell in cells) / len(cells),
            "max_relative_error": max(cell["relative_error"] for cell in cells),
            "holdout_points": len(holdouts),
            "holdout_max_relative_error": max(
                cell["relative_error"] for cell in holdouts
            ),
            "direction_matches": sum(cell["direction_match"] for cell in cells),
            "parameter_count": len(compiled["parameters"]),
            "figure23_numerically_reproduced_within_15pct": supported,
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
            "compile_checks",
            "execution_checks",
            "raw_cycle_checks",
            "formula_checks",
            "work_checks",
            "numerical_checks",
            "parameter_checks",
            "parameters",
            "cells",
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
