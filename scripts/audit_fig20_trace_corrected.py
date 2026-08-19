#!/usr/bin/env python3
"""Audit H186 trace-corrected Figure20 composition."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/fig20_trace_corrected_v1.yaml"


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    documents: dict[str, Any] = {}
    for name, spec in config["frozen_inputs"].items():
        path = PROJECT_ROOT / spec["path"]
        documents[name] = (
            yaml.safe_load(path.read_text())
            if path.suffix in {".yaml", ".yml"}
            else json.loads(path.read_text())
        )
    parent_checks = {
        name: document["hypothesis_status"] == spec["required_status"]
        and document["audit_integrity"] is spec["required_integrity"]
        for name, document in documents.items()
        for spec in [config["frozen_inputs"][name]]
        if "required_status" in spec
    }
    manifest_path = PROJECT_ROOT / config["composition_manifest"]
    manifest_file = qualify(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    composition = config["composition"]
    panels = list(composition["panels"])
    operators = list(composition["operators"])
    sequences = [int(value) for value in composition["sequence_lengths"]]
    legacy_rows = {
        row["case"]: row for row in documents["legacy_execution"]["raw"]
    }
    trace_values = {
        record["key"]: float(record["timing"]["median_ms"])
        for record in documents["rtx4090_trace"]["cases"]
    }
    raw_checks: dict[str, bool] = {}
    trace_checks: dict[str, bool] = {}
    target_section = documents["targets"]["fig20_xavier_kernels"]
    cells: list[dict[str, Any]] = []
    limit = float(config["acceptance"]["maximum_relative_error"])
    for row in manifest["rows"]:
        identity = f"{row['panel']}-{row['sequence_length']}-{row['operator']}"
        legacy = legacy_rows[row["legacy_case"]]
        raw_checks[identity] = (
            row["legacy_mlx_latency_us"] == legacy["mlx"]["latency_us"]
            and row["legacy_mlx_operations"] == legacy["mlx"]["operations"]
            and row["legacy_mlx_offchip_bytes"] == legacy["mlx"]["offchip_bytes"]
        )
        trace_checks[identity] = row["trace_median_ms"] == trace_values[row["trace_key"]]
        sequence_index = sequences.index(int(row["sequence_length"]))
        operator_index = operators.index(row["operator"])
        target_index = sequence_index * len(operators) + operator_index
        target = float(target_section[row["panel"]]["speedup"][target_index])
        prediction = float(row["speedup"])
        error = abs(prediction - target) / target
        cells.append(
            {
                "panel": row["panel"],
                "sequence_length": int(row["sequence_length"]),
                "operator": row["operator"],
                "service": row["service"],
                "prediction": prediction,
                "target": target,
                "relative_error": error,
                "pass_15pct": error <= limit,
                "direction_match": prediction >= 1.0 and target >= 1.0,
            }
        )
    geomeans: list[dict[str, Any]] = []
    for panel in panels:
        prediction = float(manifest["geomeans"][panel])
        target = float(target_section[panel]["speedup"][8])
        error = abs(prediction - target) / target
        geomeans.append(
            {
                "panel": panel,
                "prediction": prediction,
                "target": target,
                "relative_error": error,
                "pass_15pct": error <= limit,
            }
        )
    selected_parameters = documents["selected_model"]["figure20"]["parameters"]
    parameter_names = tuple(manifest["parameters"])
    parameter_checks = {
        "exact": manifest["parameters"] == selected_parameters,
        "count": len(parameter_names) == int(config["acceptance"]["require_parameter_count"]),
        "not_point_keyed": not any(
            token in name
            for name in parameter_names
            for token in ("256", "8192", "target_index")
        ),
        "services": manifest["projection_service"]["target_informed"] is True
        and manifest["attention_service"]["target_informed"] is True,
    }
    target_checks = {
        "dense": documents["legacy_execution"]["target"]["versus_dense_tcu"]["speedup"]
        == target_section["versus_dense_tcu"]["speedup"],
        "sparse": documents["legacy_execution"]["target"]["versus_sparse_cuda"][
            "speedup"
        ]
        == target_section["versus_sparse_cuda"]["speedup"],
        "current_sparse": all(
            math.isclose(
                float(cell["target_speedup"]),
                float(target_section["versus_sparse_cuda"]["speedup"][cell["target_index"]]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for cell in documents["current_audit"]["cells"]
        ),
    }
    all_errors = [cell["relative_error"] for cell in cells]
    all_errors.extend(item["relative_error"] for item in geomeans)
    numerical_checks = {
        "bars": len(cells) == int(config["acceptance"]["required_speedup_bars"])
        and all(cell["pass_15pct"] for cell in cells),
        "geomeans": len(geomeans) == int(config["acceptance"]["required_geomeans"])
        and all(item["pass_15pct"] for item in geomeans),
        "reported": len(all_errors) == int(config["acceptance"]["required_reported_points"])
        and sum(error <= limit for error in all_errors)
        == int(config["acceptance"]["required_passing_points"]),
        "directions": sum(cell["direction_match"] for cell in cells)
        == int(config["acceptance"]["required_direction_matches"]),
        "finite": all(math.isfinite(error) and error >= 0 for error in all_errors),
        "coverage": {
            (cell["panel"], cell["sequence_length"], cell["operator"])
            for cell in cells
        }
        == {
            (panel, sequence, operator)
            for panel in panels
            for sequence in sequences
            for operator in operators
        },
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        manifest_file["pass"] and all(manifest["checks"].values()),
        numerical_checks["coverage"],
        all(raw_checks.values()),
        all(trace_checks.values()),
        all(parameter_checks.values()),
        numerical_checks["bars"] and numerical_checks["directions"],
        numerical_checks["geomeans"],
        numerical_checks["reported"]
        and numerical_checks["finite"]
        and all(target_checks.values()),
        all(item["pass"] for item in source_files.values())
        and config["acceptance"]["independent_validation_claimed"] is False,
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": len(parent_checks) == 3,
        "manifest": manifest_file["pass"],
        "raw": len(raw_checks) == 16,
        "trace": len(trace_checks) == 16,
        "parameters": len(parameter_checks) == 4,
        "targets": len(target_checks) == 3,
        "numerical": len(numerical_checks) == 6,
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
        "paper_reproduction_claim": "figure20_trace_corrected_within_15pct_not_independent",
        "independent_validation_claimed": False,
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "composition_manifest": manifest_file,
        "raw_checks": raw_checks,
        "trace_checks": trace_checks,
        "parameter_checks": parameter_checks,
        "target_checks": target_checks,
        "parameters": manifest["parameters"],
        "cells": cells,
        "geomeans": geomeans,
        "numerical_checks": numerical_checks,
        "source_files": source_files,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "speedup_bars": len(cells),
            "geomeans": len(geomeans),
            "reported_points": len(all_errors),
            "passing_points": sum(error <= limit for error in all_errors),
            "mape": sum(all_errors) / len(all_errors),
            "max_relative_error": max(all_errors),
            "direction_matches": sum(cell["direction_match"] for cell in cells),
            "parameter_count": len(manifest["parameters"]),
            "raw_execution_matches": sum(raw_checks.values()),
            "trace_feature_matches": sum(trace_checks.values()),
            "figure20_numerically_reproduced_within_15pct": supported,
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
            "raw_checks",
            "trace_checks",
            "parameter_checks",
            "target_checks",
            "parameters",
            "cells",
            "geomeans",
            "numerical_checks",
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
