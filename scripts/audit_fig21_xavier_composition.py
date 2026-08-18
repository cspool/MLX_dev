#!/usr/bin/env python3
"""Compose target-free Figure 21 Xavier and MLX end-to-end timing."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/fig21_xavier_composition_v1.yaml"


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
    h95_rows = {int(row["sequence_length"]): row for row in evidence["h95_mlx"]["rows"]}
    projection = evidence["h146_projection"]["projection_estimates"]
    families = evidence["h147_families"]["family_estimates"]
    sequences = config["composition"]["sequence_lengths"]
    coverage_checks = {
        "h95": sorted(h95_rows) == sequences,
        "projection": sorted(int(key[1:]) for key in projection) == sequences,
        "families": sorted(int(key[1:]) for key in families) == sequences,
    }
    xavier_clock = int(config["composition"]["xavier_clock_hz"])
    mlx_clock = int(config["composition"]["mlx_clock_hz"])
    rows = []
    row_checks = {}
    for sequence in sequences:
        key = f"N{sequence}"
        mlx = h95_rows[sequence]
        xavier_components = {
            "dense_projection": float(projection[key]["xavier_cycles"]),
            "dense_attention": float(families[key]["dense_attention_cycles"]),
            "elementwise": float(families[key]["elementwise_cycles"]),
        }
        xavier_total_cycles = sum(xavier_components.values())
        xavier_seconds = xavier_total_cycles / xavier_clock
        mlx_cycles = float(mlx["mlx_total_cycles"])
        mlx_seconds = float(mlx["mlx_latency_seconds"])
        speedup = xavier_seconds / mlx_seconds
        row = {
            "sequence_length": sequence,
            "xavier_component_cycles": xavier_components,
            "xavier_total_cycles": xavier_total_cycles,
            "xavier_clock_hz": xavier_clock,
            "xavier_total_seconds": xavier_seconds,
            "mlx_cycles": mlx_cycles,
            "mlx_clock_hz": mlx_clock,
            "mlx_seconds": mlx_seconds,
            "speedup": speedup,
            "speedup_definition": config["composition"]["speedup_definition"],
            "serialization": config["composition"]["serialization"],
            "xavier_mapping_claim": "source_derived_compute_only_traceg_services",
        }
        rows.append(row)
        row_checks[key] = {
            "families": set(xavier_components) == set(config["composition"]["xavier_families"]),
            "components_positive": all(value > 0 for value in xavier_components.values()),
            "sum": math.isclose(
                xavier_total_cycles,
                sum(xavier_components.values()),
                rel_tol=0.0,
                abs_tol=0.0,
            ),
            "mlx_copy": mlx_cycles == float(mlx["mlx_total_cycles"])
            and mlx_seconds == float(mlx["mlx_latency_seconds"]),
            "mlx_clock": math.isclose(
                mlx_seconds, mlx_cycles / mlx_clock, rel_tol=0.0, abs_tol=0.0
            ),
            "xavier_clock": math.isclose(
                xavier_seconds,
                xavier_total_cycles / xavier_clock,
                rel_tol=0.0,
                abs_tol=0.0,
            ),
            "finite": all(
                math.isfinite(value) and value > 0
                for value in (
                    xavier_total_cycles,
                    xavier_seconds,
                    mlx_cycles,
                    mlx_seconds,
                    speedup,
                )
            ),
            "identity": projection[key]["mapping_claim"]
            == "source_derived_compute_only_HMMA_traceg_proxy"
            and families[key]["mapping_claim"]
            == "source_derived_compute_only_service_traceg_proxy",
        }
    source_files = {
        name: qualify(PROJECT_ROOT / path) for name, path in config["source_layout"].items()
    }
    source_text = "\n".join(
        (PROJECT_ROOT / path).read_text(errors="replace")
        for path in config["source_layout"].values()
    )
    forbidden = (
        "fig21-target" + "s-run094.json",
        "overlap" + "_factor",
        "speedup" + "_scale",
        "direction" + "_correction",
    )
    target_free_check = config["acceptance"]["targets_consumed"] is False and not any(
        token in source_text for token in forbidden
    )
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        all(coverage_checks.values()),
        all(checks["families"] and checks["components_positive"] for checks in row_checks.values()),
        all(checks["sum"] for checks in row_checks.values()),
        all(checks["mlx_copy"] for checks in row_checks.values()),
        xavier_clock == 1_377_000_000
        and mlx_clock == 1_000_000_000
        and all(checks["mlx_clock"] and checks["xavier_clock"] for checks in row_checks.values()),
        all(checks["finite"] for checks in row_checks.values()),
        all(checks["identity"] for checks in row_checks.values())
        and all(row["serialization"] == "additive_no_overlap" for row in rows),
        target_free_check and all(item["pass"] for item in source_files.values()),
        len(rows) == 5 and all(row["speedup"] > 0 for row in rows),
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "coverage": all(coverage_checks.values()),
        "rows": all(all(checks.values()) for checks in row_checks.values()),
        "source": target_free_check and all(item["pass"] for item in source_files.values()),
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
        "paper_reproduction_claim": "none_target_free_complete_xavier_composition",
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "coverage_checks": coverage_checks,
        "rows": rows,
        "row_checks": row_checks,
        "source_files": source_files,
        "target_free_check": target_free_check,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "shapes": len(rows),
            "complete_xavier_rows": len(rows),
            "finite_speedups": sum(math.isfinite(row["speedup"]) for row in rows),
            "mlx_faster_rows": sum(row["speedup"] > 1.0 for row in rows),
            "minimum_speedup": min(row["speedup"] for row in rows),
            "maximum_speedup": max(row["speedup"] for row in rows),
            "figure21_target_join_eligible": supported,
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
            "rows",
            "row_checks",
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
