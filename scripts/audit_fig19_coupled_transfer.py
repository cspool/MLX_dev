#!/usr/bin/env python3
"""Audit H130's frozen current-coupled Figure 19 transfer."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_fig22_coupled_transfer import (
    PROJECT_ROOT,
    git_commit,
    qualify,
    summarize,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/fig19_coupled_transfer_v1.yaml"


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    h129 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h129"]["path"]).read_text()
    )
    target = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["targets"]["path"]).read_text()
    )["digitization"]["derived_targets"]
    parent_checks = {
        "status": h129["hypothesis_status"] == "supported"
        and h129["audit_integrity"] is True,
        "estimates": len(h129["combined_full_estimates"]) == 12
        and all(
            item["cycles"] is not None
            for item in h129["combined_full_estimates"].values()
        ),
    }
    composition = config["composition"]
    layers = int(composition["layers"])
    clock = int(composition["clock_hz"])
    limit = float(config["acceptance"]["relative_error_limit"])
    points: list[dict[str, Any]] = []
    simulated: list[dict[str, Any]] = []
    mapping_checks: dict[str, bool] = {}
    for index, n_value in enumerate(composition["sequence_lengths"]):
        n = int(n_value)
        attention_key = composition["attention_path"].format(n=n)
        ffn_keys = [pattern.format(n=n) for pattern in composition["ffn_paths"]]
        attention_cycles = float(h129["combined_full_estimates"][attention_key]["cycles"])
        ffn_cycles = sum(
            float(h129["combined_full_estimates"][key]["cycles"])
            for key in ffn_keys
        )
        values = {
            "attention_latency_ms": attention_cycles * layers / clock * 1000,
            "ffn_latency_ms": ffn_cycles * layers / clock * 1000,
        }
        values["total_latency_ms"] = (
            values["attention_latency_ms"] + values["ffn_latency_ms"]
        )
        simulated.append(
            {
                "sequence_length": n,
                "attention_path": attention_key,
                "ffn_paths": ffn_keys,
                "attention_cycles": attention_cycles,
                "ffn_cycles": ffn_cycles,
                **values,
            }
        )
        mapping_checks[str(n)] = (
            attention_key in h129["combined_full_estimates"]
            and all(key in h129["combined_full_estimates"] for key in ffn_keys)
        )
        for series in composition["series"]:
            prediction = float(values[series])
            target_value = float(target["mlx"][series][index])
            error = abs(prediction - target_value) / target_value
            points.append(
                {
                    "sequence_length": n,
                    "series": series,
                    "prediction_ms": prediction,
                    "target_ms": target_value,
                    "relative_error": error,
                    "pass_10pct": error <= limit,
                }
            )
    global_summary = summarize(points)
    by_series = {
        series: summarize([point for point in points if point["series"] == series])
        for series in composition["series"]
    }
    sum_checks = {
        str(item["sequence_length"]): abs(
            item["attention_latency_ms"]
            + item["ffn_latency_ms"]
            - item["total_latency_ms"]
        )
        < 1e-12
        for item in simulated
    }
    finite_checks = {
        f"{point['sequence_length']}-{point['series']}": (
            math.isfinite(point["prediction_ms"])
            and point["prediction_ms"] > 0
            and math.isfinite(point["target_ms"])
            and point["target_ms"] > 0
            and math.isfinite(point["relative_error"])
            and point["relative_error"] >= 0
        )
        for point in points
    }
    coverage_checks = {
        "points": len(points) == int(composition["required_points"]),
        "simulated": len(simulated) == len(composition["sequence_lengths"]),
        "mapping": all(mapping_checks.values()),
        "series": all(item["points"] == 4 for item in by_series.values()),
        "target_lengths": all(
            len(target["mlx"][series]) == len(composition["sequence_lengths"])
            for series in composition["series"]
        ),
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    source_text = "\n".join(
        (PROJECT_ROOT / path).read_text(errors="replace")
        for path in config["source_layout"].values()
    )
    forbidden = (
        "fit" + "_affine",
        "correction" + "_factor",
        "overlap" + "_cycles",
        "prediction" + " *",
        "prediction" + " +",
    )
    source_checks = {
        "no_fit": not any(token in source_text for token in forbidden),
        "layers": layers == 24,
        "clock": clock == 1_000_000_000,
    }
    all_points_pass = global_summary["passing_points"] == int(
        config["acceptance"]["required_passing_points"]
    )
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        all(coverage_checks.values()),
        all(mapping_checks.values()),
        source_checks["layers"] and source_checks["clock"],
        coverage_checks["target_lengths"],
        all(finite_checks.values()) and all(sum_checks.values()),
        all_points_pass,
        global_summary["points"] == 12
        and all(item["points"] == 4 for item in by_series.values()),
        all(source_checks.values()) and all(item["pass"] for item in source_files.values()),
        int(config["acceptance"]["required_passing_points"])
        == int(composition["required_points"]),
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parent": all(parent_checks.values()),
        "coverage": all(coverage_checks.values()),
        "mapping": all(mapping_checks.values()),
        "sums": all(sum_checks.values()),
        "finite": all(finite_checks.values()),
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
            "figure19_complete" if supported else "figure19_rejected"
        ),
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "mapping_checks": mapping_checks,
        "coverage_checks": coverage_checks,
        "sum_checks": sum_checks,
        "finite_checks": finite_checks,
        "simulated": simulated,
        "points": points,
        "summaries": {"global": global_summary, "by_series": by_series},
        "source_checks": source_checks,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "points": global_summary["points"],
            "passing_points": global_summary["passing_points"],
            "mape": global_summary["mape"],
            "max_relative_error": global_summary["max_relative_error"],
            "figure19_reproduced": supported,
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
            "simulated",
            "points",
            "summaries",
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
