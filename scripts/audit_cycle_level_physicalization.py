#!/usr/bin/env python3
"""Audit H191 cycle-state physicalization and numerical preservation."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/cycle_level_physicalization_v1.yaml"


def patch_roundtrip(config: dict[str, Any]) -> dict[str, Any]:
    patch = PROJECT_ROOT / config["source_layout"]["core_patch"]
    source_root = PROJECT_ROOT / "third_party/dsa-framework/dsa-gem5/src/cpu/minor/ssim"
    current_header = source_root / "mlx_overlay.hh"
    current_source = source_root / "mlx_overlay.cc"
    with tempfile.TemporaryDirectory(prefix="h191-patch-") as directory:
        root = Path(directory)
        target = root / "src/cpu/minor/ssim"
        target.mkdir(parents=True)
        header = target / "mlx_overlay.hh"
        source = target / "mlx_overlay.cc"
        shutil.copy2(current_header, header)
        shutil.copy2(current_source, source)
        reverse = subprocess.run(
            ["git", "apply", "-R", "--check", str(patch)],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if reverse.returncode == 0:
            subprocess.run(["git", "apply", "-R", str(patch)], cwd=root, check=True)
        forward = subprocess.run(
            ["git", "apply", "--check", str(patch)],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if forward.returncode == 0:
            subprocess.run(["git", "apply", str(patch)], cwd=root, check=True)
        exact = (
            header.read_bytes() == current_header.read_bytes()
            and source.read_bytes() == current_source.read_bytes()
        )
    checks = {
        "patch": patch.is_file(),
        "reverse": reverse.returncode == 0,
        "forward": forward.returncode == 0,
        "roundtrip": exact,
    }
    return {"path": str(patch.relative_to(PROJECT_ROOT)), "checks": checks, "pass": all(checks.values())}


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    documents = {
        name: json.loads((PROJECT_ROOT / spec["path"]).read_text())
        for name, spec in config["frozen_inputs"].items()
        if Path(spec["path"]).suffix == ".json"
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
    timeline_path = PROJECT_ROOT / config["timeline_manifest"]
    compile_file = qualify(compile_path)
    run_file = qualify(run_path)
    timeline_file = qualify(timeline_path)
    compiled = json.loads(compile_path.read_text())
    executed = json.loads(run_path.read_text())
    timelines = json.loads(timeline_path.read_text())
    patch_check = patch_roundtrip(config)
    compile_checks = {
        "checks": all(compiled["checks"].values()),
        "count": len(compiled["outputs"]) == int(config["execution"]["figure23_configs"]),
        "physical": all(
            "physical_timing" in json.loads(
                (PROJECT_ROOT / item["primary"]["path"]).read_text()
            )
            for item in compiled["outputs"].values()
        ),
        "no_latency": all(
            "latency_service" not in json.loads(
                (PROJECT_ROOT / item["primary"]["path"]).read_text()
            )
            for item in compiled["outputs"].values()
        ),
        "blocks": all(
            item["blocks_sha256"] == item["parent_blocks_sha256"]
            for item in compiled["outputs"].values()
        ),
    }
    execution_checks = {
        "checks": all(executed["checks"].values()),
        "records": len(executed["records"])
        == int(config["execution"]["figure23_executions"]),
        "passes": all(record["pass"] for record in executed["records"]),
        "replays": all(executed["replay_checks"].values()),
        "no_postprocess": executed["post_processing_latency_service_enabled"] is False,
        "accounting": all(
            record["summary"]["cycles"]
            == record["summary"]["physical_timing"]["measured_scheduler_progress_cycles"]
            + record["summary"]["physical_timing"]["injected_congestion_stall_cycles"]
            and record["summary"]["raw_cycles"]
            == record["summary"]["physical_timing"]["pre_roi_progress_cycles"]
            + record["summary"]["physical_timing"]["measured_scheduler_progress_cycles"]
            for record in executed["records"]
        ),
    }
    timeline_items = [
        *timelines["figure19_timelines"],
        *timelines["figure20_timelines"],
    ]
    timeline_checks = {
        "checks": all(timelines["checks"].values()),
        "fig19": len(timelines["figure19_timelines"])
        == int(config["execution"]["figure19_timelines"]),
        "fig20": len(timelines["figure20_timelines"])
        == int(config["execution"]["figure20_timelines"]),
        "phases": sum(len(item["phases"]) for item in timeline_items)
        == int(config["execution"]["required_timeline_phases"]),
        "sums": all(
            item["total_cycles"] == sum(phase["cycles"] for phase in item["phases"])
            for item in timeline_items
        ),
        "positive": all(
            phase["cycles"] > 0 for item in timeline_items for phase in item["phases"]
        ),
        "no_postprocess": timelines["post_processing_latency_service_enabled"] is False,
    }
    fig23_parent = documents["figure23_parent"]
    fig23_cells: list[dict[str, Any]] = []
    optimized = {
        record["key"]: record["summary"]
        for record in executed["records"]
        if int(record["replay"]) == 1
    }
    for parent in fig23_parent["cells"]:
        sequence = int(parent["sequence_length"])
        window = int(parent["active_window"])
        baseline = optimized[f"N{sequence}-w{window}-baseline"]["cycles"]
        variant = optimized[f"N{sequence}-w{window}-{parent['series']}"]["cycles"]
        prediction = baseline / variant
        target = float(parent["target_speedup"])
        error = abs(prediction - target) / target
        fig23_cells.append(
            {
                "sequence_length": sequence,
                "active_window": window,
                "series": parent["series"],
                "prediction": prediction,
                "target": target,
                "relative_error": error,
                "direction_match": prediction > 1.0 and target > 1.0,
            }
        )
    fig19_parent = documents["figure19_parent"]
    f19_map = {
        (int(item["sequence_length"]), str(item["component"])): item
        for item in timelines["figure19_timelines"]
    }
    fig19_errors: list[float] = []
    fig19_directions: list[bool] = []
    for point in fig19_parent["points"]:
        component = (
            "attention"
            if point["series"] == "attention_latency_ms"
            else "ffn"
            if point["series"] == "ffn_latency_ms"
            else "fabnet"
        )
        prediction = f19_map[(int(point["sequence_length"]), component)]["latency_ms"]
        fig19_errors.append(abs(prediction - float(point["target_ms"])) / float(point["target_ms"]))
    for row in fig19_parent["derived_rows"]:
        sequence = int(row["sequence_length"])
        mlx_total = (
            f19_map[(sequence, "attention")]["total_cycles"]
            + f19_map[(sequence, "ffn")]["total_cycles"]
        )
        fabnet_total = f19_map[(sequence, "fabnet")]["total_cycles"]
        mlx_ms = mlx_total / timelines["clock_hz"] * 1000.0
        speedup = fabnet_total / mlx_total
        fig19_errors.extend(
            [
                abs(mlx_ms - float(row["mlx_total_target_ms"]))
                / float(row["mlx_total_target_ms"]),
                abs(speedup - float(row["speedup_target"]))
                / float(row["speedup_target"]),
            ]
        )
        fig19_directions.append(speedup > 1.0 and float(row["speedup_target"]) > 1.0)
    fig20_parent = documents["figure20_parent"]
    f20_map = {
        (
            str(item["panel"]),
            int(item["sequence_length"]),
            str(item["operator"]),
            str(item["role"]),
        ): item
        for item in timelines["figure20_timelines"]
    }
    fig20_errors: list[float] = []
    fig20_directions: list[bool] = []
    panel_predictions: dict[str, list[float]] = {}
    for cell in fig20_parent["cells"]:
        key = (cell["panel"], int(cell["sequence_length"]), cell["operator"])
        baseline = f20_map[(*key, "baseline")]["total_cycles"]
        mlx = f20_map[(*key, "mlx")]["total_cycles"]
        prediction = baseline / mlx
        target = float(cell["target"])
        fig20_errors.append(abs(prediction - target) / target)
        fig20_directions.append(prediction >= 1.0 and target >= 1.0)
        panel_predictions.setdefault(cell["panel"], []).append(prediction)
    for item in fig20_parent["geomeans"]:
        values = panel_predictions[item["panel"]]
        prediction = math.exp(sum(math.log(value) for value in values) / len(values))
        fig20_errors.append(abs(prediction - float(item["target"])) / float(item["target"]))
    limit = float(config["acceptance"]["maximum_relative_error"])
    numerical_checks = {
        "fig23": len(fig23_cells) == int(config["acceptance"]["figure23_required_points"])
        and all(cell["relative_error"] <= limit for cell in fig23_cells),
        "fig19": len(fig19_errors) == int(config["acceptance"]["figure19_required_points"])
        and all(error <= limit for error in fig19_errors),
        "fig20": len(fig20_errors) == int(config["acceptance"]["figure20_required_points"])
        and all(error <= limit for error in fig20_errors),
        "total": len(fig23_cells) + len(fig19_errors) + len(fig20_errors)
        == int(config["acceptance"]["required_total_points"]),
        "directions": sum(cell["direction_match"] for cell in fig23_cells)
        + sum(fig19_directions)
        + sum(fig20_directions)
        == int(config["acceptance"]["required_direction_matches"]),
    }
    all_errors = [cell["relative_error"] for cell in fig23_cells]
    all_errors.extend(fig19_errors)
    all_errors.extend(fig20_errors)
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        patch_check["pass"],
        compile_file["pass"] and all(compile_checks.values()),
        run_file["pass"]
        and execution_checks["records"]
        and execution_checks["passes"]
        and execution_checks["replays"],
        execution_checks["accounting"] and execution_checks["no_postprocess"],
        timeline_file["pass"]
        and timeline_checks["fig19"]
        and timeline_checks["fig20"]
        and timeline_checks["phases"],
        timeline_checks["checks"]
        and timeline_checks["sums"]
        and timeline_checks["positive"],
        numerical_checks["fig23"]
        and numerical_checks["fig19"]
        and numerical_checks["fig20"],
        numerical_checks["total"] and numerical_checks["directions"],
        timeline_checks["no_postprocess"]
        and all(item["pass"] for item in source_files.values())
        and config["acceptance"]["independent_validation_claimed"] is False,
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": len(parent_checks) == 6,
        "patch": patch_check["pass"],
        "compile": len(compile_checks) == 5,
        "execution": len(execution_checks) == 6,
        "timelines": len(timeline_checks) == 7,
        "numerical": len(numerical_checks) == 5,
        "finite": all(math.isfinite(error) and error >= 0 for error in all_errors),
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
        "paper_reproduction_claim": "cycle_physicalized_target_informed_not_independent",
        "independent_validation_claimed": False,
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "patch_check": patch_check,
        "compile_manifest": compile_file,
        "run_manifest": run_file,
        "timeline_manifest": timeline_file,
        "compile_checks": compile_checks,
        "execution_checks": execution_checks,
        "timeline_checks": timeline_checks,
        "numerical_checks": numerical_checks,
        "fig23_cells": fig23_cells,
        "fig19_errors": fig19_errors,
        "fig20_errors": fig20_errors,
        "source_files": source_files,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "figure23_configs": len(compiled["outputs"]),
            "figure23_executions": len(executed["records"]),
            "figure19_timelines": len(timelines["figure19_timelines"]),
            "figure20_timelines": len(timelines["figure20_timelines"]),
            "timeline_phases": sum(len(item["phases"]) for item in timeline_items),
            "reported_points": len(all_errors),
            "passing_points": sum(error <= limit for error in all_errors),
            "direction_matches": sum(cell["direction_match"] for cell in fig23_cells)
            + sum(fig19_directions)
            + sum(fig20_directions),
            "mape": sum(all_errors) / len(all_errors),
            "max_relative_error": max(all_errors),
            "latency_postprocessing_enabled": False,
            "cycle_level_physicalization_complete": supported,
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
            "patch_check",
            "compile_checks",
            "execution_checks",
            "timeline_checks",
            "numerical_checks",
            "fig23_cells",
            "fig19_errors",
            "fig20_errors",
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
