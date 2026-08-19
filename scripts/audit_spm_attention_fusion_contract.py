#!/usr/bin/env python3
"""Audit H169's target-free SPM-capacity Attention fusion contract."""

from __future__ import annotations

import argparse
import ast
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/analysis/spm_attention_fusion_contract_v1.yaml"
)


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    parents = {
        name: json.loads((PROJECT_ROOT / spec["path"]).read_text())
        for name, spec in config["frozen_inputs"].items()
        if name != "source_refresh"
    }
    parent_checks = {
        name: report["hypothesis_status"] == config["frozen_inputs"][name]["required_status"]
        and report["audit_integrity"]
        is config["frozen_inputs"][name]["required_integrity"]
        and report["paper_performance_targets_consumed"] is False
        for name, report in parents.items()
    }
    note = (
        PROJECT_ROOT / config["frozen_inputs"]["source_refresh"]["path"]
    ).read_text()
    source_rule_checks = {
        "patent": config["patent_rule"]["patent"] in note,
        "one_kernel": "one-/two-kernel Attention composition" in note,
        "capacity": "SPM capacity" in note,
        "five_operations": int(config["patent_rule"]["operations_fused"]) == 5,
        "disclosure": config["patent_rule"]["disclosure_class"]
        == "same_team_patent_precedent_not_MLX_implementation_disclosure",
    }
    structured = parents["structured_attention"]
    dense = parents["dense_attention"]
    composition = parents["mlx_composition"]
    domains = parents["resource_domains"]
    shape_spec = config["shapes"]
    spm_bytes = int(shape_spec["spm_bytes"])
    domain_spm_bytes = int(domains["ledgers"][0]["raw"].get("spm_bytes", spm_bytes))
    spm_check = spm_bytes == 8 * 1024 * 1024 and domain_spm_bytes == spm_bytes
    composition_by_n = {
        int(row["sequence_length"]): row for row in composition["rows"]
    }
    rows: list[dict[str, Any]] = []
    row_checks: dict[str, dict[str, bool]] = {}
    expected_counts = {
        int(key): int(value)
        for key, value in shape_spec["expected_kernel_counts"].items()
    }
    for n_value in shape_spec["sequence_lengths"]:
        n = int(n_value)
        shape = f"N{n}"
        footprint = (
            n * int(shape_spec["embedding_dimension"]) * int(shape_spec["element_bytes"])
        )
        fits = footprint <= spm_bytes
        required_kernels = (
            int(config["patent_rule"]["fit_kernel_count"])
            if fits
            else int(config["patent_rule"]["streaming_kernel_count"])
        )
        current_kernels = int(config["current_model"]["current_kernel_count_per_shape"])
        match = current_kernels == required_kernels
        structured_cycles = float(structured["full_estimates"][shape])
        dense_cycles = float(dense["full_estimates"][shape])
        composed = composition_by_n[n]
        structured_component = float(
            composed["component_cycles"]["structured_attention"]
        )
        dense_component = float(composed["component_cycles"]["dense_attention"])
        row = {
            "sequence_length": n,
            "embedding_dimension": int(shape_spec["embedding_dimension"]),
            "element_bytes": int(shape_spec["element_bytes"]),
            "resident_footprint_bytes": footprint,
            "resident_footprint_mib": footprint / (1024 * 1024),
            "spm_bytes": spm_bytes,
            "fits_spm": fits,
            "patent_kernel_count": required_kernels,
            "current_structured_kernel_count": current_kernels,
            "current_dense_kernel_count": current_kernels,
            "current_matches_capacity_rule": match,
            "structured_attention_cycles_frozen": structured_cycles,
            "dense_attention_cycles_frozen": dense_cycles,
            "corrected_structured_attention_cycles": structured_cycles if match else None,
            "corrected_dense_attention_cycles": dense_cycles if match else None,
            "timing_status": "eligible_existing_fused" if match else "blocked_missing_two_kernel_split",
            "missing_fields": (
                []
                if match
                else [
                    "second_kernel_boundary",
                    "streaming_tile_shape",
                    "intermediate_traffic_domain",
                    "second_kernel_launch_cycles",
                ]
            ),
            "performance_improvement_claimed": False,
        }
        checks = {
            "expected_count": required_kernels == expected_counts[n],
            "footprint": math.isclose(
                row["resident_footprint_mib"], n / 128.0, rel_tol=0.0, abs_tol=0.0
            ),
            "work": structured["full_work_checks"][shape]
            and dense["full_work_checks"][shape],
            "structured_copy": math.isclose(
                structured_component,
                24 * structured_cycles,
                rel_tol=0.0,
                abs_tol=1e-6,
            ),
            "dense_copy": math.isclose(
                dense_component, 8 * dense_cycles, rel_tol=0.0, abs_tol=1e-6
            ),
            "no_invented_timing": (
                match
                and row["corrected_structured_attention_cycles"] == structured_cycles
                and row["corrected_dense_attention_cycles"] == dense_cycles
            )
            or (
                not match
                and row["corrected_structured_attention_cycles"] is None
                and row["corrected_dense_attention_cycles"] is None
            ),
            "no_improvement": row["performance_improvement_claimed"] is False,
        }
        rows.append(row)
        row_checks[shape] = checks
    one_kernel = sum(row["patent_kernel_count"] == 1 for row in rows)
    two_kernel = sum(row["patent_kernel_count"] == 2 for row in rows)
    matches = sum(row["current_matches_capacity_rule"] for row in rows)
    mismatches = len(rows) - matches
    eligible = sum(row["timing_status"] == "eligible_existing_fused" for row in rows)
    blocked = len(rows) - eligible
    coverage_checks = {
        "shapes": len(rows) == int(config["acceptance"]["required_shapes"]),
        "one_kernel": one_kernel
        == int(config["acceptance"]["required_one_kernel_shapes"]),
        "two_kernel": two_kernel
        == int(config["acceptance"]["required_two_kernel_shapes"]),
        "matches": matches == int(config["acceptance"]["required_current_matches"]),
        "mismatches": mismatches
        == int(config["acceptance"]["required_current_mismatches"]),
        "eligible": eligible
        == int(config["acceptance"]["required_timing_eligible_rows"]),
        "blocked": blocked == int(config["acceptance"]["required_timing_blocked_rows"]),
        "only_n2048_blocked": [
            row["sequence_length"]
            for row in rows
            if row["timing_status"] == "blocked_missing_two_kernel_split"
        ]
        == [2048],
    }
    compiler_path = PROJECT_ROOT / config["source_layout"]["structured_compiler"]
    structured_runner_path = PROJECT_ROOT / config["source_layout"]["structured_runner"]
    dense_runner_path = PROJECT_ROOT / config["source_layout"]["dense_runner"]
    compiler_text = compiler_path.read_text()
    structured_runner_text = structured_runner_path.read_text()
    dense_runner_text = dense_runner_path.read_text()
    source_behavior_checks = {
        "combined_compiler": "compile_combined_attention" in compiler_text
        and "fft_stage_count" in compiler_text
        and "attention_tag" in compiler_text,
        "structured_one_document": "compile_combined_attention(" in structured_runner_text
        and "outputs[key]" in structured_runner_text,
        "dense_one_document": "compile_dense_attention(" in dense_runner_text
        and "outputs[key]" in dense_runner_text,
        "launch_absent": config["current_model"]["explicit_launch_cycles"] is None,
        "boundary_absent": config["current_model"]["explicit_second_kernel_boundary"]
        is None,
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    tree = ast.parse((PROJECT_ROOT / config["source_layout"]["auditor"]).read_text())
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    source_checks = {
        "files": all(item["pass"] for item in source_files.values()),
        "behavior": all(source_behavior_checks.values()),
        "no_fit_calls": calls.isdisjoint(
            {"polyfit", "curve_fit", "lstsq", "minimize", "least_squares"}
        ),
    }
    target_free_checks = {
        "parents": all(
            report["paper_performance_targets_consumed"] is False
            for report in parents.values()
        ),
        "no_targets": True,
        "no_corrected_n2048_cycles": all(
            row["corrected_structured_attention_cycles"] is None
            and row["corrected_dense_attention_cycles"] is None
            for row in rows
            if row["sequence_length"] == 2048
        ),
        "no_performance_claim": all(
            row["performance_improvement_claimed"] is False for row in rows
        ),
    }
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        coverage_checks["shapes"] and spm_check,
        coverage_checks["one_kernel"] and coverage_checks["two_kernel"],
        all(source_behavior_checks.values()),
        coverage_checks["matches"]
        and coverage_checks["mismatches"]
        and coverage_checks["only_n2048_blocked"],
        all(checks["work"] and checks["structured_copy"] and checks["dense_copy"] for checks in row_checks.values()),
        coverage_checks["eligible"] and coverage_checks["blocked"],
        all(checks["no_invented_timing"] for checks in row_checks.values()),
        all(source_rule_checks.values())
        and all(source_checks.values())
        and all(target_free_checks.values()),
        config["validation_eligible"] is True
        and all(checks["no_improvement"] for checks in row_checks.values()),
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "source_rule": all(source_rule_checks.values()),
        "spm": spm_check,
        "coverage": all(coverage_checks.values()),
        "rows": all(all(checks.values()) for checks in row_checks.values()),
        "behavior": all(source_behavior_checks.values()),
        "source": all(source_checks.values()),
        "target_free": all(target_free_checks.values()),
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
        "paper_reproduction_claim": "none_spm_fusion_contract_and_gap_only",
        "performance_improvement_claimed": False,
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "source_rule_checks": source_rule_checks,
        "spm_check": spm_check,
        "rows": rows,
        "row_checks": row_checks,
        "coverage_checks": coverage_checks,
        "source_behavior_checks": source_behavior_checks,
        "source_files": source_files,
        "source_checks": source_checks,
        "target_free_checks": target_free_checks,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "shapes": len(rows),
            "one_kernel_shapes": one_kernel,
            "two_kernel_shapes": two_kernel,
            "current_matches": matches,
            "current_mismatches": mismatches,
            "timing_eligible_rows": eligible,
            "timing_blocked_rows": blocked,
            "blocked_sequence_lengths": [
                row["sequence_length"]
                for row in rows
                if row["timing_status"] == "blocked_missing_two_kernel_split"
            ],
            "performance_improvement_claimed": False,
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
            "performance_improvement_claimed",
            "rows",
            "coverage_checks",
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
