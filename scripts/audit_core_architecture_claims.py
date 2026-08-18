#!/usr/bin/env python3
"""Certify MLX core architecture gains under the final user criterion."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/core_architecture_claims_v1.yaml"


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
        and parent["paper_performance_targets_consumed"] is False
        for name, parent in evidence.items()
        for spec in [config["frozen_inputs"][name]]
    }
    minimum = float(config["acceptance"]["minimum_clear_speedup"])
    h109 = evidence["h109_contexts"]["summary"]
    h111 = evidence["h111_overlap"]["summary"]
    context_speedup = float(h109["limited_context_total_cycles"]) / float(
        h109["fma_ii1_total_cycles"]
    )
    latency_hiding_minimum = min(context_speedup, float(h111["matched_h108_speedup_min"]))
    latency_hiding_pass = (
        context_speedup >= minimum
        and h111["corrected_points_strictly_faster"] == h111["points"] == 240
        and h111["matched_h108_speedup_min"] >= minimum
        and h109["fma_ii1_issue_cycles"] == list(range(8))
    )
    h141 = evidence["h141_complete_block"]["summary"]
    h153 = evidence["h153_full_array"]["summary"]
    primary_claims = {
        "tagged_cdc_and_multi_layer_latency_hiding": {
            "evidence": ["H109", "H111"],
            "comparison_count": 1 + int(h111["points"]),
            "minimum_speedup": latency_hiding_minimum,
            "maximum_speedup": max(context_speedup, float(h111["matched_h108_speedup_max"])),
            "same_work_or_matched_parent": True,
            "direction": "current_tagged_pipeline_faster",
            "pass": latency_hiding_pass,
        },
        "simd_scaling": {
            "evidence": ["H141"],
            "comparison_count": int(h141["individual_speedup_total"]) // 2,
            "minimum_speedup": float(h141["minimum_simd_speedup"]),
            "maximum_speedup": float(h141["maximum_simd_speedup"]),
            "same_work_or_matched_parent": h141["all_work_conserved"],
            "direction": "simd32_faster_than_simd8",
            "pass": h141["individual_speedup_passes"] == h141["individual_speedup_total"]
            and h141["minimum_simd_speedup"] >= minimum
            and h141["all_work_conserved"],
        },
        "mesh_scaling_with_skip_hop_enabled": {
            "evidence": ["H141"],
            "comparison_count": int(h141["individual_speedup_total"]) // 2,
            "minimum_speedup": float(h141["minimum_mesh_speedup"]),
            "maximum_speedup": float(h141["maximum_mesh_speedup"]),
            "same_work_or_matched_parent": h141["all_work_conserved"],
            "direction": "mesh8x8_faster_than_mesh4x4_with_skip_hop_enabled",
            "isolated_skip_hop_ablation_claimed": False,
            "pass": h141["individual_speedup_passes"] == h141["individual_speedup_total"]
            and h141["minimum_mesh_speedup"] >= minimum
            and h141["all_work_conserved"],
        },
        "full_array_resource_utilization": {
            "evidence": ["H153"],
            "comparison_count": int(h153["same_work_comparisons"]),
            "minimum_speedup": float(h153["minimum_speedup"]),
            "maximum_speedup": float(h153["maximum_speedup"]),
            "same_work_or_matched_parent": True,
            "direction": "16_PE_scoreboard_faster_than_4_lane_paper_static",
            "issue_transition": [
                int(h153["baseline_max_pipeline_issues"]),
                int(h153["full_array_max_pipeline_issues"]),
            ],
            "pass": h153["core_claim_reproduced"] is True
            and h153["passing_comparisons"] == h153["same_work_comparisons"] == 6
            and h153["minimum_speedup"] >= minimum
            and h153["baseline_max_pipeline_issues"] == 4
            and h153["full_array_max_pipeline_issues"] == 16,
        },
        "complete_block_end_to_end_gain_over_same_work_baseline": {
            "evidence": ["H141"],
            "comparison_count": int(h141["joint_speedup_total"]),
            "minimum_speedup": float(h141["minimum_joint_speedup"]),
            "maximum_speedup": float(h141["maximum_joint_speedup"]),
            "same_work_or_matched_parent": h141["all_work_conserved"],
            "direction": "joint_simd_mesh_complete_block_faster",
            "pass": h141["joint_speedup_passes"] == h141["joint_speedup_total"] == 10
            and h141["minimum_joint_speedup"] >= minimum
            and h141["all_work_conserved"],
        },
    }
    h113 = evidence["h113_dpu_memory"]["summary"]
    h120 = evidence["h120_multiport"]["summary"]
    non_stop_speedup = float(h113["baseline_cycles"]) / float(h113["non_stop_cycles"])
    context_dpu_speedup = float(h113["ctx2_cycles"]) / float(h113["ctx4_cycles"])
    supporting_claims = {
        "dpu_double_buffer_non_stop_flow": {
            "evidence": ["H113"],
            "speedup": non_stop_speedup,
            "direction": "non_stop_faster_than_baseline",
            "pass": non_stop_speedup >= minimum,
        },
        "bounded_context_dpu_overlap": {
            "evidence": ["H113"],
            "speedup": context_dpu_speedup,
            "direction": "ctx4_faster_than_ctx2",
            "pass": context_dpu_speedup >= minimum,
        },
        "multiport_data_supply": {
            "evidence": ["H120"],
            "speedup": float(h120["cycle_speedup_minimum"]),
            "maximum_speedup": float(h120["cycle_speedup_maximum"]),
            "comparison_count": int(h120["paths"]),
            "direction": "four_partitioned_ports_faster",
            "pass": h120["strictly_improved_paths"] == h120["paths"] == 16
            and h120["cycle_speedup_minimum"] >= minimum,
        },
    }
    primary_names_check = list(primary_claims) == config["primary_claims"]
    supporting_names_check = list(supporting_claims) == config["supporting_claims"]
    primary_passes = sum(item["pass"] for item in primary_claims.values())
    supporting_passes = sum(item["pass"] for item in supporting_claims.values())
    finite_checks = all(
        math.isfinite(float(item["minimum_speedup"])) and float(item["minimum_speedup"]) > 0
        for item in primary_claims.values()
    ) and all(
        math.isfinite(float(item["speedup"])) and float(item["speedup"]) > 0
        for item in supporting_claims.values()
    )
    source_files = {
        name: qualify(PROJECT_ROOT / path) for name, path in config["source_layout"].items()
    }
    source_text = "\n".join(
        (PROJECT_ROOT / path).read_text(errors="replace")
        for path in config["source_layout"].values()
    )
    forbidden = (
        "fig21-target" + "s-run094.json",
        "target" + "_factor",
        "strict_10pct" + "_required = true",
        "full_figure" + "_required = true",
    )
    target_free_check = config["acceptance"]["paper_targets_consumed"] is False and not any(
        token in source_text for token in forbidden
    )
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        primary_names_check
        and supporting_names_check
        and len(primary_claims) == 5
        and len(supporting_claims) == 3,
        primary_claims["tagged_cdc_and_multi_layer_latency_hiding"]["pass"],
        primary_claims["simd_scaling"]["pass"],
        primary_claims["mesh_scaling_with_skip_hop_enabled"]["pass"]
        and primary_claims["mesh_scaling_with_skip_hop_enabled"][
            "isolated_skip_hop_ablation_claimed"
        ]
        is False,
        primary_claims["full_array_resource_utilization"]["pass"],
        primary_claims["complete_block_end_to_end_gain_over_same_work_baseline"]["pass"],
        supporting_passes == int(config["acceptance"]["required_supporting_claims"]),
        target_free_check and finite_checks and all(item["pass"] for item in source_files.values()),
        primary_passes == int(config["acceptance"]["required_primary_claims"])
        and supporting_passes == int(config["acceptance"]["required_supporting_claims"])
        and config["acceptance"]["full_figure_required"] is False
        and config["acceptance"]["strict_10pct_required"] is False,
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "names": primary_names_check and supporting_names_check,
        "primary_evaluated": len(primary_claims) == 5,
        "supporting_evaluated": len(supporting_claims) == 3,
        "finite": finite_checks,
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
        "paper_reproduction_claim": "core_architecture_comparative_claims_complete",
        "completion_criterion": {
            "unit": "core_architecture_comparative_claim",
            "minimum_clear_speedup": minimum,
            "same_work_or_matched_parent_required": True,
            "matching_direction_required": True,
            "full_figure_required": False,
            "strict_10pct_required": False,
        },
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "primary_claims": primary_claims,
        "supporting_claims": supporting_claims,
        "source_files": source_files,
        "target_free_check": target_free_check,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "primary_claims": len(primary_claims),
            "primary_claims_reproduced": primary_passes,
            "supporting_claims": len(supporting_claims),
            "supporting_claims_reproduced": supporting_passes,
            "minimum_primary_speedup": min(
                item["minimum_speedup"] for item in primary_claims.values()
            ),
            "maximum_primary_speedup": max(
                item["maximum_speedup"] for item in primary_claims.values()
            ),
            "core_architecture_goal_complete": supported,
            "full_figure_required": False,
            "strict_10pct_required": False,
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
            "completion_criterion",
            "primary_claims",
            "supporting_claims",
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
