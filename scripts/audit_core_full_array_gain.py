#!/usr/bin/env python3
"""Audit MLX 16-PE full-array gains against exact H92 same-work baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/core_full_array_gain_v1.yaml"


def file_record(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    h92 = json.loads((PROJECT_ROOT / config["frozen_inputs"]["h92_baseline"]["path"]).read_text())
    h141 = json.loads((PROJECT_ROOT / config["frozen_inputs"]["h141_mapping"]["path"]).read_text())
    h150 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h150_diagnosis"]["path"]).read_text()
    )
    compiler = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h152_compile_manifest"]["path"]).read_text()
    )
    parent_checks = {
        "h92": h92["hypothesis_status"]
        == config["frozen_inputs"]["h92_baseline"]["required_status"]
        and h92["audit_integrity"] is config["frozen_inputs"]["h92_baseline"]["required_integrity"],
        "h141": h141["hypothesis_status"]
        == config["frozen_inputs"]["h141_mapping"]["required_status"]
        and h141["audit_integrity"]
        is config["frozen_inputs"]["h141_mapping"]["required_integrity"],
        "h150": h150["hypothesis_status"]
        == config["frozen_inputs"]["h150_diagnosis"]["required_status"]
        and h150["audit_integrity"]
        is config["frozen_inputs"]["h150_diagnosis"]["required_integrity"],
        "compiler": compiler["experiment_id"] == "H152"
        and compiler["paper_performance_targets_consumed"] is False,
    }
    selected = config["comparison"]["selected_paths"]
    scale_pairs = config["comparison"]["same_work_scales"]
    selection_check = selected == [
        "structured_qkv",
        "structured_ffn1",
        "elementwise",
    ] and scale_pairs == [
        {"baseline_q": 16, "full_array_q": 4},
        {"baseline_q": 32, "full_array_q": 8},
    ]
    rows = []
    work_checks = {}
    replay_checks = {}
    mechanism_checks = {}
    evidence_files = {}
    for family in selected:
        path_key = f"N128-{family}"
        for pair in scale_pairs:
            baseline_key = f"{path_key}-q{pair['baseline_q']}"
            current_key = f"{path_key}-q{pair['full_array_q']}"
            baseline = h92["measurements"][baseline_key]
            current_compile = compiler["outputs"][current_key]
            metadata = current_compile["metadata"]
            run_root = PROJECT_ROOT / "artifacts/environment/h152/runs"
            first_path = run_root / f"{current_key}-first.json"
            second_path = run_root / f"{current_key}-second.json"
            first_adapter_path = run_root / f"{current_key}-first-adapter.json"
            second_adapter_path = run_root / f"{current_key}-second-adapter.json"
            first = json.loads(first_path.read_text())
            second = json.loads(second_path.read_text())
            first_adapter = json.loads(first_adapter_path.read_text())
            second_adapter = json.loads(second_adapter_path.read_text())
            evidence_files[current_key] = {
                "first": file_record(first_path),
                "second": file_record(second_path),
                "first_adapter": file_record(first_adapter_path),
                "second_adapter": file_record(second_adapter_path),
                "config": qualify(
                    PROJECT_ROOT / current_compile["artifact"]["path"],
                    current_compile["artifact"],
                ),
            }
            baseline_metadata = baseline["metadata"]
            work_checks[current_key] = (
                baseline_metadata["operation_counts"] == metadata["operation_counts"]
                and baseline_metadata["pipeline_counts"] == metadata["pipeline_counts"]
                and baseline_metadata["memory_requests"] == metadata["memory_requests"]
            )
            expected_instructions = sum(metadata["pipeline_counts"].values())
            replay_checks[current_key] = (
                evidence_files[current_key]["first"]["sha256"]
                == evidence_files[current_key]["second"]["sha256"]
                and evidence_files[current_key]["first_adapter"]["sha256"]
                == evidence_files[current_key]["second_adapter"]["sha256"]
                and first == second
                and first_adapter == second_adapter
                and first["done"] is True
                and first["instructions_issued"]
                == first["instructions_completed"]
                == expected_instructions
                and first["boundary_events_emitted"] == metadata["dynamic_event_count"]
                and first["external_memory_requests"]
                == first["external_memory_completions"]
                == metadata["memory_requests"]
                == first_adapter["requests"]
                == first_adapter["responses"]
                and first_adapter["ports"] == 4
                and first_adapter["axis"] == "x"
            )
            mechanism_checks[current_key] = (
                first["physical_pe_count"] == 16
                and first["mapped_pe_count"] == 16
                and first["max_pipeline_issues_in_cycle"] == 16
                and first["max_active_tags"] == 4
                and first["pe_dependency_model"] == "scoreboard_experimental"
                and metadata["physical_lane_count"] == 16
                and len(metadata["mapped_pes"]) == 16
            )
            baseline_cycles = int(baseline["cycles"])
            current_cycles = int(first["cycles"])
            speedup = baseline_cycles / current_cycles
            rows.append(
                {
                    "family": family,
                    "baseline_key": baseline_key,
                    "full_array_key": current_key,
                    "baseline_cycles": baseline_cycles,
                    "full_array_cycles": current_cycles,
                    "speedup": speedup,
                    "clear_gain": speedup >= float(config["acceptance"]["minimum_clear_speedup"]),
                    "same_work": work_checks[current_key],
                    "baseline_mechanism": config["comparison"]["baseline"],
                    "full_array_mechanism": config["comparison"]["full_array"],
                }
            )
    baseline_mechanism_check = (
        h150["h92_concurrency"]["normalized_lanes"] == [4]
        and h150["h92_concurrency"]["max_pipeline_issue_values"] == [4]
        and h150["h92_checks"]["active_window"] is True
        and h150["h92_checks"]["dependency"] is True
    )
    compile_checks = {
        key: record["deterministic"] is True
        and record["metadata"]["normalized"]["simd_width"] == 32
        and record["metadata"]["normalized"]["lanes"] == 16
        and record["metadata"]["physical_lane_count"] == 16
        and len(record["metadata"]["mapped_pes"]) == 16
        for key, record in compiler["outputs"].items()
        if key in {row["full_array_key"] for row in rows}
    }
    finite_checks = all(
        row["baseline_cycles"] > 0
        and row["full_array_cycles"] > 0
        and math.isfinite(row["speedup"])
        and row["speedup"] > 0
        for row in rows
    )
    passing = sum(row["clear_gain"] for row in rows)
    source_files = {
        name: qualify(PROJECT_ROOT / path) for name, path in config["source_layout"].items()
    }
    source_text = "\n".join(
        (PROJECT_ROOT / path).read_text(errors="replace")
        for path in config["source_layout"].values()
    )
    forbidden = (
        "fig21-target" + "s-run094.json",
        "residual" + "_scale",
        "cycle" + "_factor",
        "post_result" + "_workload_selection",
    )
    target_free_check = config["acceptance"]["targets_consumed"] is False and not any(
        token in source_text for token in forbidden
    )
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        selection_check and len(rows) == int(config["acceptance"]["required_comparisons"]),
        all(compile_checks.values()),
        all(work_checks.values()),
        all(replay_checks.values()),
        baseline_mechanism_check and all(mechanism_checks.values()),
        finite_checks,
        passing == int(config["acceptance"]["required_passing_comparisons"]),
        target_free_check and all(item["pass"] for item in source_files.values()),
        all(row["same_work"] for row in rows) and all(row["speedup"] >= 1.2 for row in rows),
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "selection": selection_check,
        "compile": len(compile_checks) == 6 and all(compile_checks.values()),
        "work": len(work_checks) == 6 and all(work_checks.values()),
        "replays": len(replay_checks) == 6 and all(replay_checks.values()),
        "mechanisms": baseline_mechanism_check and all(mechanism_checks.values()),
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
        "paper_reproduction_claim": "core_full_array_same_work_gain_only",
        "core_claim": "16_PE_scoreboard_full_array_outperforms_4_lane_paper_static",
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "selection_check": selection_check,
        "evidence_files": evidence_files,
        "compile_checks": compile_checks,
        "work_checks": work_checks,
        "replay_checks": replay_checks,
        "mechanism_checks": mechanism_checks,
        "baseline_mechanism_check": baseline_mechanism_check,
        "rows": rows,
        "source_files": source_files,
        "target_free_check": target_free_check,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "representative_workloads": len(selected),
            "same_work_comparisons": len(rows),
            "passing_comparisons": passing,
            "minimum_speedup": min(row["speedup"] for row in rows),
            "maximum_speedup": max(row["speedup"] for row in rows),
            "baseline_max_pipeline_issues": 4,
            "full_array_max_pipeline_issues": 16,
            "core_claim_reproduced": supported,
            "strict_full_figure_required": False,
            "active_core_claims_reproduced": 1 if supported else 0,
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
            "core_claim",
            "work_checks",
            "replay_checks",
            "mechanism_checks",
            "rows",
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
