#!/usr/bin/env python3
"""Audit H113 live dpu_pipelined and historical-memory coupling."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from mlxsim.coupled_pipelined_dpu_memory import scenarios
from mlxsim.dsagen_overlay import canonical_json

try:
    from scripts.audit_compute_dma_overlap import git_commit, qualify
except ModuleNotFoundError:
    from audit_compute_dma_overlap import git_commit, qualify

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/simulators/coupled_pipelined_dpu_memory_v1.yaml"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def temporal_memory_checks(
    events: list[dict[str, Any]], expected: dict[str, Any], half_bytes: int
) -> dict[str, bool]:
    tiles = int(expected["tiles"])

    def cycles(kind: str, tile: int) -> list[int]:
        return [
            int(event["cycle"])
            for event in events
            if event["kind"] == kind and int(event["tile"]) == tile
        ]

    per_tile = {}
    for tile in range(tiles):
        fill_complete = cycles("fill_complete", tile)
        loads = cycles("pe_load", tile)
        stores = cycles("pe_store", tile)
        releases = cycles("pe_release", tile)
        drains = cycles("drain_complete", tile)
        per_tile[tile] = (
            len(fill_complete) == 1
            and len(loads)
            == int(expected["blocks_per_tile"]) * int(expected["trip_count"])
            and len(stores) == int(expected["stores_per_tile"])
            and len(releases) == 1
            and len(drains) == 1
            and fill_complete[0] <= min(loads)
            and max(stores) <= releases[0] <= drains[0]
        )
    reuse = all(
        max(cycles("drain_complete", tile - 2))
        <= min(cycles("fill_start", tile))
        for tile in range(2, tiles)
    )
    parity = all(
        int(event["buffer"]) == int(event["tile"]) % 2
        for event in events
        if "tile" in event and int(event["tile"]) < tiles
    )
    addresses = all(
        0 <= int(event["relative_address"]) < half_bytes
        and int(event["physical_address"])
        == int(event["buffer"]) * half_bytes
        + int(event["relative_address"])
        for event in events
        if event["kind"] in {"pe_load", "pe_store"}
    )
    return {
        "per_tile": all(per_tile.values()),
        "reuse": reuse,
        "parity": parity,
        "addresses": addresses,
    }


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    parents = {
        name: json.loads(
            (PROJECT_ROOT / config["frozen_inputs"][name]["path"]).read_text()
        )
        for name in ("h109", "h106", "h107", "h112")
    }
    parent_checks = {
        name: report["hypothesis_status"]
        == config["frozen_inputs"][name]["required_status"]
        and report["audit_integrity"]
        is config["frozen_inputs"][name]["required_integrity"]
        for name, report in parents.items()
    }
    output_root = PROJECT_ROOT / config["output_root"]
    compile_path = (
        output_root / "coupled-pipelined-dpu-memory-compile-manifest.json"
    )
    run_path = output_root / "coupled-pipelined-dpu-memory-run-manifest.json"
    compile_file = qualify(compile_path)
    run_file = qualify(run_path)
    compiled = json.loads(compile_path.read_text())
    run = json.loads(run_path.read_text())

    rebuilt = scenarios(config)
    compile_checks = {}
    for name, item in compiled["outputs"].items():
        overlay_path = PROJECT_ROOT / item["overlay"]["path"]
        memory_path = PROJECT_ROOT / item["memory"]["path"]
        compile_checks[name] = (
            qualify(overlay_path, item["overlay"])["pass"]
            and qualify(memory_path, item["memory"])["pass"]
            and overlay_path.read_text() == canonical_json(rebuilt[name]["overlay"])
            and memory_path.read_text() == canonical_json(rebuilt[name]["memory"])
            and item["expected"] == rebuilt[name]["expected"]
        )

    records = {
        (item["mode"], item["scenario"], int(item["replay"])): item
        for item in run["records"]
    }
    record_checks = {}
    for key, item in records.items():
        files = all(
            qualify(
                PROJECT_ROOT / item[path_name], {"sha256": item[hash_name]}
            )["pass"]
            for path_name, hash_name in (
                ("summary_path", "summary_sha256"),
                ("overlay_trace_path", "overlay_trace_sha256"),
                ("memory_trace_path", "memory_trace_sha256"),
            )
        )
        record_checks["/".join(map(str, key))] = (
            item["pass"] is True
            and item["returncode"] == 0
            and item["stderr"] == ""
            and files
        )

    scenario_checks: dict[str, dict[str, bool]] = {}
    measurements: dict[str, dict[str, Any]] = {}
    half_bytes = int(config["hardware"]["spm_bytes"]) // int(
        config["hardware"]["buffer_halves"]
    )
    for name, item in compiled["outputs"].items():
        expected = item["expected"]
        record = records[("debug", name, 1)]
        summary = record["summary"]
        overlay = summary["overlay"]
        memory = summary["memory"]
        memory_events = read_jsonl(
            PROJECT_ROOT / record["memory_trace_path"]
        )
        temporal = temporal_memory_checks(memory_events, expected, half_bytes)
        checks = {
            "mode": overlay["pe_dependency_model"] == "dpu_pipelined"
            and overlay["memory_backend"] == "dpu_memory",
            "contexts": overlay["iteration_contexts_per_block"]
            == int(expected["contexts"])
            and 0 < overlay["max_inflight_iterations_per_block"]
            <= int(expected["contexts"]),
            "completion": overlay["done"] is True
            and overlay["instructions_issued"]
            == overlay["instructions_completed"]
            == int(expected["instructions"]),
            "fma": overlay["issued_by_pipeline"]["compute"]
            == int(expected["fma_issues"]),
            "external": overlay["external_memory_requests"]
            == overlay["external_memory_completions"]
            == int(expected["external_requests"])
            and overlay["issued_by_pipeline"]["load"]
            == int(expected["external_reads"])
            and overlay["issued_by_pipeline"]["store"]
            == int(expected["external_writes"]),
            "adapter_requests": memory["requests"]
            == memory["responses"]
            == int(expected["external_requests"])
            and memory["read_requests"] == int(expected["external_reads"])
            and memory["write_requests"] == int(expected["external_writes"]),
            "ownership": memory["ownership_wait_checks"] > 0
            and memory["ownership_violations"] == 0
            and overlay["external_memory_wait_cycles"] > 0,
            "tiles": memory["released_tiles"]
            == memory["drained_tiles"]
            == int(expected["tiles"]),
            "bytes": memory["offchip_read_bytes"]
            == int(expected["offchip_read_bytes"])
            and memory["offchip_write_bytes"]
            == int(expected["offchip_write_bytes"])
            and memory["dma_data_cycles"] == int(expected["dma_data_cycles"])
            and memory["dma_setup_cycles"] == 0,
            "spad": memory["spad"]["requests"]
            == memory["spad"]["responses"]
            == int(expected["external_requests"])
            and memory["spad"]["issued_bank_operations"]
            == int(expected["external_requests"]),
            "temporal": all(temporal.values()),
            "paper_free": summary["paper_performance_targets_consumed"] is False,
        }
        scenario_checks[name] = checks
        measurements[name] = {
            "overlay_cycles": summary["overlay_cycles"],
            "end_to_end_cycles": summary["end_to_end_cycles"],
            "max_inflight_iterations_per_block": overlay[
                "max_inflight_iterations_per_block"
            ],
            "ownership_wait_checks": memory["ownership_wait_checks"],
            "memory_queue_full_stalls": overlay["stalls_by_reason"].get(
                "memory_queue_full", 0
            ),
            "bank_issue_stalls": memory["spad"]["bank_issue_stalls"],
            "temporal_checks": temporal,
        }

    nonstop = records[("debug", "four_tile_non_stop_ctx4", 1)]["summary"]
    baseline = records[("debug", "four_tile_baseline_ctx4", 1)]["summary"]
    ctx4 = records[("debug", "single_tile_ctx4", 1)]["summary"]
    ctx2 = records[("debug", "single_tile_ctx2", 1)]["summary"]
    same = records[("debug", "same_bank_ctx4", 1)]["summary"]
    split = records[("debug", "split_bank_ctx4", 1)]["summary"]

    def work_signature(summary: dict[str, Any]) -> tuple[int, ...]:
        overlay, memory = summary["overlay"], summary["memory"]
        return (
            int(overlay["instructions_completed"]),
            int(overlay["issued_by_pipeline"]["compute"]),
            int(memory["requests"]),
            int(memory["offchip_read_bytes"]),
            int(memory["offchip_write_bytes"]),
        )

    relationship_checks = {
        "non_stop_work": work_signature(nonstop) == work_signature(baseline),
        "non_stop_faster": nonstop["end_to_end_cycles"]
        < baseline["end_to_end_cycles"],
        "contexts_work": work_signature(ctx4) == work_signature(ctx2),
        "contexts_not_slower": ctx4["end_to_end_cycles"]
        <= ctx2["end_to_end_cycles"],
        "contexts_reached": ctx4["overlay"][
            "max_inflight_iterations_per_block"
        ]
        == int(config["execution"]["contexts_full"])
        and ctx2["overlay"]["max_inflight_iterations_per_block"]
        == int(config["execution"]["contexts_limited"]),
        "bank_work": work_signature(same) == work_signature(split),
        "same_bank_pressure": same["memory"]["spad"]["bank_issue_stalls"]
        > split["memory"]["spad"]["bank_issue_stalls"],
    }
    replay_checks = {
        "manifest": all(run["checks"].values()),
        "per_scenario": all(run["replay_checks"].values()),
        "cross_build": all(run["cross_build_checks"].values()),
        "records": len(records) == int(config["execution"]["required_executions"])
        and all(record_checks.values()),
    }
    h106_manifest = qualify(
        PROJECT_ROOT / parents["h106"]["run_manifest"]["path"],
        parents["h106"]["run_manifest"],
    )
    h109_manifest = qualify(
        PROJECT_ROOT / parents["h109"]["run_manifest"]["path"],
        parents["h109"]["run_manifest"],
    )
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    executable_source = "\n".join(
        (PROJECT_ROOT / config["source_layout"][name]).read_text().lower()
        for name in ("scenario_core", "compiler", "runner")
    )
    target_free = (
        "fig25_roofline_utilization" not in executable_source
        and "paper_targets" not in executable_source
        and "residual_scale" not in executable_source
        and config["execution"]["paper_performance_targets_consumed"] is False
    )
    acceptance_gates = [
        all(item["pass"] for item in frozen.values())
        and all(parent_checks.values()),
        len(compiled["outputs"]) == int(config["execution"]["required_scenarios"])
        and all(compile_checks.values()),
        all(
            check["mode"] and check["contexts"] and check["completion"]
            for check in scenario_checks.values()
        ),
        all(
            check["fma"] and check["external"] and check["adapter_requests"]
            for check in scenario_checks.values()
        ),
        all(
            check["ownership"] and check["tiles"]
            for check in scenario_checks.values()
        ),
        all(check["temporal"] for check in scenario_checks.values()),
        all(
            check["bytes"] and check["spad"]
            for check in scenario_checks.values()
        ),
        relationship_checks["non_stop_work"]
        and relationship_checks["non_stop_faster"],
        relationship_checks["contexts_work"]
        and relationship_checks["contexts_not_slower"]
        and relationship_checks["contexts_reached"],
        relationship_checks["bank_work"]
        and relationship_checks["same_bank_pressure"],
        all(replay_checks.values()),
        h106_manifest["pass"] and h109_manifest["pass"] and target_free,
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "compile_manifest": compile_file["pass"],
        "run_manifest": run_file["pass"],
        "compile": all(compile_checks.values()),
        "records": all(record_checks.values()),
        "scenario_evaluated": len(scenario_checks)
        == int(config["execution"]["required_scenarios"]),
        "replays": all(replay_checks.values()),
        "regressions": h106_manifest["pass"] and h109_manifest["pass"],
        "source_files": all(item["pass"] for item in source_files.values()),
        "target_free": target_free,
        "acceptance_evaluated": len(acceptance_gates) == 12
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
        "paper_reproduction_claim": "none_target_free_coupling_contract_only",
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "compile_manifest": compile_file,
        "run_manifest": run_file,
        "compile_checks": compile_checks,
        "record_checks": record_checks,
        "scenario_checks": scenario_checks,
        "measurements": measurements,
        "relationship_checks": relationship_checks,
        "replay_checks": replay_checks,
        "regression_files": {
            "h106": h106_manifest,
            "h109": h109_manifest,
        },
        "acceptance_gates": acceptance_gates,
        "summary": {
            "scenarios": len(scenario_checks),
            "executions": len(records),
            "sanitizer_executions": sum(
                key[0] in {"asan", "ubsan"} for key in records
            ),
            "acceptance_gates_passed": sum(acceptance_gates),
            "acceptance_gates_total": len(acceptance_gates),
            "non_stop_cycles": nonstop["end_to_end_cycles"],
            "baseline_cycles": baseline["end_to_end_cycles"],
            "ctx4_cycles": ctx4["end_to_end_cycles"],
            "ctx2_cycles": ctx2["end_to_end_cycles"],
            "same_bank_stalls": same["memory"]["spad"]["bank_issue_stalls"],
            "split_bank_stalls": split["memory"]["spad"]["bank_issue_stalls"],
            "full_paper_rows_reproduced": 0,
            "full_paper_rows_total": 18,
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
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report = build_audit(config)
    output = PROJECT_ROOT / config["result_path"]
    if args.verify_existing:
        existing = json.loads(output.read_text())
        keys = (
            "hypothesis_status",
            "audit_integrity",
            "scenario_checks",
            "measurements",
            "relationship_checks",
            "acceptance_gates",
            "summary",
            "integrity_checks",
        )
        matches = all(existing.get(key) == report.get(key) for key in keys)
        print(json.dumps({"existing_matches": matches, **report}, indent=2))
        return 0 if matches else 1
    if args.preflight_only:
        print(json.dumps(report, indent=2))
        return 0 if report["audit_integrity"] and not output.exists() else 1
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"status": report["hypothesis_status"], **report["summary"]},
            indent=2,
        )
    )
    return 0 if report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
