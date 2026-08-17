#!/usr/bin/env python3
"""Audit H48's full programmable-PE Transformer-block proxy."""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any

from scripts.audit_dsagen_mlx_dma_memory import (
    git_revision,
    load_yaml,
    parse_dsagen,
    parse_prefixed_json,
    parse_stats,
    qualify_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/dsagen_mlx_full_block_v1.yaml"
EVIDENCE_ROOT = PROJECT_ROOT / "artifacts/environment/h48"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--verify-existing", action="store_true")
    return parser.parse_args()


def read_log(path: Path) -> dict[str, Any]:
    artifact = qualify_file(path)
    text = path.read_text(encoding="utf-8", errors="replace") if artifact["pass"] else ""
    return {
        "artifact": artifact,
        "overlay": parse_prefixed_json(text, "MLX_OVERLAY_SUMMARY"),
        "dma_adapter": parse_prefixed_json(text, "MLX_DMA_ADAPTER_SUMMARY"),
        "spad_adapter": parse_prefixed_json(text, "MLX_SPAD_ADAPTER_SUMMARY"),
        "guest": parse_prefixed_json(text, "MLX_DMA_GUEST_SUMMARY"),
        "dsagen": parse_dsagen(text),
    }


def parent_audit(config: dict[str, Any]) -> dict[str, Any]:
    parents: dict[str, Any] = {}
    for name in ("paper", "dma_result", "hybrid_result"):
        specification = config["frozen_inputs"][name]
        artifact = qualify_file(PROJECT_ROOT / specification["path"], specification)
        document: dict[str, Any] = {}
        if artifact["pass"] and name != "paper":
            document = json.loads(
                (PROJECT_ROOT / specification["path"]).read_text(encoding="utf-8")
            )
        checks = {"artifact": artifact["pass"]}
        if name != "paper":
            checks["status"] = document.get("hypothesis_status") == specification[
                "required_status"
            ]
            checks["integrity"] = document.get("audit_integrity") is specification[
                "required_integrity"
            ]
        parents[name] = {
            "artifact": artifact,
            "checks": checks,
            "pass": all(checks.values()),
        }
    return {"parents": parents, "pass": all(item["pass"] for item in parents.values())}


def source_audit(config: dict[str, Any]) -> dict[str, Any]:
    layout = config["source_layout"]
    token_map = {
        "compiler_core": [
            "class NodeSpec",
            "def stage_graph",
            "compile_full_block",
            "score_v_relay",
            "attention_residual_rmsnorm",
            "final_residual_store",
        ],
        "compiler_cli": ["compile_full_block", "mlx-full-block-compile-manifest.json"],
        "runner": [
            "mlx_overlay_json_driver_sanitize",
            '"external_memory_requests":40',
            "system\\.mem_ctrls\\.num_reads::\\.cpu\\.mlx_dma",
        ],
        "overlay_header": ["FunctionalUnit", "RegisterFileConfig", "active_window"],
        "overlay_source": ["functional_units", "register_pending", "rf_read_bank"],
        "dma_adapter": ["lsq->findResponse", "sendStoreToStoreBuffer", "pushRequest"],
        "scoreboard_reference": [
            "Scoreboard::reserveRegisters",
            "Scoreboard::releaseRegisters",
            "Scoreboard::checkCollision",
        ],
        "operand_collector_reference": [
            "class opndcoll_rfu_t",
            "allocate_reads",
            "process_banks",
        ],
        "fu_pipeline_reference": ["pipelined_simd_unit::cycle", "ldst_unit::cycle"],
    }
    files: dict[str, Any] = {}
    for key, tokens in token_map.items():
        path = PROJECT_ROOT / layout[key]
        text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
        checks = {token: token in text for token in tokens}
        files[key] = {
            "path": layout[key],
            "tokens": checks,
            "pass": path.is_file() and all(checks.values()),
        }
    compiler_text = (PROJECT_ROOT / layout["compiler_core"]).read_text(encoding="utf-8")
    forbidden = re.findall(
        r"\b(?:warp|simt|cta|coher(?:ence|ent)?)\b", compiler_text, flags=re.IGNORECASE
    )
    checks = {
        "files": all(item["pass"] for item in files.values()),
        "gpu_execution_state_absent": not forbidden,
    }
    return {
        "files": files,
        "forbidden_gpu_tokens": forbidden,
        "checks": checks,
        "pass": all(checks.values()),
    }


def compiler_audit(config: dict[str, Any]) -> dict[str, Any]:
    paths = {
        "fixed": EVIDENCE_ROOT / "mlx-full-block-fixed.json",
        "dma": EVIDENCE_ROOT / "mlx-full-block-dma.json",
        "manifest": EVIDENCE_ROOT / "mlx-full-block-compile-manifest.json",
        "replay_check": EVIDENCE_ROOT / "compiler-replay-check.json",
    }
    files = {name: qualify_file(path) for name, path in paths.items()}
    fixed = json.loads(paths["fixed"].read_text(encoding="utf-8"))
    dma = json.loads(paths["dma"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    replay_check = json.loads(paths["replay_check"].read_text(encoding="utf-8"))
    fixed_copy = copy.deepcopy(fixed)
    dma_copy = copy.deepcopy(dma)
    fixed_backend = fixed_copy.pop("memory_backend")
    dma_backend = dma_copy.pop("memory_backend")
    metadata = dma.get("metadata") or {}
    blocks = dma.get("blocks") or []

    event_emitters: dict[str, int] = {}
    operation_counts: dict[str, int] = {}
    pipeline_counts = {name: 0 for name in ("load", "store", "compute", "xfer")}
    layout_ok = True
    memory_ok = True
    tags_and_placement = True
    dynamic_instructions = 0
    for block in blocks:
        tag = int(block["tag"])
        lane = int(block["id"].rsplit("lane", 1)[1])
        tags_and_placement &= block["pe"] == [lane, (tag - 1) % 4]
        tags_and_placement &= block["trip_count"] == 2
        pipelines = [instruction["pipeline"] for instruction in block["instructions"]]
        if "load" in pipelines:
            layout_ok &= pipelines[0] == "load"
        if "store" in pipelines and "xfer" in pipelines:
            layout_ok &= pipelines.index("store") < pipelines.index("xfer")
        dynamic_instructions += len(block["instructions"]) * int(block["trip_count"])
        for instruction in block["instructions"]:
            pipeline = instruction["pipeline"]
            pipeline_counts[pipeline] += int(block["trip_count"])
            if pipeline == "compute":
                operation = instruction["operation"]
                operation_counts[operation] = (
                    operation_counts.get(operation, 0) + int(block["trip_count"])
                )
            if pipeline in {"load", "store"}:
                sequence = instruction.get("memory_address_sequence") or []
                memory_ok &= instruction.get("memory_bytes") == 16
                memory_ok &= len(sequence) == 2
                memory_ok &= all(address % 16 == 0 for address in sequence)
            if event := instruction.get("emit_event"):
                event_emitters[event] = tag

    waits_ok = True
    all_waits_have_emitters = True
    for block in blocks:
        for event in block.get("wait_events") or []:
            all_waits_have_emitters &= event in event_emitters
            if event in event_emitters:
                waits_ok &= int(block["tag"]) == event_emitters[event] + 1

    expected_operations = {
        "add": 360,
        "mul": 40,
        "fma": 344,
        "fmax": 8,
        "fexp": 16,
        "fdiv": 16,
        "frsqrt": 16,
        "shuffle": 40,
    }
    functional_units = config["functional_units"]
    checks = {
        "files": all(item["pass"] for item in files.values()),
        "replay_fixed": replay_check.get("comparisons", {}).get("fixed", {}).get(
            "reference_sha256"
        )
        == files["fixed"]["sha256"]
        and replay_check.get("comparisons", {}).get("fixed", {}).get("identical") is True,
        "replay_dma": replay_check.get("comparisons", {}).get("dma", {}).get(
            "reference_sha256"
        )
        == files["dma"]["sha256"]
        and replay_check.get("comparisons", {}).get("dma", {}).get("identical") is True,
        "only_backend_differs": fixed_copy == dma_copy,
        "backends": fixed_backend == "fixed" and dma_backend == "dsagen_dma",
        "stages": metadata.get("stage_groups") == config["stage_groups"]
        and metadata.get("stage_count") == 28,
        "blocks": metadata.get("block_count") == len(blocks) == 228,
        "tags": sorted({int(block["tag"]) for block in blocks}) == list(range(1, 29)),
        "placement": tags_and_placement,
        "adjacent_events": waits_ok and all_waits_have_emitters,
        "event_count": metadata.get("event_edge_count") == 236
        and len(metadata.get("final_events") or []) == 4,
        "primitive_coverage": operation_counts == expected_operations
        and metadata.get("operation_counts") == expected_operations,
        "fu_table": dma.get("functional_units") == functional_units,
        "layout": layout_ok,
        "memory": memory_ok
        and metadata.get("memory_requests") == pipeline_counts["load"] + pipeline_counts["store"]
        == 40
        and pipeline_counts["load"] == 24
        and pipeline_counts["store"] == 16,
        "dynamic_instructions": dynamic_instructions == sum(pipeline_counts.values()) == 1352,
        "manifest": manifest.get("paper_performance_targets_consumed") is False,
        "no_targets": metadata.get("paper_performance_targets_consumed") is False,
    }
    return {
        "files": files,
        "metadata": metadata,
        "pipeline_counts": pipeline_counts,
        "operation_counts": operation_counts,
        "checks": checks,
        "pass": all(checks.values()),
    }


def standalone_audit() -> dict[str, Any]:
    root = EVIDENCE_ROOT / "runs/standalone"
    summary_files = {
        name: qualify_file(root / f"{name}-summary.json")
        for name in ("debug", "opt", "sanitize")
    }
    summaries = {
        name: json.loads((root / f"{name}-summary.json").read_text(encoding="utf-8"))
        for name in summary_files
    }
    debug_trace = qualify_file(root / "debug-trace.jsonl")
    trace_hashes_file = qualify_file(root / "trace-sha256.txt")
    trace_bytes_file = qualify_file(root / "trace-bytes.txt")
    hash_lines = (root / "trace-sha256.txt").read_text(encoding="utf-8").splitlines()
    byte_lines = (root / "trace-bytes.txt").read_text(encoding="utf-8").splitlines()
    hashes = [line.split()[0] for line in hash_lines]
    sizes = [int(line.split()[0]) for line in byte_lines if not line.endswith(" total")]
    summary = summaries["debug"]
    checks = {
        "summaries_present": all(item["pass"] for item in summary_files.values()),
        "summary_identity": len({item["sha256"] for item in summary_files.values()}) == 1,
        "trace_manifest": trace_hashes_file["pass"] and trace_bytes_file["pass"],
        "trace_identity": len(hashes) == 3
        and len(set(hashes)) == 1
        and hashes[0] == debug_trace["sha256"],
        "trace_sizes": len(sizes) == 3 and len(set(sizes)) == 1 and sizes[0] > 0,
        "sanitize_clean": qualify_file(root / "sanitize-stderr.log")["bytes"] == 0,
        "done": summary.get("done") is True and summary.get("cycles") == 393,
        "instructions": summary.get("instructions_issued")
        == summary.get("instructions_completed")
        == 1352,
        "window": summary.get("max_active_tags") == 4,
        "events": summary.get("boundary_events_emitted") == 480
        and summary.get("event_unblocked_issues_before_tag_complete") == 320,
        "pipelines": summary.get("issued_by_pipeline")
        == {"load": 24, "store": 16, "compute": 840, "xfer": 472},
        "no_external_memory": summary.get("external_memory_requests") == 0,
    }
    return {
        "summary_files": summary_files,
        "summary": summary,
        "debug_trace": debug_trace,
        "trace_hashes": hashes,
        "trace_sizes": sizes,
        "checks": checks,
        "pass": all(checks.values()),
    }


def gem5_audit() -> dict[str, Any]:
    fixed = read_log(EVIDENCE_ROOT / "runs/gem5/fixed/run.log")
    dma = read_log(EVIDENCE_ROOT / "runs/gem5/dma/run.log")
    stats_path = EVIDENCE_ROOT / "runs/gem5/dma/m5out/stats.txt"
    stats = parse_stats(stats_path)
    fixed_summary = fixed["overlay"] or {}
    dma_summary = dma["overlay"] or {}
    adapter = dma["dma_adapter"] or {}
    metrics = {
        "l1_read_accesses": stats.get("system.cpu.dcache.ReadReq_accesses::.cpu.mlx_dma"),
        "l1_read_misses": stats.get("system.cpu.dcache.ReadReq_misses::.cpu.mlx_dma"),
        "l1_write_accesses": stats.get("system.cpu.dcache.WriteReq_accesses::.cpu.mlx_dma"),
        "l1_write_misses": stats.get("system.cpu.dcache.WriteReq_misses::.cpu.mlx_dma"),
        "l2_read_accesses": stats.get("system.l2.ReadSharedReq_accesses::.cpu.mlx_dma"),
        "l2_read_misses": stats.get("system.l2.ReadSharedReq_misses::.cpu.mlx_dma"),
        "l2_store_rfo_hits": stats.get("system.l2.ReadExReq_hits::.cpu.mlx_dma"),
        "dram_reads": stats.get("system.mem_ctrls.num_reads::.cpu.mlx_dma"),
        "dram_read_bytes": stats.get("system.mem_ctrls.bytes_read::.cpu.mlx_dma"),
    }
    fixed_checks = {
        "done": fixed_summary.get("done") is True,
        "backend": fixed_summary.get("memory_backend") == "fixed",
        "instructions": fixed_summary.get("instructions_issued")
        == fixed_summary.get("instructions_completed")
        == 1352,
        "checksum": (fixed["guest"] or {}).get("store_checksum") == 84480,
        "normal_exit": fixed["dsagen"]["sanity"] and fixed["dsagen"]["normal_exit"],
    }
    dma_checks = {
        "done": dma_summary.get("done") is True,
        "backend": dma_summary.get("memory_backend") == "dsagen_dma",
        "instructions": dma_summary.get("instructions_issued")
        == dma_summary.get("instructions_completed")
        == 1352,
        "window": dma_summary.get("max_active_tags") == 4,
        "overlap": dma_summary.get("event_unblocked_issues_before_tag_complete", 0) > 320,
        "external_memory": dma_summary.get("external_memory_requests")
        == dma_summary.get("external_memory_completions")
        == 40,
        "directions": adapter.get("read_requests") == adapter.get("read_responses") == 24
        and adapter.get("write_requests") == adapter.get("write_responses") == 16,
        "completion": adapter.get("failed_responses") == 0
        and adapter.get("outstanding") == 0,
        "concurrency": adapter.get("max_outstanding") == 4,
        "latency": adapter.get("max_response_cycles", 0) > 1,
        "read_data": adapter.get("read_byte_sum") == 192,
        "write_data": (dma["guest"] or {}).get("store_checksum") == 63360,
        "normal_exit": dma["dsagen"]["sanity"] and dma["dsagen"]["normal_exit"],
        "l1": metrics["l1_read_accesses"] == metrics["l1_read_misses"] == 24
        and metrics["l1_write_accesses"] == metrics["l1_write_misses"] == 16,
        "l2": metrics["l2_read_accesses"] == metrics["l2_read_misses"] == 24
        and metrics["l2_store_rfo_hits"] == 16,
        "dram": metrics["dram_reads"] == 24 and metrics["dram_read_bytes"] == 1536,
    }
    return {
        "fixed": {**fixed, "checks": fixed_checks, "pass": all(fixed_checks.values())},
        "dma": {**dma, "checks": dma_checks, "pass": all(dma_checks.values())},
        "stats": qualify_file(stats_path),
        "memory_metrics": metrics,
        "pass": all(fixed_checks.values()) and all(dma_checks.values()),
    }


def regression_audit() -> dict[str, Any]:
    paths = {
        "h47_fixed": EVIDENCE_ROOT / "regressions/h47/fixed/run.log",
        "h47_dma": EVIDENCE_ROOT / "regressions/h47/dma/run.log",
        "h42_bsmm": EVIDENCE_ROOT / "regressions/cdc/bsmm-b8/run.log",
        "h42_fft": EVIDENCE_ROOT / "regressions/cdc/fft-l8/run.log",
        "h42_stress": EVIDENCE_ROOT / "regressions/cdc/bsmm-b16-stress/run.log",
        "h41_fixed": EVIDENCE_ROOT / "regressions/overlay/enabled/run.log",
        "h41_disabled": EVIDENCE_ROOT / "regressions/overlay/disabled/run.log",
    }
    runs = {name: read_log(path) for name, path in paths.items()}

    def base_ok(item: dict[str, Any]) -> bool:
        metrics = item["dsagen"]
        return (
            metrics["roi_cycles"] == 569
            and metrics["cgra_instances"] == 256
            and metrics["cgra_instructions"] == 1024
            and metrics["sanity"]
            and metrics["normal_exit"]
        )

    checks = {
        "h47_fixed": (runs["h47_fixed"]["overlay"] or {}).get("cycles") == 17
        and (runs["h47_fixed"]["guest"] or {}).get("store_checksum") == 84480,
        "h47_dma": (runs["h47_dma"]["overlay"] or {}).get("external_memory_requests")
        == 128
        and (runs["h47_dma"]["dma_adapter"] or {}).get("responses") == 128
        and (runs["h47_dma"]["guest"] or {}).get("store_checksum") == 0,
        "h42_bsmm": (runs["h42_bsmm"]["spad_adapter"] or {}).get("responses") == 36
        and base_ok(runs["h42_bsmm"]),
        "h42_fft": (runs["h42_fft"]["spad_adapter"] or {}).get("responses") == 36
        and base_ok(runs["h42_fft"]),
        "h42_stress": (runs["h42_stress"]["overlay"] or {})
        .get("stalls_by_reason", {})
        .get("memory_queue_full", 0)
        > 0
        and base_ok(runs["h42_stress"]),
        "h41_fixed": (runs["h41_fixed"]["overlay"] or {}).get("cycles") == 5
        and base_ok(runs["h41_fixed"]),
        "h41_disabled": runs["h41_disabled"]["overlay"] is None
        and base_ok(runs["h41_disabled"]),
    }
    return {"runs": runs, "checks": checks, "pass": all(checks.values())}


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    parents = parent_audit(config)
    source = source_audit(config)
    compiler = compiler_audit(config)
    standalone = standalone_audit()
    gem5 = gem5_audit()
    regressions = regression_audit()
    evidence_files = {
        "protocol": qualify_file(PROJECT_ROOT / "experiments/h48-dsagen-mlx-full-block/protocol.md"),
        "config": qualify_file(DEFAULT_CONFIG),
        "manifest": qualify_file(EVIDENCE_ROOT / "mlx-full-block-compile-manifest.json"),
        "replay_check": qualify_file(EVIDENCE_ROOT / "compiler-replay-check.json"),
        "gem5_binary": qualify_file(
            PROJECT_ROOT / "third_party/dsa-framework/dsa-gem5/build/RISCV/gem5.opt"
        ),
        "guest_binary": qualify_file(
            PROJECT_ROOT / "third_party/dsa-framework/dsa-apps/sdk/compiled/ss-mlx-dma.out"
        ),
    }
    pass_criteria = {
        "parents": parents["pass"],
        "source": source["pass"],
        "compiler": compiler["pass"],
        "standalone": standalone["pass"],
        "gem5": gem5["pass"],
        "regressions": regressions["pass"],
        "evidence_files": all(item["pass"] for item in evidence_files.values()),
        "no_paper_targets": config["frozen_inputs"]["paper"][
            "consumed_performance_targets"
        ]
        == [],
    }
    integrity = all(pass_criteria.values())
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "classification": config["classification"],
        "hypothesis_status": "supported" if integrity else "rejected",
        "audit_integrity": integrity,
        "claim_scope": {
            "supported": [
                "one folded schedule covers all registered structured Transformer phases",
                "inter-PE dependencies are adjacent-tag CDC events with explicit relays",
                "PE-local heterogeneous FU/RF hazards and real DSAGEN DMA timing coexist",
            ],
            "not_claimed": [
                "numerically exact Llama2 inference",
                "authors' unpublished instruction schedule",
                "Figure 18-25 performance accuracy",
                "GPU warp/SIMT/CTA/coherence behavior",
            ],
        },
        "git_revision": git_revision(PROJECT_ROOT),
        "dsagen_revision": git_revision(
            PROJECT_ROOT / "third_party/dsa-framework/dsa-gem5"
        ),
        "parents": parents,
        "source": source,
        "compiler": compiler,
        "standalone": standalone,
        "gem5": gem5,
        "regressions": regressions,
        "evidence_files": evidence_files,
        "pass_criteria": pass_criteria,
        "paper_performance_targets_consumed": False,
    }


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config.resolve())
    report = build_audit(config)
    output = PROJECT_ROOT / config["result_path"]
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.verify_existing:
        if not output.is_file() or output.read_text(encoding="utf-8") != rendered:
            raise SystemExit("existing H48 result does not match a fresh audit")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("experiment_id", "run_id", "hypothesis_status", "audit_integrity")
            },
            indent=2,
        )
    )
    return 0 if report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
