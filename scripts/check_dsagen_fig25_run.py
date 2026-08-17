#!/usr/bin/env python3
"""Validate one H49 dsa-gem5 run and emit a compact measurement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.audit_dsagen_mlx_dma_memory import parse_prefixed_json, parse_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def build_measurement(config_path: Path, log_path: Path, stats_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    text = log_path.read_text(encoding="utf-8", errors="replace")
    stats = parse_stats(stats_path)
    overlay = parse_prefixed_json(text, "MLX_OVERLAY_SUMMARY") or {}
    adapter = parse_prefixed_json(text, "MLX_DMA_ADAPTER_SUMMARY") or {}
    metadata = config["metadata"]
    pipeline = metadata["pipeline_counts"]
    expected_requests = int(metadata["memory_requests"])
    expected_reads = int(pipeline["load"])
    expected_writes = int(pipeline["store"])
    metrics = {
        "compute_pipeline_occupancy": overlay.get("busy_cycles_by_pipeline", {}).get(
            "compute", 0
        )
        / overlay.get("cycles", 1),
        "cycles": overlay.get("cycles"),
        "compute_busy_cycles": overlay.get("busy_cycles_by_pipeline", {}).get("compute"),
        "dram_reads": stats.get("system.mem_ctrls.num_reads::.cpu.mlx_dma"),
        "dram_read_bytes": stats.get("system.mem_ctrls.bytes_read::.cpu.mlx_dma"),
        "l1_read_accesses": stats.get(
            "system.cpu.dcache.ReadReq_accesses::.cpu.mlx_dma"
        ),
        "l1_write_accesses": stats.get(
            "system.cpu.dcache.WriteReq_accesses::.cpu.mlx_dma"
        ),
    }
    checks = {
        "done": overlay.get("done") is True,
        "backend": overlay.get("memory_backend") == "dsagen_dma",
        "requests": overlay.get("external_memory_requests")
        == overlay.get("external_memory_completions")
        == adapter.get("requests")
        == adapter.get("responses")
        == expected_requests,
        "directions": adapter.get("read_requests")
        == adapter.get("read_responses")
        == expected_reads
        and adapter.get("write_requests") == adapter.get("write_responses") == expected_writes,
        "completion": adapter.get("failed_responses") == 0 and adapter.get("outstanding") == 0,
        "latency": adapter.get("max_response_cycles", 0) > 1,
        "concurrency": adapter.get("max_outstanding", 0) > 1,
        "l1_accesses": metrics["l1_read_accesses"] == expected_reads
        and metrics["l1_write_accesses"] == expected_writes,
        "dram": metrics["dram_reads"] == expected_reads
        and metrics["dram_read_bytes"] == expected_reads * 64,
        "guest": "sanity check passed successfully!" in text,
        "no_targets": metadata.get("paper_target_values_consumed") is False,
    }
    return {
        "schema_version": 1,
        "experiment_id": "H49",
        "operator": metadata["operator"]["name"],
        "case": metadata["case"]["name"],
        "metadata": metadata,
        "overlay": overlay,
        "adapter": adapter,
        "metrics": metrics,
        "checks": checks,
        "pass": all(checks.values()),
    }


def main() -> int:
    args = parse_args()
    report = build_measurement(args.config, args.log, args.stats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"operator": report["operator"], "case": report["case"], "pass": report["pass"]}))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
