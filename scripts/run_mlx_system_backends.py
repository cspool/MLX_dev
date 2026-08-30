#!/usr/bin/env python3
"""Run identical MLX workloads on the cycle and executable 4x4 RTL backends."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/system/mlx_riscv_system_v1.yaml"
RTL_SOURCES = [
    "rtl/mlx/mlx_fp16.sv",
    "rtl/mlx/mlx_fu.sv",
    "rtl/mlx/mlx_register_file.sv",
    "rtl/mlx/mlx_tag_buffer.sv",
    "rtl/mlx/mlx_config_network.sv",
    "rtl/mlx/mlx_data_network.sv",
    "rtl/mlx/mlx_control_logic.sv",
    "rtl/mlx/mlx_pe_top.sv",
    "rtl/mlx/mlx_array_pe_tile.sv",
    "rtl/mlx/mlx_array_4x4_distributed.sv",
    "rtl/mlx/mlx_array_4x4.sv",
]
OPCODE_NAMES = {
    0: "load",
    1: "store",
    2: "fma",
    3: "add",
    4: "max",
    5: "exp",
    6: "div",
    7: "shuffle",
    8: "xfer",
    9: "mul",
}


def digest(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def execute(command: list[str], log: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(result.stdout + result.stderr)
    return result


def parse_summary(text: str) -> dict[str, int | str]:
    match = re.search(
        r"MLX_ARRAY_PASS workload=(\S+) cycles=(\d+) instructions=(\d+) "
        r"load=(\d+) store=(\d+) compute=(\d+) xfer=(\d+) "
        r"stalls=(\d+) hops=(\d+) conflicts=(\d+)",
        text,
    )
    if not match:
        raise ValueError("backend log has no pass summary")
    fields = [
        "workload",
        "cycles",
        "instructions",
        "load",
        "store",
        "compute",
        "xfer",
        "stalls",
        "hops",
        "conflicts",
    ]
    return {
        name: value if name == "workload" else int(value)
        for name, value in zip(fields, match.groups(), strict=True)
    }


def parse_rtl_events(text: str) -> list[dict[str, int | str]]:
    events: list[dict[str, int | str]] = []
    pattern = re.compile(
        r"MLX_TRACE backend=rtl event=(issue|complete) cycle=(\d+) "
        r"pe=(\d+) pc=(\d+) op=(\d+)"
    )
    for match in pattern.finditer(text):
        event, cycle, pe, pc, operation = match.groups()
        events.append(
            {
                "event": event,
                "cycle": int(cycle),
                "pe": int(pe),
                "pc": int(pc),
                "opcode": int(operation),
                "operation": OPCODE_NAMES[int(operation)],
            }
        )
    return events


def parse_issue_events(text: str, backend: str) -> list[dict[str, int]]:
    """Return the architectural issue stream emitted by either backend."""
    pattern = re.compile(
        rf"MLX_TRACE backend={backend} event=issue cycle=(\d+) "
        r"pe=(\d+) pc=(\d+) op=(\d+)"
    )
    return [
        {
            "cycle": int(cycle),
            "pe": int(pe),
            "pc": int(pc),
            "opcode": int(operation),
        }
        for cycle, pe, pc, operation in pattern.findall(text)
    ]


def event_sequence_comparison(
    cycle_events: list[dict[str, int]], rtl_events: list[dict[str, int]]
) -> dict[str, Any]:
    """Compare identity/program order separately from global timing order."""

    def architectural_keys(events: list[dict[str, int]]) -> list[tuple[int, int, int]]:
        return [(item["pe"], item["pc"], item["opcode"]) for item in events]

    def per_pe(events: list[dict[str, int]]) -> dict[str, list[list[int]]]:
        streams: dict[str, list[list[int]]] = {str(pe): [] for pe in range(16)}
        for item in events:
            streams[str(item["pe"])].append([item["pc"], item["opcode"]])
        return streams

    cycle_keys = architectural_keys(cycle_events)
    rtl_keys = architectural_keys(rtl_events)
    cycle_per_pe = per_pe(cycle_events)
    rtl_per_pe = per_pe(rtl_events)
    return {
        "cycle_issue_events": len(cycle_keys),
        "rtl_issue_events": len(rtl_keys),
        "same_instruction_multiset": sorted(cycle_keys) == sorted(rtl_keys),
        "same_per_pe_program_order": cycle_per_pe == rtl_per_pe,
        "same_global_issue_order": cycle_keys == rtl_keys,
        "cycle_per_pe_issue_sequence": cycle_per_pe,
        "rtl_per_pe_issue_sequence": rtl_per_pe,
        "explanation": (
            "both backends preserve the identical (pc, opcode) sequence within each PE; "
            "their global issue interleaving differs because the cycle model serializes one "
            "ready tag while the physical RTL can issue concurrently across sixteen PEs"
        ),
    }


def instruction_timing(events: list[dict[str, int | str]]) -> dict[str, Any]:
    issues: dict[tuple[int, int, int], int] = {}
    latencies: dict[str, list[int]] = {name: [] for name in OPCODE_NAMES.values()}
    issue_cycles: dict[str, list[int]] = {name: [] for name in OPCODE_NAMES.values()}
    for event in events:
        key = (int(event["pe"]), int(event["pc"]), int(event["opcode"]))
        operation = str(event["operation"])
        if event["event"] == "issue":
            issues[key] = int(event["cycle"])
            issue_cycles[operation].append(int(event["cycle"]))
        elif key in issues:
            latencies[operation].append(int(event["cycle"]) - issues[key])
    result: dict[str, Any] = {}
    for opcode, name in OPCODE_NAMES.items():
        observed = latencies[name]
        cycles = sorted(set(issue_cycles[name]))
        intervals = [right - left for left, right in pairwise(cycles) if right > left]
        result[name] = {
            "opcode": opcode,
            "observations": len(observed),
            "latency_cycles_min": min(observed) if observed else None,
            "latency_cycles_max": max(observed) if observed else None,
            "observed_global_initiation_interval_min": min(intervals) if intervals else None,
            "ii_scope": "minimum positive issue-cycle separation across physical PEs in the workload suite",
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reuse-build", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    workload_manifest_path = PROJECT_ROOT / config["manifest"]
    workload_manifest = json.loads(workload_manifest_path.read_text())
    output = PROJECT_ROOT / "artifacts/environment/h205/backend-runs"
    build = PROJECT_ROOT / "build/mlx-system-array"
    build.mkdir(parents=True, exist_ok=True)
    cycle_binary = build / "tb_mlx_cycle_model.vvp"
    rtl_directory = build / "verilator"
    rtl_binary = rtl_directory / "Vmlx_array_4x4"

    build_records: dict[str, Any] = {}
    if not args.reuse_build or not cycle_binary.is_file():
        cycle_compile = execute(
            [
                "iverilog",
                "-g2012",
                "-DMLX_CYCLE_MODEL",
                "-s",
                "tb_mlx_array_4x4",
                "-o",
                str(cycle_binary),
                "rtl/mlx/mlx_fp16.sv",
                "rtl/mlx/mlx_fu.sv",
                "rtl/mlx/mlx_cycle_model.sv",
                "rtl/mlx/tb_mlx_array_4x4.sv",
            ],
            output / "compile-cycle.log",
        )
        build_records["cycle_compile_returncode"] = cycle_compile.returncode
    else:
        build_records["cycle_compile_returncode"] = 0
        build_records["cycle_reused"] = True

    if not args.reuse_build or not rtl_binary.is_file():
        rtl_directory.mkdir(parents=True, exist_ok=True)
        verilate = execute(
            [
                "verilator",
                "--cc",
                "--exe",
                "--trace",
                "--top-module",
                "mlx_array_4x4",
                "-Wno-fatal",
                "-Wno-PINCONNECTEMPTY",
                "-Wno-DECLFILENAME",
                "-Wno-WIDTH",
                "-Wno-UNUSED",
                "-DMLX_NO_WRAPPERS",
                "--Mdir",
                str(rtl_directory),
                *RTL_SOURCES,
                str(PROJECT_ROOT / "rtl/mlx/sim_mlx_array.cpp"),
            ],
            output / "compile-rtl-verilate.log",
        )
        make = execute(
            ["make", "-C", str(rtl_directory), "-f", "Vmlx_array_4x4.mk", "-j4"],
            output / "compile-rtl-cxx.log",
        )
        build_records.update(
            {
                "rtl_verilate_returncode": verilate.returncode,
                "rtl_compile_returncode": make.returncode,
            }
        )
    else:
        build_records.update(
            {
                "rtl_verilate_returncode": 0,
                "rtl_compile_returncode": 0,
                "rtl_reused": True,
            }
        )

    records = []
    rtl_events: list[dict[str, int | str]] = []
    for workload in workload_manifest["workloads"]:
        name = workload["name"]
        common = [
            str(PROJECT_ROOT / workload["program"]),
            str(PROJECT_ROOT / workload["input_hex"]),
            str(PROJECT_ROOT / workload["golden_hex"]),
            name,
            str(workload["input_vectors"]),
            str(workload["output_vectors"]),
            str(workload["output_spm_base"]),
        ]
        cycle_log = output / "cycle" / f"{name}.log"
        cycle_result = execute(
            [
                "vvp",
                str(cycle_binary),
                f"+PROGRAM={common[0]}",
                f"+INPUT={common[1]}",
                f"+GOLDEN={common[2]}",
                f"+WORKLOAD={name}",
                f"+INPUT_VECTORS={common[4]}",
                f"+OUTPUT_VECTORS={common[5]}",
                f"+OUTPUT_BASE={common[6]}",
            ],
            cycle_log,
        )
        rtl_log = output / "rtl" / f"{name}.log"
        rtl_result = execute([str(rtl_binary), *common], rtl_log)
        cycle_summary = parse_summary(cycle_log.read_text())
        rtl_summary = parse_summary(rtl_log.read_text())
        cycle_text = cycle_log.read_text()
        rtl_text = rtl_log.read_text()
        events = parse_rtl_events(rtl_text)
        event_comparison = event_sequence_comparison(
            parse_issue_events(cycle_text, "cycle"),
            parse_issue_events(rtl_text, "rtl"),
        )
        rtl_events.extend(events)
        records.append(
            {
                "workload": name,
                "cycle": {
                    "returncode": cycle_result.returncode,
                    "summary": cycle_summary,
                    "log": digest(cycle_log),
                },
                "rtl": {
                    "returncode": rtl_result.returncode,
                    "summary": rtl_summary,
                    "trace_events": len(events),
                    "log": digest(rtl_log),
                },
                "comparison": {
                    "same_instruction_count": cycle_summary["instructions"]
                    == rtl_summary["instructions"]
                    == workload["instruction_count"],
                    "cycle_difference": int(cycle_summary["cycles"])
                    - int(rtl_summary["cycles"]),
                    "cycle_to_rtl_ratio": int(cycle_summary["cycles"])
                    / int(rtl_summary["cycles"]),
                    "event_sequence": event_comparison,
                    "explanation": (
                        "cycle backend serializes ready tags through one shared SIMD service; "
                        "RTL issues across 16 physical PEs, arbitrates one SPM port, and routes "
                        "packets through neighbor/skip-hop flow control"
                    ),
                },
            }
        )

    performance_parent = PROJECT_ROOT / "artifacts/results/core-architecture-claims-run159.json"
    performance = json.loads(performance_parent.read_text())
    timing = instruction_timing(rtl_events)
    checks = {
        "builds": all(
            int(value) == 0
            for key, value in build_records.items()
            if key.endswith("returncode")
        ),
        "eight_functional_runs": len(records) == 4
        and all(
            item[backend]["returncode"] == 0
            for item in records
            for backend in ("cycle", "rtl")
        ),
        "same_instruction_counts": all(
            item["comparison"]["same_instruction_count"] for item in records
        ),
        "same_architectural_event_sequences": all(
            item["comparison"]["event_sequence"]["same_instruction_multiset"]
            and item["comparison"]["event_sequence"]["same_per_pe_program_order"]
            for item in records
        ),
        "global_interleaving_difference_observed": any(
            not item["comparison"]["event_sequence"]["same_global_issue_order"]
            for item in records
        ),
        "all_instruction_latencies": all(
            timing[name]["observations"] > 0 for name in OPCODE_NAMES.values()
        ),
        "stalls_measured": all(
            int(item["cycle"]["summary"]["stalls"])
            + int(item["rtl"]["summary"]["stalls"])
            > 0
            for item in records
        ),
        "conflicts_measured": any(
            int(item["cycle"]["summary"]["conflicts"])
            + int(item["rtl"]["summary"]["conflicts"])
            > 0
            for item in records
        ),
        "performance_trends": performance["hypothesis_status"] == "supported"
        and performance["summary"]["primary_claims_reproduced"]
        == performance["summary"]["primary_claims"],
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paper_performance_targets_consumed": False,
        "builds": build_records,
        "sources": {
            "rtl": {
                path: digest(PROJECT_ROOT / path)
                for path in [*RTL_SOURCES, "rtl/mlx/mlx_cycle_model.sv"]
            },
            "workloads": digest(workload_manifest_path),
            "performance_trend_parent": digest(performance_parent),
        },
        "records": records,
        "instruction_timing": timing,
        "checks": checks,
    }
    manifest_path = PROJECT_ROOT / "artifacts/environment/h205/backend-run-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    result = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "status": "supported" if all(checks.values()) else "rejected",
        "records": records,
        "instruction_timing": timing,
        "performance_trends": {
            "source": "architecture simulation",
            "result": digest(performance_parent),
            "primary_claims_reproduced": performance["summary"]["primary_claims_reproduced"],
            "primary_speedup_range": [
                performance["summary"]["minimum_primary_speedup"],
                performance["summary"]["maximum_primary_speedup"],
            ],
        },
        "checks": checks,
        "manifest": digest(manifest_path),
    }
    result_path = PROJECT_ROOT / "artifacts/results/mlx-system-backends-run210.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "checks": checks}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
