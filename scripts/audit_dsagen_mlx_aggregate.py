#!/usr/bin/env python3
"""Audit H43's trip-count-folded MLX CDC aggregation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from mlxsim.dsagen_overlay import pair_indices

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = PROJECT_ROOT / "artifacts/environment/h43"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/dsagen_mlx_aggregate_v1.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--verify-existing", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected YAML mapping: {path}")
    return value


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def qualify_file(path: Path, expected: dict[str, Any] | None = None) -> dict[str, Any]:
    is_file = path.is_file()
    size = path.stat().st_size if is_file else None
    digest = sha256_file(path) if is_file else None
    checks = {"is_file": is_file}
    if expected is not None:
        checks.update(
            {
                "bytes": size == int(expected["bytes"]),
                "sha256": digest == expected["sha256"],
            }
        )
    return {
        "path": str(path.relative_to(PROJECT_ROOT)) if is_file else str(path),
        "bytes": size,
        "sha256": digest,
        "checks": checks,
        "pass": all(checks.values()),
    }


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={PROJECT_ROOT}", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def source_audit(config: dict[str, Any]) -> dict[str, Any]:
    layout = config["source_layout"]
    token_map = {
        "compiler": [
            "compile_aggregate_radix2_cdc",
            "memory_address_sequence",
            "logical_pairs",
            "max_active_instruction_footprint_per_pe",
        ],
        "compiler_cli": ["FIXTURES", "bsmm-8-aggregate-fixed"],
        "overlay_header": ["memory_address_sequence", "memoryAddress"],
        "overlay_source": [
            "memory_address_sequence.size() != block.trip_count",
            "memoryAddress(block, state, instruction)",
        ],
        "json_driver": ["Overlay::FromJsonFile", "standalone JSON driver"],
        "trace_comparator": ["normalize_event", "normalized_events_identical"],
    }
    files: dict[str, Any] = {}
    for key, tokens in token_map.items():
        path = PROJECT_ROOT / layout[key]
        text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
        token_checks = {token: token in text for token in tokens}
        files[key] = {
            "path": layout[key],
            "tokens": token_checks,
            "pass": path.is_file() and all(token_checks.values()),
        }
    patch_path = PROJECT_ROOT / layout["tracked_patch"]
    patch = qualify_file(patch_path)
    patch_text = patch_path.read_text(encoding="utf-8", errors="replace") if patch_path.is_file() else ""
    forbidden = re.findall(
        r"\b(?:warp|simt|cta|coher(?:ence|ent)?)\b", patch_text, flags=re.IGNORECASE
    )
    reverse = subprocess.run(
        ["git", "apply", "--check", "--reverse", str(patch_path)],
        cwd=PROJECT_ROOT / "third_party/dsa-framework/dsa-gem5",
        check=False,
        capture_output=True,
        text=True,
    ) if patch_path.is_file() else None
    patch_checks = {
        "present": patch["pass"],
        "reverse_applies": reverse is not None and reverse.returncode == 0,
        "forbidden_gpu_state_absent": not forbidden,
    }
    return {
        "files": files,
        "patch": {
            **patch,
            "forbidden_gpu_tokens": forbidden,
            "reverse_stderr": reverse.stderr if reverse is not None else None,
            "checks": patch_checks,
            "pass": all(patch_checks.values()),
        },
        "pass": all(item["pass"] for item in files.values()) and all(patch_checks.values()),
    }


def audit_aggregate_document(document: dict[str, Any]) -> dict[str, Any]:
    metadata = document.get("metadata") or {}
    operator_kind = metadata.get("operator")
    width = int(metadata.get("width", 0))
    stages = int(metadata.get("stages", 0))
    pairs_per_stage = int(metadata.get("pairs_per_stage", 0))
    scalar_bytes = 8
    per_stage_pairs: dict[int, list[int]] = defaultdict(list)
    address_ok = True
    route_ok = True
    layout_ok = True
    event_counts: dict[str, int] = {}
    waits: list[tuple[int, str, int]] = []
    mesh_width = int(document.get("routing", {}).get("mesh_width", 0))
    mesh_height = int(document.get("routing", {}).get("mesh_height", 0))
    capacity = mesh_width * mesh_height
    for block in document.get("blocks") or []:
        stage = int(block["tag"]) - 1
        logical_pairs = [int(item) for item in block.get("logical_pairs") or []]
        per_stage_pairs[stage].extend(logical_pairs)
        trip_count = int(block["trip_count"])
        address_instructions = [
            item
            for item in block["instructions"]
            if item["pipeline"] in {"load", "store"}
        ]
        expected_layout = (
            ["load", "load", "compute", "compute", "store", "xfer"]
            if operator_kind == "bsmm"
            else ["load", "load", "compute", "compute", "compute", "store", "xfer"]
        )
        layout_ok = layout_ok and [
            item["pipeline"] for item in block["instructions"]
        ] == expected_layout
        address_ok = address_ok and len(address_instructions) == 3
        address_ok = address_ok and all(
            len(item.get("memory_address_sequence") or []) == trip_count
            for item in address_instructions
        )
        expected_a: list[int] = []
        expected_b: list[int] = []
        expected_out: list[int] = []
        for pair in logical_pairs:
            first, second = pair_indices(width, stage, pair)
            expected_a.append((stage * width + first) * scalar_bytes)
            expected_b.append((stage * width + second) * scalar_bytes)
            expected_out.append(((stage + 1) * width + first) * scalar_bytes)
        address_ok = address_ok and address_instructions[0]["memory_address_sequence"] == expected_a
        address_ok = address_ok and address_instructions[1]["memory_address_sequence"] == expected_b
        address_ok = address_ok and address_instructions[2]["memory_address_sequence"] == expected_out
        route_slots = {pair % capacity for pair in logical_pairs}
        route_ok = route_ok and len(route_slots) == 1
        xfer = block["instructions"][-1]
        route_ok = route_ok and xfer["pipeline"] == "xfer" and bool(xfer.get("route"))
        event = xfer.get("emit_event")
        if event:
            event_counts[event] = trip_count
        for event in block.get("wait_events") or []:
            waits.append((stage, event, trip_count))
    pair_coverage = all(
        sorted(per_stage_pairs[stage]) == list(range(pairs_per_stage))
        for stage in range(stages)
    )
    waits_ok = all(
        event in event_counts and event_counts[event] == trip_count
        for _, event, trip_count in waits
    )
    executed_pairs = sum(len(items) for items in per_stage_pairs.values())
    instructions_per_pair = 6 if operator_kind == "bsmm" else 7
    checks = {
        "mode": metadata.get("compilation_mode") == "aggregate",
        "paper_targets_absent": metadata.get("paper_performance_targets_consumed") is False,
        "pair_coverage_no_padding": pair_coverage,
        "trip_weighted_pairs": executed_pairs == metadata.get("total_pairs"),
        "trip_weighted_instructions": metadata.get("instruction_count")
        == executed_pairs * instructions_per_pair,
        "trip_weighted_memory": metadata.get("memory_requests") == executed_pairs * 3,
        "trip_weighted_transfers": metadata.get("transfers") == executed_pairs,
        "address_sequences": address_ok,
        "static_layout": layout_ok,
        "fixed_route_classes": route_ok,
        "event_counts": waits_ok,
    }
    return {
        "per_stage_pair_counts": {
            str(stage): len(items) for stage, items in per_stage_pairs.items()
        },
        "emitter_count": len(event_counts),
        "consumer_wait_count": len(waits),
        "checks": checks,
        "pass": all(checks.values()),
    }


def compiler_audit(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = EVIDENCE_ROOT / "mlx-aggregate-manifest.json"
    manifest_file = qualify_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_file["pass"] else {}
    replay_manifest = EVIDENCE_ROOT / "replay/mlx-aggregate-manifest.json"
    replay_ok = replay_manifest.is_file() and manifest_file["pass"] and (
        sha256_file(replay_manifest) == manifest_file["sha256"]
    )
    documents: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    bindings: dict[str, bool] = {}
    replay_files: dict[str, bool] = {}
    primary_names = set((manifest.get("comparisons") or {}).keys())
    for name, specification in (manifest.get("outputs") or {}).items():
        path = EVIDENCE_ROOT / specification["path"]
        artifact = qualify_file(path)
        artifacts[name] = artifact
        bindings[name] = (
            artifact["bytes"] == specification["bytes"]
            and artifact["sha256"] == specification["sha256"]
        )
        replay = EVIDENCE_ROOT / "replay" / path.name
        replay_files[name] = replay.is_file() and sha256_file(replay) == artifact["sha256"]
        if name in primary_names:
            document = json.loads(path.read_text(encoding="utf-8"))
            documents[name] = audit_aggregate_document(document)
    comparisons = manifest.get("comparisons") or {}
    comparison_checks = {
        name: item.get("pass") is True and all((item.get("checks") or {}).values())
        for name, item in comparisons.items()
    }
    fft8192 = artifacts.get("fft-8192") or {}
    fft8192_doc = json.loads(
        (EVIDENCE_ROOT / "mlx-fft-8192-aggregate.json").read_text(encoding="utf-8")
    ) if fft8192.get("pass") else {}
    fft_metadata = fft8192_doc.get("metadata") or {}
    bounds = config["fixtures"]["expected_bounds"]
    bound_checks = {
        "fft8192_blocks": fft_metadata.get("block_count")
        == bounds["fft8192"]["aggregate_blocks"],
        "fft8192_trip": fft_metadata.get("max_trip_count")
        == bounds["fft8192"]["trip_count"],
        "fft8192_bytes": (fft8192.get("bytes") or 10**12)
        < bounds["fft8192_config_max_bytes"],
        "fft8192_active_instruction_footprint": fft_metadata.get(
            "max_active_instruction_footprint_per_pe"
        )
        == 21,
        "bsmm64_blocks": (
            manifest.get("outputs", {}).get("bsmm-64", {}).get("metadata", {}).get(
                "block_count"
            )
            == bounds["bsmm64"]["aggregate_blocks"]
        ),
        "bsmm64_pairwise_reference_blocks": (
            manifest.get("comparisons", {}).get("bsmm-64", {}).get("reference", {}).get(
                "block_count"
            )
            == bounds["bsmm64"]["pairwise_blocks"]
        ),
        "bsmm64_trip": (
            manifest.get("outputs", {}).get("bsmm-64", {}).get("metadata", {}).get(
                "max_trip_count"
            )
            == bounds["bsmm64"]["max_trip_count"]
        ),
    }
    checks = {
        "manifest": manifest_file["pass"],
        "manifest_no_targets": manifest.get("paper_performance_targets_consumed") is False,
        "manifest_replay": replay_ok,
        "bindings": all(bindings.values()),
        "file_replays": all(replay_files.values()),
        "documents": all(item["pass"] for item in documents.values()),
        "conservation": all(comparison_checks.values()),
        "bounds": all(bound_checks.values()),
    }
    return {
        "manifest": manifest_file,
        "artifacts": artifacts,
        "bindings": bindings,
        "replay_files": replay_files,
        "documents": documents,
        "comparison_checks": comparison_checks,
        "bound_checks": bound_checks,
        "checks": checks,
        "pass": all(checks.values()),
    }


def parse_dsagen(text: str) -> dict[str, Any]:
    def last(pattern: str) -> int | None:
        matches = re.findall(pattern, text, flags=re.MULTILINE)
        return int(matches[-1]) if matches else None

    def prefixed(prefix: str) -> dict[str, Any] | None:
        matches = re.findall(rf"^{prefix} (\{{.*\}})$", text, flags=re.MULTILINE)
        return json.loads(matches[-1]) if matches else None

    return {
        "overlay": prefixed("MLX_OVERLAY_SUMMARY"),
        "adapter": prefixed("MLX_SPAD_ADAPTER_SUMMARY"),
        "roi_cycles": last(r"^Cycles:\s*(\d+)"),
        "cgra_instances": last(r"^CGRA Instances:\s*(\d+)"),
        "cgra_instructions": last(r"^CGRA Insts / Cycle:\s*(\d+)\s*/"),
        "sanity": "sanity check passed successfully!" in text,
        "normal_exit": "exiting with last active thread context" in text
        and "Simulated exit code not 0!" not in text,
    }


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    frozen = {
        name: qualify_file(PROJECT_ROOT / specification["path"], specification)
        for name, specification in config["frozen_inputs"].items()
        if isinstance(specification, dict) and "path" in specification
    }
    bridge = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["bridge_result"]["path"]).read_text(
            encoding="utf-8"
        )
    ) if frozen["bridge_result"]["pass"] else {}
    source = source_audit(config)
    compiler = compiler_audit(config)

    equivalence_root = EVIDENCE_ROOT / "equivalence"
    equivalence_files = {
        name: qualify_file(equivalence_root / name)
        for name in (
            "debug-pairwise-trace.jsonl",
            "debug-aggregate-trace.jsonl",
            "opt-pairwise-trace.jsonl",
            "opt-aggregate-trace.jsonl",
            "sanitize-pairwise-trace.jsonl",
            "sanitize-aggregate-trace.jsonl",
            "debug-pairwise-summary.json",
            "debug-aggregate-summary.json",
            "opt-pairwise-summary.json",
            "opt-aggregate-summary.json",
            "sanitize-pairwise-summary.json",
            "sanitize-aggregate-summary.json",
            "sanitize-pairwise-stderr.log",
            "sanitize-aggregate-stderr.log",
            "trace-equivalence.json",
        )
    }
    trace_report = json.loads(
        (equivalence_root / "trace-equivalence.json").read_text(encoding="utf-8")
    ) if equivalence_files["trace-equivalence.json"]["pass"] else {}
    pair_summary = json.loads(
        (equivalence_root / "debug-pairwise-summary.json").read_text(encoding="utf-8")
    ) if equivalence_files["debug-pairwise-summary.json"]["pass"] else {}
    aggregate_summary = json.loads(
        (equivalence_root / "debug-aggregate-summary.json").read_text(encoding="utf-8")
    ) if equivalence_files["debug-aggregate-summary.json"]["pass"] else {}
    equivalence_checks = {
        "all_files": all(item["pass"] for item in equivalence_files.values()),
        "normalized_trace": trace_report.get("normalized_events_identical") is True
        and trace_report.get("pairwise_event_count") == trace_report.get(
            "aggregate_event_count"
        ),
        "summary_exact": pair_summary == aggregate_summary,
        "cycle_exact": pair_summary.get("cycles") == aggregate_summary.get("cycles")
        and pair_summary.get("cycles", 0) > 0,
        "pairwise_cross_build": equivalence_files["debug-pairwise-trace.jsonl"][
            "sha256"
        ]
        == equivalence_files["opt-pairwise-trace.jsonl"]["sha256"]
        == equivalence_files["sanitize-pairwise-trace.jsonl"]["sha256"],
        "aggregate_cross_build": equivalence_files["debug-aggregate-trace.jsonl"][
            "sha256"
        ]
        == equivalence_files["opt-aggregate-trace.jsonl"]["sha256"]
        == equivalence_files["sanitize-aggregate-trace.jsonl"]["sha256"],
        "sanitizers_clean": equivalence_files["sanitize-pairwise-stderr.log"][
            "bytes"
        ]
        == equivalence_files["sanitize-aggregate-stderr.log"]["bytes"]
        == 0,
    }

    gem5_path = EVIDENCE_ROOT / "dsagen-bsmm64-aggregate-spad.log"
    gem5_file = qualify_file(gem5_path)
    gem5 = parse_dsagen(
        gem5_path.read_text(encoding="utf-8", errors="replace")
    ) if gem5_file["pass"] else {}
    overlay = gem5.get("overlay") or {}
    adapter = gem5.get("adapter") or {}
    gem5_checks = {
        "done": overlay.get("done") is True and overlay.get("cycles", 0) > 0,
        "instructions": overlay.get("instructions_issued")
        == overlay.get("instructions_completed")
        == 1152,
        "memory": overlay.get("external_memory_requests")
        == overlay.get("external_memory_completions")
        == adapter.get("requests")
        == adapter.get("responses")
        == 576,
        "events_and_routes": overlay.get("boundary_events_emitted") == 192
        and overlay.get("event_unblocked_issues_before_tag_complete", 0) > 0
        and overlay.get("skip_hops") == 192,
        "real_backpressure": overlay.get("stalls_by_reason", {}).get(
            "memory_queue_full", 0
        )
        > 0
        and adapter.get("unavailable_checks", 0) > 0,
        "base_workload": gem5.get("roi_cycles") == 569
        and gem5.get("cgra_instances") == 256
        and gem5.get("cgra_instructions") == 1024
        and gem5.get("sanity")
        and gem5.get("normal_exit"),
    }

    regression_files = {
        name: qualify_file(EVIDENCE_ROOT / filename)
        for name, filename in {
            "pairwise_bsmm8": "regression-pairwise-bsmm8.log",
            "pairwise_fft8": "regression-pairwise-fft8.log",
            "fixed": "regression-fixed.log",
            "disabled": "regression-disabled.log",
        }.items()
    }
    regressions = {
        name: parse_dsagen((PROJECT_ROOT / item["path"]).read_text(encoding="utf-8"))
        for name, item in regression_files.items()
        if item["pass"]
    }
    regression_checks = {
        "all_files": all(item["pass"] for item in regression_files.values()),
        "pairwise": all(
            (regressions.get(name, {}).get("overlay") or {}).get("done") is True
            for name in ("pairwise_bsmm8", "pairwise_fft8")
        ),
        "fixed": (regressions.get("fixed", {}).get("overlay") or {}).get(
            "memory_backend"
        )
        == "fixed",
        "disabled": regressions.get("disabled", {}).get("overlay") is None,
        "base_metrics": all(
            item.get("roi_cycles") == 569
            and item.get("cgra_instances") == 256
            and item.get("cgra_instructions") == 1024
            and item.get("sanity")
            and item.get("normal_exit")
            for item in regressions.values()
        ),
    }

    build_path = EVIDENCE_ROOT / "dsagen-gem5-aggregate-build-attempt1.log"
    build_file = qualify_file(build_path)
    build_text = build_path.read_text(encoding="utf-8", errors="replace") if build_file["pass"] else ""
    binary = qualify_file(
        PROJECT_ROOT / "third_party/dsa-framework/dsa-gem5/build/RISCV/gem5.opt"
    )
    pass_criteria = {
        "frozen_inputs": all(item["pass"] for item in frozen.values()),
        "h42_supported": bridge.get("hypothesis_status")
        == config["frozen_inputs"]["bridge_result"]["required_status"]
        and bridge.get("audit_integrity")
        is config["frozen_inputs"]["bridge_result"]["required_integrity"],
        "source": source["pass"],
        "compiler": compiler["pass"],
        "b8_equivalence": all(equivalence_checks.values()),
        "b64_gem5": all(gem5_checks.values()),
        "regressions": all(regression_checks.values()),
        "gem5_build": "RISCV/cpu/minor/ssim/mlx_overlay.cc -> .o" in build_text
        and "[    LINK]  -> RISCV/gem5.opt" in build_text
        and "scons: done building targets." in build_text,
        "gem5_binary": binary["pass"],
        "paper_performance_targets_consumed": False,
    }
    integrity = all(
        value
        for key, value in pass_criteria.items()
        if key != "paper_performance_targets_consumed"
    ) and pass_criteria["paper_performance_targets_consumed"] is False
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_commit": git_commit(),
        "hypothesis_status": "supported" if integrity else "rejected",
        "audit_integrity": integrity,
        "frozen_inputs": frozen,
        "source": source,
        "compiler": compiler,
        "equivalence": {
            "files": equivalence_files,
            "trace_report": trace_report,
            "pairwise_summary": pair_summary,
            "aggregate_summary": aggregate_summary,
            "checks": equivalence_checks,
            "pass": all(equivalence_checks.values()),
        },
        "b64_gem5": {
            "artifact": gem5_file,
            "parsed": gem5,
            "checks": gem5_checks,
            "pass": all(gem5_checks.values()),
        },
        "regressions": {
            "files": regression_files,
            "parsed": regressions,
            "checks": regression_checks,
            "pass": all(regression_checks.values()),
        },
        "build": build_file,
        "gem5_binary": binary,
        "pass_criteria": pass_criteria,
        "wall_seconds": time.perf_counter() - started,
    }


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    report = build_audit(config)
    output = PROJECT_ROOT / config["result_path"]
    if args.verify_existing:
        if not output.is_file():
            raise FileNotFoundError(output)
        existing = json.loads(output.read_text(encoding="utf-8"))
        keys = ("hypothesis_status", "audit_integrity", "pass_criteria")
        matches = all(existing.get(key) == report.get(key) for key in keys)
        print(json.dumps({"existing_matches": matches, **report}, indent=2, sort_keys=True))
        return 0 if matches else 1
    if args.preflight_only:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["audit_integrity"] and not output.exists() else 1
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
