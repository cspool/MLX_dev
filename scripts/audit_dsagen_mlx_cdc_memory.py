#!/usr/bin/env python3
"""Audit H42's CDC compiler, event wakeup, and DSAGEN scratchpad adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/dsagen_mlx_cdc_memory_v1.yaml"
EVIDENCE_ROOT = PROJECT_ROOT / "artifacts/environment/h42"


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


def last_int(text: str, pattern: str) -> int | None:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    return int(matches[-1]) if matches else None


def parse_dsagen_metrics(text: str) -> dict[str, Any]:
    return {
        "roi_cycles": last_int(text, r"^Cycles:\s*(\d+)"),
        "cgra_instances": last_int(text, r"^CGRA Instances:\s*(\d+)"),
        "cgra_instructions": last_int(text, r"^CGRA Insts / Cycle:\s*(\d+)\s*/"),
        "sanity_check_passed": "sanity check passed successfully!" in text,
        "normal_exit": "exiting with last active thread context" in text
        and "Simulated exit code not 0!" not in text,
    }


def parse_prefixed_json(text: str, prefix: str) -> dict[str, Any] | None:
    matches = re.findall(rf"^{re.escape(prefix)} (\{{.*\}})$", text, flags=re.MULTILINE)
    return json.loads(matches[-1]) if matches else None


def source_audit(config: dict[str, Any]) -> dict[str, Any]:
    layout = config["source_layout"]
    token_map = {
        "compiler": [
            "compile_radix2_cdc",
            "pair_indices",
            "memory_backend",
            "emit_event",
            "wait_events",
        ],
        "compiler_cli": ["mlx-bsmm-b8.json", "mlx-fft-l8.json"],
        "memory_adapter_header": [
            "class Gem5ScratchpadAdapter",
            "MemoryAdapter",
            "handleResponse",
            "ResponseBase",
        ],
        "memory_adapter_source": [
            "scratchpad.rb->Available()",
            "scratchpad.rb->Decode",
            "MemoryOperation::DMO_Read",
            "MemoryOperation::DMO_Write",
            "handleResponse",
        ],
        "overlay_header": [
            "MemoryBackend",
            "MemoryAdapter",
            "wait_events",
            "emit_event",
            "external_memory",
        ],
        "overlay_source": [
            "iterationEventsReady",
            "event_dependency",
            "memory_queue_full",
            "takeCompletion",
            "event_unblocked_issues_before_tag_complete",
        ],
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
    gem5_root = PROJECT_ROOT / "third_party/dsa-framework/dsa-gem5"
    reverse = subprocess.run(
        ["git", "apply", "--check", "--reverse", str(patch_path)],
        cwd=gem5_root,
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


def audit_operator(document: dict[str, Any], expected: dict[str, Any], kind: str) -> dict[str, Any]:
    metadata = document.get("metadata") or {}
    blocks = document.get("blocks") or []
    emitted: dict[str, int] = {}
    waits: list[tuple[int, str]] = []
    layout_ok = True
    addresses_ok = True
    routes_ok = True
    skip_steps = set(document.get("routing", {}).get("skip_steps") or [])
    mesh_width = int(document.get("routing", {}).get("mesh_width", 0))
    mesh_height = int(document.get("routing", {}).get("mesh_height", 0))
    for block in blocks:
        pipelines = [item["pipeline"] for item in block["instructions"]]
        expected_layout = (
            ["load", "load", "compute", "compute", "store", "xfer"]
            if kind == "bsmm"
            else ["load", "load", "compute", "compute", "compute", "store", "xfer"]
        )
        layout_ok = layout_ok and pipelines == expected_layout
        for event in block.get("wait_events") or []:
            waits.append((int(block["tag"]), event))
        for instruction in block["instructions"]:
            if instruction["pipeline"] in {"load", "store"}:
                address = int(instruction["memory_address"])
                size = int(instruction["memory_bytes"])
                addresses_ok = addresses_ok and size == 8 and address % size == 0
                addresses_ok = addresses_ok and address + size <= 16 * 1024 * 1024
            event = instruction.get("emit_event")
            if event:
                emitted[event] = int(block["tag"])
            if instruction["pipeline"] == "xfer":
                destination = instruction["destination"]
                routes_ok = routes_ok and 0 <= destination[0] < mesh_width
                routes_ok = routes_ok and 0 <= destination[1] < mesh_height
                current = list(block["pe"])
                for hop in instruction.get("route") or []:
                    axis = 0 if hop["axis"] == "x" else 1
                    step = int(hop["step"])
                    routes_ok = routes_ok and [hop["from_x"], hop["from_y"]] == current
                    routes_ok = routes_ok and abs(step) in skip_steps
                    residual = destination[axis] - current[axis]
                    admissible = max(item for item in skip_steps if item <= abs(residual))
                    routes_ok = routes_ok and abs(step) == admissible
                    routes_ok = routes_ok and (step > 0) == (residual > 0)
                    current[axis] += step
                    routes_ok = routes_ok and [hop["to_x"], hop["to_y"]] == current
                routes_ok = routes_ok and current == destination
    waits_ok = all(
        event in emitted and producer_tag in {consumer_tag, consumer_tag - 1}
        for consumer_tag, event in waits
        for producer_tag in [emitted.get(event, -999)]
    )
    count_checks = {
        "stages": metadata.get("stages") == expected["stages"],
        "pairs_per_stage": metadata.get("pairs_per_stage") == expected["pairs_per_stage"],
        "total_pairs": metadata.get("total_pairs") == expected["total_pairs"],
        "memory_requests": metadata.get("memory_requests") == expected["memory_requests"],
        "transfers": metadata.get("transfers") == expected["transfers"],
    }
    for key, value in expected.items():
        if key not in count_checks and key not in {"width", "length"}:
            count_checks[key] = metadata.get("operation_counts", {}).get(key) == value
    checks = {
        "memory_backend": document.get("memory_backend") == "dsagen_spad",
        "paper_targets_absent": metadata.get("paper_performance_targets_consumed") is False,
        "block_count": len(blocks) == expected["total_pairs"],
        "static_layout": layout_ok,
        "addresses": addresses_ok,
        "unique_emitters": len(emitted) == expected["total_pairs"],
        "adjacent_wait_events": waits_ok,
        "routes": routes_ok,
        **count_checks,
    }
    return {
        "metadata": metadata,
        "emitted_event_count": len(emitted),
        "wait_event_count": len(waits),
        "checks": checks,
        "pass": all(checks.values()),
    }


def compiler_audit(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = EVIDENCE_ROOT / "mlx-cdc-compiler-manifest.json"
    manifest_file = qualify_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_file["pass"] else {}
    output_specs = {
        "bsmm": (EVIDENCE_ROOT / "mlx-bsmm-b8.json", config["compiler_fixture"]["bsmm"]),
        "fft": (EVIDENCE_ROOT / "mlx-fft-l8.json", config["compiler_fixture"]["fft"]),
    }
    operators: dict[str, Any] = {}
    files: dict[str, Any] = {}
    replay_checks: dict[str, bool] = {}
    for kind, (path, expected) in output_specs.items():
        files[kind] = qualify_file(path)
        document = json.loads(path.read_text(encoding="utf-8")) if files[kind]["pass"] else {}
        operators[kind] = audit_operator(document, expected, kind) if document else {"pass": False}
        replay = EVIDENCE_ROOT / "replay" / path.name
        replay_checks[kind] = replay.is_file() and sha256_file(replay) == files[kind]["sha256"]
    stress_path = EVIDENCE_ROOT / "mlx-bsmm-b16-memory-stress.json"
    stress = qualify_file(stress_path)
    stress_doc = json.loads(stress_path.read_text(encoding="utf-8")) if stress["pass"] else {}
    stress_checks = {
        "stages": stress_doc.get("metadata", {}).get("stages") == 4,
        "pairs_per_stage": stress_doc.get("metadata", {}).get("pairs_per_stage") == 8,
        "memory_requests": stress_doc.get("metadata", {}).get("memory_requests") == 96,
        "eight_initial_blocks": sum(
            block.get("tag") == 1 for block in stress_doc.get("blocks", [])
        )
        == 8,
    }
    replay_manifest = EVIDENCE_ROOT / "replay/mlx-cdc-compiler-manifest.json"
    manifest_replay = replay_manifest.is_file() and manifest_file["pass"] and (
        sha256_file(replay_manifest) == manifest_file["sha256"]
    )
    manifest_bindings: dict[str, bool] = {}
    manifest_outputs = manifest.get("outputs") or {}
    binding_files = {
        "bsmm": files["bsmm"],
        "fft": files["fft"],
        "bsmm_b16_memory_stress": stress,
    }
    for name, artifact in binding_files.items():
        binding = manifest_outputs.get(name) or {}
        manifest_bindings[name] = (
            binding.get("path") == Path(artifact["path"]).name
            and binding.get("bytes") == artifact["bytes"]
            and binding.get("sha256") == artifact["sha256"]
        )
    checks = {
        "manifest": manifest_file["pass"],
        "manifest_no_targets": manifest.get("paper_performance_targets_consumed") is False,
        "operators": all(item["pass"] for item in operators.values()),
        "operator_replays": all(replay_checks.values()),
        "manifest_replay": manifest_replay,
        "manifest_bindings": all(manifest_bindings.values()),
        "stress": stress["pass"] and all(stress_checks.values()),
    }
    return {
        "manifest": manifest_file,
        "files": files,
        "operators": operators,
        "replay_checks": replay_checks,
        "stress_file": stress,
        "stress_checks": stress_checks,
        "manifest_bindings": manifest_bindings,
        "checks": checks,
        "pass": all(checks.values()),
    }


def evaluate_micro_report(report: dict[str, Any]) -> dict[str, Any]:
    scenarios = report.get("scenarios") or []
    by_id = {item.get("id"): item for item in scenarios}
    event = by_id.get("event_counted_cross_layer_overlap", {})
    memory = by_id.get("memory_adapter_backpressure", {})
    count = sum(len(item.get("assertions") or []) + 1 for item in scenarios)
    checks = {
        "integrity": report.get("audit_integrity") is True,
        "scenario_count": report.get("scenario_count") == 2 == len(scenarios),
        "assertion_count": report.get("assertion_count") == 10 == count,
        "all_pass": all(item.get("pass") is True for item in scenarios),
        "deterministic": all(item.get("deterministic_replay") is True for item in scenarios),
        "event_overlap": event.get("summary", {}).get(
            "event_unblocked_issues_before_tag_complete", 0
        )
        >= 2,
        "adapter_counts": memory.get("summary", {}).get("external_memory_requests") == 3
        and memory.get("summary", {}).get("external_memory_completions") == 3,
        "adapter_backpressure": memory.get("summary", {})
        .get("stalls_by_reason", {})
        .get("memory_queue_full", 0)
        >= 1,
        "no_targets": report.get("paper_performance_targets_consumed") is False,
    }
    return {"checks": checks, "pass": all(checks.values())}


def simulator_log(name: str) -> dict[str, Any]:
    path = EVIDENCE_ROOT / name
    artifact = qualify_file(path)
    text = path.read_text(encoding="utf-8", errors="replace") if artifact["pass"] else ""
    return {
        "artifact": artifact,
        "overlay": parse_prefixed_json(text, "MLX_OVERLAY_SUMMARY"),
        "adapter": parse_prefixed_json(text, "MLX_SPAD_ADAPTER_SUMMARY"),
        "dsagen": parse_dsagen_metrics(text),
        "text": text,
    }


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    frozen = {
        name: qualify_file(PROJECT_ROOT / specification["path"], specification)
        for name, specification in config["frozen_inputs"].items()
        if isinstance(specification, dict) and "path" in specification
    }
    h41 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["overlay_result"]["path"]).read_text(
            encoding="utf-8"
        )
    ) if frozen["overlay_result"]["pass"] else {}
    source = source_audit(config)
    compiler = compiler_audit(config)

    debug_report_file = qualify_file(EVIDENCE_ROOT / "cdc-memory-debug-report.json")
    opt_report_file = qualify_file(EVIDENCE_ROOT / "cdc-memory-opt-report.json")
    sanitize_report_file = qualify_file(EVIDENCE_ROOT / "cdc-memory-sanitize-report.json")
    debug_trace = qualify_file(EVIDENCE_ROOT / "cdc-memory-debug-trace.jsonl")
    opt_trace = qualify_file(EVIDENCE_ROOT / "cdc-memory-opt-trace.jsonl")
    sanitize_trace = qualify_file(EVIDENCE_ROOT / "cdc-memory-sanitize-trace.jsonl")
    sanitize_stderr = qualify_file(EVIDENCE_ROOT / "cdc-memory-sanitize-stderr.log")
    debug_report = json.loads(
        (EVIDENCE_ROOT / "cdc-memory-debug-report.json").read_text(encoding="utf-8")
    ) if debug_report_file["pass"] else {}
    opt_report = json.loads(
        (EVIDENCE_ROOT / "cdc-memory-opt-report.json").read_text(encoding="utf-8")
    ) if opt_report_file["pass"] else {}
    sanitize_report = json.loads(
        (EVIDENCE_ROOT / "cdc-memory-sanitize-report.json").read_text(encoding="utf-8")
    ) if sanitize_report_file["pass"] else {}
    micro = evaluate_micro_report(debug_report) if debug_report else {"pass": False}

    bsmm = simulator_log("dsagen-bsmm-b8-spad-smoke.log")
    fft = simulator_log("dsagen-fft-l8-spad-smoke.log")
    stress = simulator_log("dsagen-bsmm-b16-memory-stress.log")
    fixed = simulator_log("fixed-backend-regression.log")
    disabled = simulator_log("disabled-overlay-regression.log")

    def base_pass(item: dict[str, Any]) -> bool:
        metrics = item["dsagen"]
        return (
            metrics["roi_cycles"] == 569
            and metrics["cgra_instances"] == 256
            and metrics["cgra_instructions"] == 1024
            and metrics["sanity_check_passed"]
            and metrics["normal_exit"]
        )

    operator_checks: dict[str, bool] = {}
    for name, item, instructions, compute_instructions in (
        ("bsmm", bsmm, 72, 24),
        ("fft", fft, 84, 36),
    ):
        overlay = item["overlay"] or {}
        adapter = item["adapter"] or {}
        operator_checks[f"{name}_done"] = overlay.get("done") is True
        operator_checks[f"{name}_instructions"] = (
            overlay.get("instructions_issued") == instructions
            and overlay.get("instructions_completed") == instructions
            and overlay.get("issued_by_pipeline", {}).get("compute")
            == compute_instructions
        )
        operator_checks[f"{name}_memory"] = (
            overlay.get("external_memory_requests") == 36
            and overlay.get("external_memory_completions") == 36
            and adapter.get("requests") == 36
            and adapter.get("responses") == 36
            and overlay.get("external_memory_wait_cycles", 0) > 36
            and adapter.get("max_response_cycles", 0) > 1
        )
        operator_checks[f"{name}_events_and_routes"] = (
            overlay.get("boundary_events_emitted") == 12
            and overlay.get("event_unblocked_issues_before_tag_complete", 0) > 0
            and overlay.get("skip_hops") == 12
        )
        operator_checks[f"{name}_base_workload"] = base_pass(item)
    operator_checks["fft_extra_compute_is_visible"] = (
        (fft["overlay"] or {}).get("cycles", 0) > (bsmm["overlay"] or {}).get("cycles", 0)
    )

    stress_overlay = stress["overlay"] or {}
    stress_adapter = stress["adapter"] or {}
    stress_checks = {
        "done": stress_overlay.get("done") is True,
        "request_completion_exact": stress_overlay.get("external_memory_requests") == 96
        and stress_overlay.get("external_memory_completions") == 96
        and stress_adapter.get("requests") == 96
        and stress_adapter.get("responses") == 96,
        "real_queue_backpressure": stress_overlay.get("stalls_by_reason", {}).get(
            "memory_queue_full", 0
        )
        > 0
        and stress_adapter.get("unavailable_checks", 0) > 0,
        "real_response_delay": stress_adapter.get("max_response_cycles", 0) > 1,
        "base_workload": base_pass(stress),
    }

    fixed_overlay = fixed["overlay"] or {}
    regression_checks = {
        "fixed_backend": fixed_overlay.get("memory_backend") == "fixed"
        and fixed_overlay.get("cycles") == 5
        and fixed_overlay.get("external_memory_requests") == 0
        and base_pass(fixed),
        "disabled_overlay": disabled["overlay"] is None and base_pass(disabled),
    }

    build_attempt3 = qualify_file(EVIDENCE_ROOT / "dsagen-gem5-cdc-memory-build-attempt3.log")
    build_attempt4 = qualify_file(EVIDENCE_ROOT / "dsagen-gem5-cdc-memory-build-attempt4.log")
    build_text3 = (
        (EVIDENCE_ROOT / "dsagen-gem5-cdc-memory-build-attempt3.log").read_text(
            encoding="utf-8", errors="replace"
        )
        if build_attempt3["pass"]
        else ""
    )
    build_text4 = (
        (EVIDENCE_ROOT / "dsagen-gem5-cdc-memory-build-attempt4.log").read_text(
            encoding="utf-8", errors="replace"
        )
        if build_attempt4["pass"]
        else ""
    )
    binary = qualify_file(
        PROJECT_ROOT / "third_party/dsa-framework/dsa-gem5/build/RISCV/gem5.opt"
    )
    pass_criteria = {
        "frozen_inputs": all(item["pass"] for item in frozen.values()),
        "h41_supported": h41.get("hypothesis_status")
        == config["frozen_inputs"]["overlay_result"]["required_status"]
        and h41.get("audit_integrity")
        is config["frozen_inputs"]["overlay_result"]["required_integrity"],
        "source": source["pass"],
        "compiler": compiler["pass"],
        "micro_debug": micro.get("pass") is True,
        "micro_optimized": evaluate_micro_report(opt_report).get("pass") is True
        if opt_report
        else False,
        "micro_sanitized": evaluate_micro_report(sanitize_report).get("pass") is True
        if sanitize_report
        else False,
        "micro_byte_identity": debug_report_file.get("sha256")
        == opt_report_file.get("sha256")
        == sanitize_report_file.get("sha256")
        and debug_trace.get("sha256") == opt_trace.get("sha256") == sanitize_trace.get("sha256")
        and sanitize_stderr.get("bytes") == 0,
        "gem5_build": "RISCV/cpu/minor/ssim/mlx_overlay.cc -> .o" in build_text3
        and "RISCV/cpu/minor/ssim/mlx_spad_adapter.cc -> .o" in build_text3
        and "[    LINK]  -> RISCV/gem5.opt" in build_text4
        and "scons: done building targets." in build_text4,
        "operator_runs": all(operator_checks.values()),
        "stress_run": all(stress_checks.values()),
        "regressions": all(regression_checks.values()),
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
        "microtraces": {
            "debug_report": debug_report_file,
            "optimized_report": opt_report_file,
            "sanitized_report": sanitize_report_file,
            "debug_trace": debug_trace,
            "optimized_trace": opt_trace,
            "sanitized_trace": sanitize_trace,
            "sanitizer_stderr": sanitize_stderr,
            "evaluation": micro,
        },
        "simulator_runs": {
            "bsmm_b8": {key: value for key, value in bsmm.items() if key != "text"},
            "fft_l8": {key: value for key, value in fft.items() if key != "text"},
            "bsmm_b16_stress": {
                key: value for key, value in stress.items() if key != "text"
            },
            "fixed_regression": {key: value for key, value in fixed.items() if key != "text"},
            "disabled_regression": {
                key: value for key, value in disabled.items() if key != "text"
            },
            "operator_checks": operator_checks,
            "stress_checks": stress_checks,
            "regression_checks": regression_checks,
        },
        "builds": {"attempt3": build_attempt3, "attempt4": build_attempt4},
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
