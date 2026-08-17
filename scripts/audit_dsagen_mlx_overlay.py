#!/usr/bin/env python3
"""Audit H41's source-integrated DSAGEN MLX overlay and microtraces."""

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
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/dsagen_mlx_overlay_v1.yaml"


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


def git_output(path: Path, *arguments: str) -> str | None:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={path}", *arguments],
        cwd=path,
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
        "overlay_summary_present": "MLX_OVERLAY_SUMMARY" in text,
    }


def parse_overlay_summary(text: str) -> dict[str, Any] | None:
    matches = re.findall(r"^MLX_OVERLAY_SUMMARY (\{.*\})$", text, flags=re.MULTILINE)
    return json.loads(matches[-1]) if matches else None


def evaluate_driver_report(report: dict[str, Any]) -> dict[str, Any]:
    expected_ids = [
        "lower_tag_compute_contention",
        "four_pipeline_overlap",
        "active_window_bound",
        "register_raw_and_bank_pressure",
        "fu_initiation_interval",
        "greedy_skip_hop",
        "adjacent_layer_dependency",
    ]
    scenarios = report.get("scenarios") or []
    actual_ids = [item.get("id") for item in scenarios]
    assertion_count = sum(len(item.get("assertions") or []) + 1 for item in scenarios)
    by_id = {item.get("id"): item for item in scenarios}
    semantic_checks = {
        "tag_priority": by_id.get("lower_tag_compute_contention", {}).get("pass") is True,
        "four_pipeline_overlap": (
            by_id.get("four_pipeline_overlap", {})
            .get("summary", {})
            .get("max_pipeline_issues_in_cycle")
            == 4
        ),
        "active_window": (
            by_id.get("active_window_bound", {})
            .get("summary", {})
            .get("max_active_tags")
            == 3
        ),
        "register_hazards": by_id.get("register_raw_and_bank_pressure", {}).get("pass")
        is True,
        "fu_initiation": (
            by_id.get("fu_initiation_interval", {})
            .get("summary", {})
            .get("stalls_by_reason", {})
            .get("fu_initiation", 0)
            >= 1
        ),
        "skip_hop_and_link_contention": (
            by_id.get("greedy_skip_hop", {}).get("summary", {}).get("skip_hops") == 1
            and by_id.get("greedy_skip_hop", {}).get("summary", {}).get("link_stalls")
            == 1
        ),
        "adjacent_dependency": by_id.get("adjacent_layer_dependency", {}).get("pass")
        is True,
    }
    checks = {
        "schema_version": report.get("schema_version") == 1,
        "audit_integrity": report.get("audit_integrity") is True,
        "scenario_ids": actual_ids == expected_ids,
        "scenario_count": report.get("scenario_count") == 7 == len(scenarios),
        "assertion_count": report.get("assertion_count") == 25 == assertion_count,
        "all_scenarios_pass": all(item.get("pass") is True for item in scenarios),
        "all_replays_deterministic": all(
            item.get("deterministic_replay") is True for item in scenarios
        ),
        "no_paper_targets": report.get("paper_target_values_consumed") is False,
        **semantic_checks,
    }
    return {
        "scenario_ids": actual_ids,
        "assertion_count_recomputed": assertion_count,
        "semantic_checks": semantic_checks,
        "checks": checks,
        "pass": all(checks.values()),
    }


def qualify_source(config: dict[str, Any]) -> dict[str, Any]:
    layout = config["source_layout"]
    token_map = {
        "implementation_header": [
            "class Overlay",
            "struct TaggedBlock",
            "enum class PipelineKind",
            "GreedyRoute",
            "active_window",
            "pending_writers_",
        ],
        "implementation_source": [
            "Overlay::step()",
            "Overlay::admitTags()",
            "Overlay::issueInstructions()",
            "Overlay::routePackets()",
            "fu_next_issue_",
            "link_capacity",
        ],
        "integration_header": ["_mlx_overlay", "_mlx_summary_printed"],
        "integration_source": [
            'std::getenv("MLX_CONFIG")',
            "Overlay::FromJsonFile",
            "_mlx_overlay->step()",
            "MLX_OVERLAY_SUMMARY",
        ],
        "build_manifest": ["Source('ssim/mlx_overlay.cc')"],
        "standalone_driver": [
            "LowerTagContention",
            "FourPipelineOverlap",
            "RegisterHazards",
            "GreedySkipHop",
        ],
    }
    reports: dict[str, Any] = {}
    for key, tokens in token_map.items():
        path = PROJECT_ROOT / layout[key]
        text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
        token_checks = {token: token in text for token in tokens}
        reports[key] = {
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
    reverse_check = subprocess.run(
        ["git", "apply", "--check", "--reverse", str(patch_path)],
        cwd=gem5_root,
        check=False,
        capture_output=True,
        text=True,
    ) if patch_path.is_file() else None
    patch_checks = {
        "is_file": patch["pass"],
        "reverse_applies_to_worktree": reverse_check is not None and reverse_check.returncode == 0,
        "forbidden_gpu_state_absent": not forbidden,
    }
    return {
        "files": reports,
        "patch": {
            **patch,
            "forbidden_gpu_tokens": forbidden,
            "reverse_check_stderr": reverse_check.stderr if reverse_check is not None else None,
            "checks": patch_checks,
            "pass": all(patch_checks.values()),
        },
        "pass": all(item["pass"] for item in reports.values()) and all(patch_checks.values()),
    }


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    frozen = {
        name: qualify_file(PROJECT_ROOT / specification["path"], specification)
        for name, specification in config["frozen_inputs"].items()
        if isinstance(specification, dict) and "path" in specification
    }
    substrate = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["substrate_audit"]["path"]).read_text(
            encoding="utf-8"
        )
    ) if frozen["substrate_audit"]["pass"] else {}
    source = qualify_source(config)

    evidence_root = PROJECT_ROOT / "artifacts/environment/h41"
    debug_report_file = qualify_file(evidence_root / "mlx-overlay-debug-report.json")
    opt_report_file = qualify_file(evidence_root / "mlx-overlay-opt-report.json")
    sanitize_report_file = qualify_file(evidence_root / "mlx-overlay-sanitize-report.json")
    debug_trace = qualify_file(evidence_root / "mlx-overlay-debug-trace.jsonl")
    opt_trace = qualify_file(evidence_root / "mlx-overlay-opt-trace.jsonl")
    sanitize_trace = qualify_file(evidence_root / "mlx-overlay-sanitize-trace.jsonl")
    sanitize_stderr = qualify_file(evidence_root / "mlx-overlay-sanitize-stderr.log")
    debug_report = json.loads(
        (evidence_root / "mlx-overlay-debug-report.json").read_text(encoding="utf-8")
    ) if debug_report_file["pass"] else {}
    opt_report = json.loads(
        (evidence_root / "mlx-overlay-opt-report.json").read_text(encoding="utf-8")
    ) if opt_report_file["pass"] else {}
    sanitize_report = json.loads(
        (evidence_root / "mlx-overlay-sanitize-report.json").read_text(encoding="utf-8")
    ) if sanitize_report_file["pass"] else {}
    driver = evaluate_driver_report(debug_report) if debug_report else {"pass": False}
    build_logs = {
        "driver_debug": qualify_file(
            evidence_root / "mlx-overlay-driver-debug-build-attempt2.log"
        ),
        "driver_optimized": qualify_file(
            evidence_root / "mlx-overlay-driver-opt-build-attempt1.log"
        ),
        "driver_sanitized": qualify_file(
            evidence_root / "mlx-overlay-driver-sanitize-build-attempt1.log"
        ),
        "gem5_failure": qualify_file(
            evidence_root / "dsagen-gem5-mlx-overlay-build-attempt1.log"
        ),
        "gem5_pre_round_robin": qualify_file(
            evidence_root / "dsagen-gem5-mlx-overlay-build-attempt2.log"
        ),
        "gem5_success": qualify_file(
            evidence_root / "dsagen-gem5-mlx-overlay-build-attempt3.log"
        ),
    }
    gem5_build_text = (
        (evidence_root / "dsagen-gem5-mlx-overlay-build-attempt3.log").read_text(
            encoding="utf-8", errors="replace"
        )
        if build_logs["gem5_success"]["pass"]
        else ""
    )

    enabled_file = qualify_file(evidence_root / "dsagen-overlay-enabled-smoke.log")
    disabled_file = qualify_file(evidence_root / "dsagen-overlay-disabled-regression.log")
    enabled_text = (
        (evidence_root / "dsagen-overlay-enabled-smoke.log").read_text(
            encoding="utf-8", errors="replace"
        )
        if enabled_file["pass"]
        else ""
    )
    disabled_text = (
        (evidence_root / "dsagen-overlay-disabled-regression.log").read_text(
            encoding="utf-8", errors="replace"
        )
        if disabled_file["pass"]
        else ""
    )
    enabled_metrics = parse_dsagen_metrics(enabled_text)
    disabled_metrics = parse_dsagen_metrics(disabled_text)
    overlay_summary = parse_overlay_summary(enabled_text)
    expected_regression = config["regression_gate"]["expected_metrics"]
    regression_canonical = {
        key: disabled_metrics[key] for key in expected_regression
    }
    enabled_checks = {
        "summary_present": overlay_summary is not None,
        "overlay_done": overlay_summary is not None and overlay_summary.get("done") is True,
        "four_instructions": overlay_summary is not None
        and overlay_summary.get("instructions_issued") == 4
        and overlay_summary.get("instructions_completed") == 4,
        "window_three": overlay_summary is not None
        and overlay_summary.get("max_active_tags") == 3,
        "four_pipeline_issue": overlay_summary is not None
        and overlay_summary.get("max_pipeline_issues_in_cycle") == 4,
        "all_pipeline_classes": overlay_summary is not None
        and overlay_summary.get("issued_by_pipeline")
        == {"load": 1, "store": 1, "compute": 1, "xfer": 1},
        "underlying_dsagen_passes": enabled_metrics["sanity_check_passed"]
        and enabled_metrics["normal_exit"],
    }
    disabled_checks = {
        "overlay_absent": not disabled_metrics["overlay_summary_present"],
        "expected_metrics": regression_canonical == expected_regression,
        "normal_exit": disabled_metrics["normal_exit"],
    }
    binary = qualify_file(
        PROJECT_ROOT / "third_party/dsa-framework/dsa-gem5/build/RISCV/gem5.opt"
    )
    pass_criteria = {
        "frozen_inputs": all(item["pass"] for item in frozen.values()),
        "h40_supported": substrate.get("hypothesis_status")
        == config["frozen_inputs"]["substrate_audit"]["required_status"]
        and substrate.get("audit_integrity")
        is config["frozen_inputs"]["substrate_audit"]["required_integrity"],
        "source_integrated": source["pass"],
        "debug_driver": driver.get("pass") is True,
        "optimized_driver": evaluate_driver_report(opt_report).get("pass") is True
        if opt_report
        else False,
        "sanitized_driver": evaluate_driver_report(sanitize_report).get("pass") is True
        if sanitize_report
        else False,
        "debug_optimized_trace_identical": debug_trace.get("sha256")
        == opt_trace.get("sha256")
        and debug_trace["pass"]
        and opt_trace["pass"],
        "sanitized_trace_identical_and_clean": sanitize_trace.get("sha256")
        == debug_trace.get("sha256")
        and sanitize_trace["pass"]
        and sanitize_stderr["pass"]
        and sanitize_stderr["bytes"] == 0,
        "debug_optimized_report_identical": debug_report_file.get("sha256")
        == opt_report_file.get("sha256")
        and debug_report_file["pass"]
        and opt_report_file["pass"],
        "gem5_object_compiled_and_linked": "RISCV/cpu/minor/ssim/mlx_overlay.cc -> .o"
        in gem5_build_text
        and "[    LINK]  -> RISCV/gem5.opt" in gem5_build_text
        and "scons: done building targets." in gem5_build_text,
        "gem5_overlay_enabled": all(enabled_checks.values()),
        "gem5_overlay_disabled_regression": all(disabled_checks.values()),
        "gem5_binary_present": binary["pass"],
        "paper_target_values_consumed": False,
    }
    integrity = all(
        value for key, value in pass_criteria.items() if key != "paper_target_values_consumed"
    ) and pass_criteria["paper_target_values_consumed"] is False
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_commit": git_output(PROJECT_ROOT, "rev-parse", "HEAD"),
        "hypothesis_status": "supported" if integrity else "rejected",
        "audit_integrity": integrity,
        "frozen_inputs": frozen,
        "source_integration": source,
        "driver": {
            "debug_report": debug_report_file,
            "optimized_report": opt_report_file,
            "sanitized_report": sanitize_report_file,
            "debug_trace": debug_trace,
            "optimized_trace": opt_trace,
            "sanitized_trace": sanitize_trace,
            "sanitizer_stderr": sanitize_stderr,
            "evaluation": driver,
        },
        "build_logs": build_logs,
        "gem5_binary": binary,
        "gem5_enabled": {
            "evidence": enabled_file,
            "dsagen_metrics": enabled_metrics,
            "overlay_summary": overlay_summary,
            "checks": enabled_checks,
            "pass": all(enabled_checks.values()),
        },
        "gem5_disabled_regression": {
            "evidence": disabled_file,
            "metrics": disabled_metrics,
            "canonical_metrics": regression_canonical,
            "checks": disabled_checks,
            "pass": all(disabled_checks.values()),
        },
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
