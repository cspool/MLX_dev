#!/usr/bin/env python3
"""Audit H52's paper-static MLX PE semantics correction."""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from scripts.audit_dsagen_mlx_dma_memory import (
    git_revision,
    load_yaml,
    parse_prefixed_json,
    parse_stats,
    qualify_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/mlx_pe_semantics_correction_v1.yaml"
EVIDENCE_ROOT = PROJECT_ROOT / "artifacts/environment/h52"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--verify-existing", action="store_true")
    return parser.parse_args()


def read_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_gem5(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "artifact": qualify_file(path),
        "overlay": parse_prefixed_json(text, "MLX_OVERLAY_SUMMARY"),
        "adapter": parse_prefixed_json(text, "MLX_DMA_ADAPTER_SUMMARY"),
        "guest": parse_prefixed_json(text, "MLX_DMA_GUEST_SUMMARY"),
        "normal_exit": "exiting with last active thread context" in text,
        "sanity": "sanity check passed successfully!" in text,
    }


def compile_audit() -> dict[str, Any]:
    parent_root = PROJECT_ROOT / "artifacts/environment/h48"
    manifest_path = EVIDENCE_ROOT / "mlx-pe-semantics-compile-manifest.json"
    replay_path = EVIDENCE_ROOT / "compiler-replay-check.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    checks = {}
    files = {}
    for name in ("fixed", "dma"):
        parent_path = parent_root / f"mlx-full-block-{name}.json"
        output_path = EVIDENCE_ROOT / f"mlx-full-block-{name}.json"
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        output = json.loads(output_path.read_text(encoding="utf-8"))
        stripped = copy.deepcopy(output)
        model = stripped.pop("pe_dependency_model", None)
        metadata = stripped["metadata"]
        metadata_model = metadata.pop("pe_dependency_model", None)
        scoreboard_claim = metadata.pop("scoreboard_is_paper_semantics", None)
        files[name] = {
            "parent": qualify_file(parent_path),
            "output": qualify_file(output_path),
        }
        checks[name] = (
            stripped == parent
            and model == "paper_static"
            and metadata_model == "paper_static"
            and scoreboard_claim is False
            and output["metadata"]["paper_performance_targets_consumed"] is False
        )
    checks["manifest"] = manifest.get("pe_dependency_model") == "paper_static"
    checks["replay"] = replay.get("all_identical") is True and all(
        item.get("identical") is True for item in replay.get("comparisons", {}).values()
    )
    return {
        "files": files,
        "manifest": qualify_file(manifest_path),
        "replay": qualify_file(replay_path),
        "checks": checks,
        "pass": all(checks.values()),
    }


def source_audit(config: dict[str, Any]) -> dict[str, Any]:
    layout = config["source_layout"]
    header = PROJECT_ROOT / layout["overlay_header"]
    source = PROJECT_ROOT / layout["overlay_source"]
    header_text = header.read_text(encoding="utf-8", errors="replace")
    source_text = source.read_text(encoding="utf-8", errors="replace")
    paper_path = PROJECT_ROOT / config["frozen_inputs"]["paper"]["path"]
    paper_text = paper_path.read_text(encoding="utf-8", errors="replace")
    source_checks = {
        "models": "enum class PeDependencyModel" in header_text
        and "PaperStatic" in header_text
        and "ScoreboardExperimental" in header_text,
        "parser": 'name == "paper_static"' in source_text
        and '"scoreboard_experimental"' in source_text,
        "conditional_issue": source_text.count("usesExperimentalScoreboard()") >= 5,
        "summary": '"paper_static" : "scoreboard_experimental"' in source_text,
    }
    paper_checks = {
        "avoids_fine_hazards": "Rather than tracking these interactions with fine-grained instruction-level hazards"
        in paper_text,
        "static_local_schedule": "software fixes the deterministic local schedule" in paper_text,
        "frontier": re.search(r"frontier\*?\s+instruction", paper_text) is not None,
        "lower_tag": "arbiter grants the lower tag" in paper_text,
        "four_pipelines": "independent pipelines" in paper_text,
    }
    patch_path = PROJECT_ROOT / layout["tracked_patch"]
    gem5_root = PROJECT_ROOT / "third_party/dsa-framework/dsa-gem5"
    reverse = subprocess.run(
        ["git", "apply", "--check", "--reverse", str(patch_path)],
        cwd=gem5_root,
        check=False,
        capture_output=True,
        text=True,
    )
    patch_checks = {
        "present": patch_path.is_file(),
        "reverse_applies": reverse.returncode == 0,
    }
    return {
        "paper": qualify_file(paper_path, config["frozen_inputs"]["paper"]),
        "header": qualify_file(header),
        "source": qualify_file(source),
        "patch": qualify_file(patch_path),
        "source_checks": source_checks,
        "paper_checks": paper_checks,
        "patch_checks": patch_checks,
        "pass": all(source_checks.values())
        and all(paper_checks.values())
        and all(patch_checks.values()),
    }


def standalone_audit() -> dict[str, Any]:
    root = EVIDENCE_ROOT / "runs/standalone"
    summary_files = {
        name: qualify_file(root / f"{name}-summary.json")
        for name in ("debug", "opt", "sanitize")
    }
    summaries = {
        name: read_summary(root / f"{name}-summary.json") for name in summary_files
    }
    summary = summaries["debug"]
    hazard_keys = {
        "register_pending",
        "register_waw",
        "rf_read_ports",
        "rf_read_bank",
        "rf_write_ports",
        "rf_write_bank",
    }
    trace_hashes = [
        line.split()[0]
        for line in (root / "trace-sha256.txt").read_text(encoding="utf-8").splitlines()
    ]
    legacy = read_summary(EVIDENCE_ROOT / "legacy/summary.json")
    checks = {
        "summaries": all(item["pass"] for item in summary_files.values()),
        "identity": len({item["sha256"] for item in summary_files.values()}) == 1,
        "traces": len(trace_hashes) == 3 and len(set(trace_hashes)) == 1,
        "sanitize": qualify_file(root / "sanitize-stderr.log")["bytes"] == 0,
        "paper_model": summary.get("pe_dependency_model") == "paper_static",
        "cycles": summary.get("cycles") == 293,
        "counts": summary.get("instructions_issued")
        == summary.get("instructions_completed")
        == 1352
        and summary.get("boundary_events_emitted") == 480
        and summary.get("route_hops") == 576,
        "no_hazard_stalls": not hazard_keys.intersection(
            summary.get("stalls_by_reason", {})
        ),
        "only_paper_stalls": set(summary.get("stalls_by_reason", {}))
        == {"event_dependency", "pipeline_contention"},
        "legacy_parseable": legacy.get("pe_dependency_model")
        == "scoreboard_experimental"
        and legacy.get("cycles") == 393
        and bool(hazard_keys.intersection(legacy.get("stalls_by_reason", {}))),
    }
    return {
        "summary_files": summary_files,
        "summary": summary,
        "legacy": {
            "artifact": qualify_file(EVIDENCE_ROOT / "legacy/summary.json"),
            "summary": legacy,
        },
        "checks": checks,
        "pass": all(bool(value) for value in checks.values()),
    }


def gem5_audit() -> dict[str, Any]:
    fixed = read_gem5(EVIDENCE_ROOT / "runs/gem5/fixed/run.log")
    dma = read_gem5(EVIDENCE_ROOT / "runs/gem5/dma/run.log")
    fixed_summary = fixed["overlay"] or {}
    dma_summary = dma["overlay"] or {}
    adapter = dma["adapter"] or {}
    hazard_pattern = re.compile(r"^(register_|rf_)")
    stats_path = EVIDENCE_ROOT / "runs/gem5/dma/m5out/stats.txt"
    stats = parse_stats(stats_path)
    fixed_checks = {
        "model": fixed_summary.get("pe_dependency_model") == "paper_static",
        "cycles": fixed_summary.get("cycles") == 293,
        "counts": fixed_summary.get("instructions_issued")
        == fixed_summary.get("instructions_completed")
        == 1352,
        "no_hazards": not any(
            hazard_pattern.match(key) for key in fixed_summary.get("stalls_by_reason", {})
        ),
        "checksum": (fixed["guest"] or {}).get("store_checksum") == 84480,
        "exit": fixed["sanity"] and fixed["normal_exit"],
    }
    dma_checks = {
        "model": dma_summary.get("pe_dependency_model") == "paper_static",
        "cycles": dma_summary.get("cycles") == 1025,
        "counts": dma_summary.get("instructions_issued")
        == dma_summary.get("instructions_completed")
        == 1352,
        "events_routes": dma_summary.get("boundary_events_emitted") == 480
        and dma_summary.get("route_hops") == 576,
        "memory": dma_summary.get("external_memory_requests")
        == dma_summary.get("external_memory_completions")
        == adapter.get("requests")
        == adapter.get("responses")
        == 40,
        "directions": adapter.get("read_requests") == adapter.get("read_responses") == 24
        and adapter.get("write_requests") == adapter.get("write_responses") == 16,
        "completion": adapter.get("failed_responses") == 0
        and adapter.get("outstanding") == 0,
        "no_hazards": not any(
            hazard_pattern.match(key) for key in dma_summary.get("stalls_by_reason", {})
        ),
        "dram": stats.get("system.mem_ctrls.num_reads::.cpu.mlx_dma") == 24
        and stats.get("system.mem_ctrls.bytes_read::.cpu.mlx_dma") == 1536,
        "checksum": (dma["guest"] or {}).get("store_checksum") == 63360,
        "exit": dma["sanity"] and dma["normal_exit"],
    }
    return {
        "fixed": {**fixed, "checks": fixed_checks, "pass": all(fixed_checks.values())},
        "dma": {**dma, "checks": dma_checks, "pass": all(dma_checks.values())},
        "stats": qualify_file(stats_path),
        "pass": all(fixed_checks.values()) and all(dma_checks.values()),
    }


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    parent_spec = config["frozen_inputs"]["full_block_result"]
    parent_artifact = qualify_file(PROJECT_ROOT / parent_spec["path"], parent_spec)
    parent = json.loads((PROJECT_ROOT / parent_spec["path"]).read_text(encoding="utf-8"))
    compiler = compile_audit()
    source = source_audit(config)
    standalone = standalone_audit()
    gem5 = gem5_audit()
    binary = qualify_file(
        PROJECT_ROOT / "third_party/dsa-framework/dsa-gem5/build/RISCV/gem5.opt"
    )
    checks = {
        "parent": parent_artifact["pass"]
        and parent.get("hypothesis_status") == parent_spec["required_status"]
        and parent.get("audit_integrity") is parent_spec["required_integrity"],
        "compiler": compiler["pass"],
        "source": source["pass"],
        "standalone": standalone["pass"],
        "gem5": gem5["pass"],
        "binary": binary["pass"],
        "no_targets": True,
    }
    integrity = all(checks.values())
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "classification": config["classification"],
        "hypothesis_status": "supported" if integrity else "rejected",
        "audit_integrity": integrity,
        "git_revision": git_revision(PROJECT_ROOT),
        "parent": parent_artifact,
        "compiler": compiler,
        "source": source,
        "standalone": standalone,
        "gem5": gem5,
        "binary": binary,
        "integrity_checks": checks,
        "architectural_correction": {
            "old_shorthand": "GPU-SM-like programmable PE",
            "new_contract": "static tagged-block spatial PE with tag-level arbitration",
            "gpgpu_sim_role": "independent GPU baseline only",
        },
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
            raise SystemExit("existing H52 result does not match a fresh audit")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            {
                "hypothesis_status": report["hypothesis_status"],
                "audit_integrity": report["audit_integrity"],
                "fixed_cycles": report["gem5"]["fixed"]["overlay"]["cycles"],
                "dma_cycles": report["gem5"]["dma"]["overlay"]["cycles"],
            },
            indent=2,
        )
    )
    return 0 if report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
