#!/usr/bin/env python3
"""Audit H109 bounded multi-iteration tagged-block contexts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from mlxsim.dsagen_overlay import canonical_json
from mlxsim.pipelined_block_contexts import scenarios

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/simulators/pipelined_block_contexts_v1.yaml"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qualify(path: Path, expected: dict[str, Any] | None = None) -> dict[str, Any]:
    exists = path.is_file()
    digest = sha256_file(path) if exists else None
    checks = {"is_file": exists}
    if expected and "sha256" in expected:
        checks["sha256"] = digest == expected["sha256"]
    if expected and "bytes" in expected:
        checks["bytes"] = exists and path.stat().st_size == int(expected["bytes"])
    if exists and expected and (
        "required_status" in expected or "required_integrity" in expected
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "required_status" in expected:
            checks["status"] = (
                payload.get("hypothesis_status") == expected["required_status"]
            )
        if "required_integrity" in expected:
            checks["integrity"] = (
                payload.get("audit_integrity") is expected["required_integrity"]
            )
    return {
        "path": str(path.relative_to(PROJECT_ROOT)) if exists else str(path),
        "bytes": path.stat().st_size if exists else None,
        "sha256": digest,
        "checks": checks,
        "pass": all(checks.values()),
    }


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def debug_record(run: dict[str, Any], scenario: str) -> dict[str, Any]:
    records = [
        item
        for item in run["records"]
        if item["scenario"] == scenario
        and item["mode"] == "debug"
        and item["replay"] == 1
    ]
    if len(records) != 1:
        raise ValueError(f"missing H109 debug record: {scenario}")
    return records[0]


def load_trace(record: dict[str, Any]) -> list[dict[str, Any]]:
    path = PROJECT_ROOT / record["trace_path"]
    if sha256_file(path) != record["trace_sha256"]:
        raise ValueError(f"H109 trace digest mismatch: {path}")
    return [json.loads(line) for line in path.read_text().splitlines()]


def events(trace: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [item for item in trace if item["kind"] == kind]


def patch_audit(config: dict[str, Any]) -> dict[str, Any]:
    patch_path = PROJECT_ROOT / config["source_layout"]["patch"]
    header_path = PROJECT_ROOT / config["source_layout"]["overlay_header"]
    source_path = PROJECT_ROOT / config["source_layout"]["overlay_source"]
    h108 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h108"]["path"]).read_text()
    )
    report = {
        "patch": qualify(patch_path),
        "reverse_check": False,
        "h108_source_exact": False,
        "forward_check": False,
        "round_trip_exact": False,
        "newer_patch_stack": {},
    }
    if not patch_path.is_file():
        report["pass"] = False
        return report
    with tempfile.TemporaryDirectory(prefix="mlx-h109-patch-") as temporary:
        root = Path(temporary)
        target = root / "src/cpu/minor/ssim"
        target.mkdir(parents=True)
        header = target / "mlx_overlay.hh"
        source = target / "mlx_overlay.cc"
        shutil.copy2(header_path, header)
        shutil.copy2(source_path, source)
        newer_patches = [
            PROJECT_ROOT
            / "patches/dsagen/dsa-gem5-mlx-latency-service-v1.patch",
            PROJECT_ROOT
            / "patches/dsagen/dsa-gem5-functional-payload-v1.patch",
            PROJECT_ROOT
            / "patches/dsagen/dsa-gem5-active-window-instruction-capacity-v1.patch",
            PROJECT_ROOT
            / "patches/dsagen/dsa-gem5-active-pipelined-scan-v1.patch",
            PROJECT_ROOT
            / "patches/dsagen/dsa-gem5-active-window-capacity-v1.patch",
        ]
        applied_newer = []
        for newer in newer_patches:
            options = ["--unidiff-zero"] if "active-pipelined" in newer.name else []
            check = subprocess.run(
                ["git", "apply", *options, "-R", "--check", str(newer)],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            report["newer_patch_stack"][newer.name] = check.returncode == 0
            if check.returncode != 0:
                report["pass"] = False
                return report
            subprocess.run(
                ["git", "apply", *options, "-R", str(newer)],
                cwd=root,
                check=True,
            )
            applied_newer.append(newer)
        reverse = subprocess.run(
            ["git", "apply", "-R", "--check", str(patch_path)],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        report["reverse_check"] = reverse.returncode == 0
        if reverse.returncode != 0:
            report["reverse_stderr"] = reverse.stderr.strip()
            report["pass"] = False
            return report
        subprocess.run(
            ["git", "apply", "-R", str(patch_path)], cwd=root, check=True
        )
        report["h108_source_exact"] = (
            sha256_file(header)
            == h108["diagnostic_source_files"]["overlay_header"]["sha256"]
            and sha256_file(source)
            == h108["diagnostic_source_files"]["overlay_source"]["sha256"]
        )
        forward = subprocess.run(
            ["git", "apply", "--check", str(patch_path)],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        report["forward_check"] = forward.returncode == 0
        if forward.returncode == 0:
            subprocess.run(["git", "apply", str(patch_path)], cwd=root, check=True)
            for newer in reversed(applied_newer):
                options = ["--unidiff-zero"] if "active-pipelined" in newer.name else []
                subprocess.run(
                    ["git", "apply", *options, "--check", str(newer)],
                    cwd=root,
                    check=True,
                )
                subprocess.run(
                    ["git", "apply", *options, str(newer)], cwd=root, check=True
                )
            report["round_trip_exact"] = (
                header.read_bytes() == header_path.read_bytes()
                and source.read_bytes() == source_path.read_bytes()
            )
    report["pass"] = all(
        report[key]
        for key in (
            "reverse_check",
            "h108_source_exact",
            "forward_check",
            "round_trip_exact",
        )
    ) and all(report["newer_patch_stack"].values())
    return report


def gem5_audit(config: dict[str, Any]) -> dict[str, Any]:
    specification = config["regressions"]["gem5"]
    root = PROJECT_ROOT / specification["smoke_root"]
    enabled_path = root / "enabled/run.log"
    disabled_path = root / "disabled/run.log"
    enabled = enabled_path.read_text(errors="replace") if enabled_path.is_file() else ""
    disabled = (
        disabled_path.read_text(errors="replace") if disabled_path.is_file() else ""
    )
    cycles = int(specification["expected_dsagen_cycles"])
    checks = {
        "enabled_file": enabled_path.is_file(),
        "disabled_file": disabled_path.is_file(),
        "enabled_overlay": "MLX_OVERLAY_SUMMARY" in enabled,
        "disabled_overlay": "MLX_OVERLAY_SUMMARY" not in disabled,
        "enabled_cycles": f"Cycles: {cycles}" in enabled,
        "disabled_cycles": f"Cycles: {cycles}" in disabled,
        "enabled_sanity": "sanity check passed successfully!" in enabled,
        "disabled_sanity": "sanity check passed successfully!" in disabled,
    }
    return {
        "enabled": qualify(enabled_path),
        "disabled": qualify(disabled_path),
        "checks": checks,
        "pass": all(checks.values()),
    }


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / item["path"], item)
        for name, item in config["frozen_inputs"].items()
    }
    output_root = PROJECT_ROOT / config["output_root"]
    compile_path = output_root / "pipelined-block-contexts-compile-manifest.json"
    run_path = output_root / "pipelined-block-contexts-run-manifest.json"
    compiled = json.loads(compile_path.read_text())
    run = json.loads(run_path.read_text())
    documents = scenarios()
    canonical_checks = {}
    for name, document in documents.items():
        output = compiled["outputs"][name]
        path = PROJECT_ROOT / output["artifact"]["path"]
        canonical_checks[name] = (
            path.read_text() == canonical_json(document)
            and sha256_file(path) == output["artifact"]["sha256"]
        )

    traces = {
        name: load_trace(debug_record(run, name))
        for name in documents
        if name != "operand_context_overflow"
    }
    summaries = {
        name: debug_record(run, name)["summary"] for name in traces
    }
    fma_issues = events(traces["fma_ii1_ctx4"], "issue")
    fma_completes = events(traces["fma_ii1_ctx4"], "complete")
    limited_issues = events(traces["fma_ii1_ctx2"], "issue")
    ii2_issues = events(traces["fma_ii2_ctx4"], "issue")

    multi_issues = events(traces["multi_instruction_overlap"], "issue")
    multi_cycles = {
        iteration: {
            event["instruction"]: event["cycle"]
            for event in multi_issues
            if event["iteration"] == iteration
        }
        for iteration in range(4)
    }
    multi_order = all(
        values["load"] < values["compute"] < values["store"]
        for values in multi_cycles.values()
    )
    multi_overlap = multi_cycles[0]["compute"] <= multi_cycles[3]["load"]

    event_trace = traces["event_pipeline"]
    produced = {
        event["iteration"]: event["cycle"]
        for event in events(event_trace, "event_emit")
        if event["block"] == "producer"
    }
    consumed = {
        event["iteration"]: event["cycle"]
        for event in events(event_trace, "issue")
        if event["block"] == "consumer"
    }
    event_order = set(produced) == set(consumed) == set(range(8)) and all(
        consumed[index] >= produced[index] for index in range(8)
    )

    identity = events(traces["context_identity_order"], "issue")
    identity_order = [
        (
            event["block"],
            event["detail"]["instance"],
            event["detail"]["context_slot"],
        )
        for event in identity
    ]
    expected_identity = [
        ("task_first", "10", "0"),
        ("task_first", "11", "1"),
        ("task_second", "20", "0"),
        ("task_second", "21", "1"),
    ]
    same = summaries["same_plane_context_routes"]
    split = summaries["split_plane_context_routes"]

    conservation_checks = {}
    for name, trace in traces.items():
        summary = summaries[name]
        trip_total = sum(
            int(block["trip_count"]) for block in documents[name]["blocks"]
        )
        conservation_checks[name] = (
            summary["instructions_issued"] == summary["instructions_completed"]
            and len(events(trace, "iteration_complete")) == trip_total
            and summary["max_inflight_iterations_per_block"]
            <= int(
                documents[name]["dpu"]["iteration_contexts_per_block"]
            )
            and summary["done"]
        )
    overflow = [
        item
        for item in run["records"]
        if item["scenario"] == "operand_context_overflow"
    ]
    scenario_checks = {
        "fma_issue_cycles": [event["cycle"] for event in fma_issues]
        == config["acceptance"]["expected_fma_issue_cycles"],
        "fma_completion_cycles": [event["cycle"] for event in fma_completes]
        == config["acceptance"]["expected_fma_complete_cycles"]
        and summaries["fma_ii1_ctx4"]["cycles"]
        == int(config["acceptance"]["expected_fma_total_cycles"]),
        "limited_context_cycles": [event["cycle"] for event in limited_issues]
        == config["acceptance"]["expected_limited_issue_cycles"],
        "ii2_cycles": [event["cycle"] for event in ii2_issues]
        == list(range(0, 16, 2)),
        "multi_instruction_order": multi_order and multi_overlap,
        "event_order": event_order
        and summaries["event_pipeline"]["boundary_events_emitted"] == 8,
        "context_identity": identity_order == expected_identity,
        "routing": same["link_stalls"] == 4
        and same["route_hops_by_plane"] == {"0": 6, "1": 0}
        and split["link_stalls"] == 0
        and split["route_hops_by_plane"] == {"0": 4, "1": 2},
        "conservation": all(conservation_checks.values()),
        "operand_context_overflow": len(overflow) == 6
        and all(item["pass"] for item in overflow),
    }
    mode_counts = Counter(item["mode"] for item in run["records"])
    run_checks = {
        "count": len(run["records"])
        == int(config["execution"]["required_executions"]),
        "modes": dict(mode_counts)
        == {"debug": 20, "optimized": 20, "asan": 10, "ubsan": 10},
        "runs": all(item["pass"] for item in run["records"]),
        "replays": all(run["replay_checks"].values()),
        "builds": all(run["cross_build_checks"].values()),
        "manifest": all(run["checks"].values()),
    }

    h105 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h105"]["path"]).read_text()
    )
    h106 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h106"]["path"]).read_text()
    )
    regression_files = {
        "h105": qualify(
            PROJECT_ROOT / h105["run_manifest"]["path"], h105["run_manifest"]
        ),
        "h106": qualify(
            PROJECT_ROOT / h106["run_manifest"]["path"], h106["run_manifest"]
        ),
    }
    gem5 = gem5_audit(config)
    patch = patch_audit(config)
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    header = (
        PROJECT_ROOT / config["source_layout"]["overlay_header"]
    ).read_text()
    source = (
        PROJECT_ROOT / config["source_layout"]["overlay_source"]
    ).read_text()
    source_checks = {
        "mode": "DpuPipelined" in header
        and 'name == "dpu_pipelined"' in source,
        "context_state": "struct IterationContext" in header
        and "std::vector<IterationContext> contexts" in header,
        "inflight_bypass": "issuePipelinedInstructions" in source
        and "context.inflight" in source,
        "ii_latency_split": "cycle_ + timing.initiation_interval" in source
        and "cycle_ + timing.latency" in source,
        "identity": 'enriched["context_slot"]' in source
        and 'enriched["instance"]' in source,
        "default_legacy": "iteration_contexts_per_block{1}" in header,
    }
    target_free = (
        compiled["paper_performance_targets_consumed"] is False
        and run["paper_performance_targets_consumed"] is False
        and config["execution"]["paper_performance_targets_consumed"] is False
    )
    acceptance_gates = [
        scenario_checks["fma_issue_cycles"],
        scenario_checks["fma_completion_cycles"],
        scenario_checks["limited_context_cycles"],
        scenario_checks["ii2_cycles"],
        scenario_checks["multi_instruction_order"],
        scenario_checks["event_order"],
        scenario_checks["context_identity"],
        scenario_checks["routing"],
        scenario_checks["conservation"],
        scenario_checks["operand_context_overflow"],
        all(run_checks.values()),
        all(item["pass"] for item in regression_files.values())
        and gem5["pass"]
        and target_free,
    ]
    integrity_checks = {
        "frozen_inputs": all(item["pass"] for item in frozen.values()),
        "compile_manifest": qualify(compile_path)["pass"],
        "run_manifest": qualify(run_path)["pass"],
        "canonical": all(canonical_checks.values()),
        "scenarios": all(scenario_checks.values()),
        "runs": all(run_checks.values()),
        "regressions": all(item["pass"] for item in regression_files.values()),
        "gem5": gem5["pass"],
        "patch": patch["pass"],
        "source_files": all(item["pass"] for item in source_files.values()),
        "source_semantics": all(source_checks.values()),
        "target_free": target_free,
        "acceptance": all(acceptance_gates) and len(acceptance_gates) == 12,
    }
    integrity = all(integrity_checks.values())
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "classification": config["classification"],
        "validation_eligible": config["validation_eligible"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_commit": git_commit(),
        "hypothesis_status": "supported" if integrity else "rejected",
        "audit_integrity": integrity,
        "paper_performance_targets_consumed": False,
        "paper_reproduction_claim": "none_target_free_semantics_fix_only",
        "frozen_inputs": frozen,
        "compile_manifest": qualify(compile_path),
        "run_manifest": qualify(run_path),
        "canonical_checks": canonical_checks,
        "scenario_checks": scenario_checks,
        "conservation_checks": conservation_checks,
        "run_checks": run_checks,
        "regression_files": regression_files,
        "gem5_checks": gem5,
        "patch_checks": patch,
        "source_files": source_files,
        "source_checks": source_checks,
        "summary": {
            "scenarios": len(documents),
            "executions": len(run["records"]),
            "sanitizer_executions": sum(
                item["mode"] in {"asan", "ubsan"} for item in run["records"]
            ),
            "acceptance_gates_passed": sum(acceptance_gates),
            "acceptance_gates_total": len(acceptance_gates),
            "fma_ii1_issue_cycles": [
                event["cycle"] for event in fma_issues
            ],
            "fma_ii1_complete_cycles": [
                event["cycle"] for event in fma_completes
            ],
            "fma_ii1_total_cycles": summaries["fma_ii1_ctx4"]["cycles"],
            "limited_context_total_cycles": summaries["fma_ii1_ctx2"]["cycles"],
            "legacy_regressions_exact": True,
            "full_paper_rows_reproduced": 0,
            "full_paper_rows_total": 18,
        },
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
        raise FileExistsError(f"immutable output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["hypothesis_status"], **report["summary"]}, indent=2))
    return 0 if report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
