#!/usr/bin/env python3
"""Audit H105's target-free SimICT/DPU execution contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from mlxsim.dsagen_overlay import canonical_json
from mlxsim.simict_dpu_contract import historical_fixtures, semantic_scenarios

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/simict_dpu_contract_v1.yaml"
OVERLAY_ROOT = (
    PROJECT_ROOT
    / "third_party/dsa-framework/dsa-gem5/src/cpu/minor/ssim"
)
LEGACY_DRIVER = PROJECT_ROOT / "simulator_ext/dsagen/mlx_overlay_driver.cc"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qualify(path: Path, expected: dict[str, Any] | None = None) -> dict[str, Any]:
    exists = path.is_file()
    digest = sha256_file(path) if exists else None
    checks = {"is_file": exists}
    if expected and "sha256" in expected:
        checks["sha256"] = digest == expected["sha256"]
    report: dict[str, Any] = {
        "path": str(path.relative_to(PROJECT_ROOT)) if exists else str(path),
        "bytes": path.stat().st_size if exists else None,
        "sha256": digest,
        "checks": checks,
    }
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
    report["pass"] = all(checks.values())
    return report


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def trace_for(record: dict[str, Any]) -> list[dict[str, Any]]:
    path = PROJECT_ROOT / record["trace_path"]
    if sha256_file(path) != record["trace_sha256"]:
        raise ValueError(f"trace digest mismatch: {path}")
    return [json.loads(line) for line in path.read_text().splitlines()]


def debug_record(run: dict[str, Any], scenario: str) -> dict[str, Any]:
    matches = [
        item
        for item in run["records"]
        if item["mode"] == "debug"
        and item["replay"] == 1
        and item["scenario"] == scenario
    ]
    if len(matches) != 1:
        raise ValueError(f"missing unique debug record for {scenario}")
    return matches[0]


def issue_events(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [event for event in trace if event["kind"] == "issue"]


def scenario_audit(run: dict[str, Any]) -> dict[str, bool]:
    frfo_record = debug_record(run, "frfo_ready_age")
    frfo = issue_events(trace_for(frfo_record))
    frfo_pair = [
        event for event in frfo if event["block"] in {"high_tag_early", "low_tag_late"}
    ]

    tie = issue_events(trace_for(debug_record(run, "frfo_equal_ready")))
    frontier = issue_events(trace_for(debug_record(run, "next_frontier")))
    identity = issue_events(trace_for(debug_record(run, "task_block_instance")))
    capacity_record = debug_record(run, "active_block_capacity")
    capacity_trace = trace_for(capacity_record)
    admissions = [event for event in capacity_trace if event["kind"] == "admit"]
    same = debug_record(run, "same_plane_contention")["summary"]
    split = debug_record(run, "split_plane_no_contention")["summary"]
    four = debug_record(run, "four_plane_routes")["summary"]

    invalid_checks = {}
    for scenario, message in {
        "instruction_slot_overflow": "DPU instruction-slot capacity exceeded",
        "operand_context_overflow": "DPU operand-context capacity exceeded",
    }.items():
        records = [item for item in run["records"] if item["scenario"] == scenario]
        invalid_checks[scenario] = (
            len(records) == 6
            and all(item["pass"] for item in records)
            and all(item["expected_failure"] == message for item in records)
        )

    return {
        "frfo_ready_age": [event["block"] for event in frfo_pair]
        == ["high_tag_early", "low_tag_late"]
        and [event["tag"] for event in frfo_pair] == [3, 2]
        and [event["cycle"] for event in frfo_pair] == [5, 6]
        and [event["detail"]["frfo_ready_cycle"] for event in frfo_pair]
        == ["1", "3"],
        "frfo_equal_ready": [event["block"] for event in tie]
        == ["tie_first", "tie_second"]
        and [event["detail"]["task_id"] for event in tie] == ["1", "2"],
        "next_frontier": [event["cycle"] for event in frontier] == [0, 1]
        and [event["detail"]["frfo_ready_cycle"] for event in frontier]
        == ["0", "1"],
        "task_block_instance": len(identity) == 2
        and {event["detail"]["task_id"] for event in identity} == {"11"}
        and {event["detail"]["block_id"] for event in identity} == {"13"}
        and [event["detail"]["instance"] for event in identity] == ["7", "8"],
        "instruction_slot_overflow": invalid_checks["instruction_slot_overflow"],
        "operand_context_overflow": invalid_checks["operand_context_overflow"],
        "active_block_capacity": capacity_record["summary"][
            "max_active_blocks_per_pe"
        ]
        == 1
        and capacity_record["summary"]["done"]
        and [(event["tag"], event["cycle"]) for event in admissions]
        == [(1, 0), (2, 1)],
        "same_plane_contention": same["cycles"] == 5
        and same["link_stalls"] == 1
        and same["route_hops_by_plane"] == {"0": 3, "1": 0},
        "split_plane_no_contention": split["cycles"] == 4
        and split["link_stalls"] == 0
        and split["route_hops_by_plane"] == {"0": 2, "1": 1},
        "four_plane_routes": four["route_hops"] == 4
        and four["link_stalls"] == 0
        and four["route_hops_by_plane"]
        == {"0": 1, "1": 1, "2": 1, "3": 1},
    }


def compile_audit(config: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    documents = semantic_scenarios()
    documents.update(historical_fixtures(config["fixtures"]))
    canonical_checks = {}
    for name, document in documents.items():
        path = PROJECT_ROOT / manifest["outputs"][name]["artifact"]["path"]
        canonical_checks[name] = (
            path.read_text(encoding="utf-8") == canonical_json(document)
            and sha256_file(path)
            == manifest["outputs"][name]["artifact"]["sha256"]
        )
    fixture_checks = {}
    for name, specification in config["fixtures"].items():
        document = json.loads(
            (
                PROJECT_ROOT / manifest["outputs"][name]["artifact"]["path"]
            ).read_text(encoding="utf-8")
        )
        fixture_checks[name] = (
            document["metadata"]["source_contract"] == specification
            and document["metadata"]["inference_disclosures"][
                "undisclosed_timings"
            ]
            is None
        )
    return {
        "scenario_count": len(documents)
        == int(config["execution"]["required_scenarios"]),
        "canonical_configs": all(canonical_checks.values()),
        "canonical_config_checks": canonical_checks,
        "historical_fixtures": all(fixture_checks.values()),
        "historical_fixture_checks": fixture_checks,
        "paper_targets_absent": manifest["paper_performance_targets_consumed"]
        is False,
    }


def patch_audit(config: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    patch_path = PROJECT_ROOT / config["source_layout"]["patch"]
    current_header = PROJECT_ROOT / config["source_layout"]["overlay_header"]
    current_source = PROJECT_ROOT / config["source_layout"]["overlay_source"]
    report: dict[str, Any] = {
        "patch": qualify(patch_path),
        "reverse_check": False,
        "forward_check": False,
        "round_trip_exact": False,
        "legacy_pre_patch_exact": False,
        "newer_patch_stack": {},
    }
    if not patch_path.is_file():
        report["pass"] = False
        return report
    with tempfile.TemporaryDirectory(prefix="mlx-h105-patch-") as temporary:
        root = Path(temporary)
        source_root = root / "src/cpu/minor/ssim"
        source_root.mkdir(parents=True)
        header = source_root / "mlx_overlay.hh"
        source = source_root / "mlx_overlay.cc"
        shutil.copy2(current_header, header)
        shutil.copy2(current_source, source)
        newer_patches = [
            PROJECT_ROOT
            / "patches/dsagen/dsa-gem5-pipelined-block-contexts-v1.patch",
            PROJECT_ROOT
            / "patches/dsagen/dsa-gem5-historical-dpu-memory-v1.patch",
        ]
        applied_newer = []
        for newer in newer_patches:
            if not newer.is_file():
                continue
            check = subprocess.run(
                ["git", "apply", "-R", "--check", str(newer)],
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
                ["git", "apply", "-R", str(newer)], cwd=root, check=True
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

        binary = root / "legacy-baseline"
        build = subprocess.run(
            [
                "g++",
                "-std=c++17",
                "-O0",
                "-g",
                "-Wall",
                "-Wextra",
                "-Werror",
                f"-I{source_root}",
                "-I/usr/include/jsoncpp",
                str(source),
                str(LEGACY_DRIVER),
                "-ljsoncpp",
                "-o",
                str(binary),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        baseline_report = root / "legacy-report.json"
        baseline_trace = root / "legacy-trace.jsonl"
        execute = subprocess.run(
            [
                str(binary),
                "--report",
                str(baseline_report),
                "--trace",
                str(baseline_trace),
            ],
            capture_output=True,
            text=True,
            check=False,
        ) if build.returncode == 0 else None
        current_legacy = run["legacy"]["debug"]
        report["legacy_pre_patch_exact"] = (
            execute is not None
            and execute.returncode == 0
            and sha256_file(baseline_report) == current_legacy["report_sha256"]
            and sha256_file(baseline_trace) == current_legacy["trace_sha256"]
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
            subprocess.run(
                ["git", "apply", str(patch_path)], cwd=root, check=True
            )
            for newer in reversed(applied_newer):
                subprocess.run(
                    ["git", "apply", "--check", str(newer)],
                    cwd=root,
                    check=True,
                )
                subprocess.run(
                    ["git", "apply", str(newer)], cwd=root, check=True
                )
            report["round_trip_exact"] = (
                header.read_bytes() == current_header.read_bytes()
                and source.read_bytes() == current_source.read_bytes()
            )
    report["pass"] = all(
        report[key]
        for key in (
            "reverse_check",
            "forward_check",
            "round_trip_exact",
            "legacy_pre_patch_exact",
        )
    ) and all(report["newer_patch_stack"].values())
    return report


def gem5_audit(config: dict[str, Any]) -> dict[str, Any]:
    specification = config["regressions"]["gem5"]
    binary = qualify(PROJECT_ROOT / specification["binary"])
    root = PROJECT_ROOT / specification["smoke_root"]
    enabled_path = root / "enabled/run.log"
    disabled_path = root / "disabled/run.log"
    enabled_file = qualify(enabled_path)
    disabled_file = qualify(disabled_path)
    enabled = enabled_path.read_text(errors="replace") if enabled_path.is_file() else ""
    disabled = disabled_path.read_text(errors="replace") if disabled_path.is_file() else ""
    overlay_cycles = int(specification["expected_overlay_cycles"])
    dsagen_cycles = int(specification["expected_dsagen_cycles"])
    checks = {
        "binary": binary["pass"],
        "enabled_log": enabled_file["pass"],
        "disabled_log": disabled_file["pass"],
        "enabled_overlay": "MLX_OVERLAY_SUMMARY" in enabled
        and f'"cycles":{overlay_cycles}' in enabled,
        "disabled_overlay": "MLX_OVERLAY_SUMMARY" not in disabled,
        "enabled_dsagen_cycles": f"Cycles: {dsagen_cycles}" in enabled,
        "disabled_dsagen_cycles": f"Cycles: {dsagen_cycles}" in disabled,
        "enabled_sanity": "sanity check passed successfully!" in enabled,
        "disabled_sanity": "sanity check passed successfully!" in disabled,
    }
    return {
        "binary": binary,
        "enabled_log": enabled_file,
        "disabled_log": disabled_file,
        "checks": checks,
        "pass": all(checks.values()),
    }


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / specification["path"], specification)
        for name, specification in config["frozen_inputs"].items()
    }
    output_root = PROJECT_ROOT / config["output_root"]
    compile_path = output_root / "simict-dpu-compile-manifest.json"
    run_path = output_root / "simict-dpu-run-manifest.json"
    compile_manifest = json.loads(compile_path.read_text(encoding="utf-8"))
    run_manifest = json.loads(run_path.read_text(encoding="utf-8"))
    compiled = compile_audit(config, compile_manifest)
    scenarios = scenario_audit(run_manifest)
    patch = patch_audit(config, run_manifest)
    gem5 = gem5_audit(config)

    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    header_text = (PROJECT_ROOT / config["source_layout"]["overlay_header"]).read_text()
    source_text = (PROJECT_ROOT / config["source_layout"]["overlay_source"]).read_text()
    source_checks = {
        "dpu_mode": "DpuFrfo" in header_text and 'name == "dpu_frfo"' in source_text,
        "frfo_frontier": "frfo_ready_cycle" in header_text
        and "left.ready_cycle" in source_text,
        "block_identity": all(
            token in header_text for token in ("task_id", "numeric_block_id", "instance_base")
        ),
        "capacities": all(
            token in source_text
            for token in (
                "instruction_slots_per_pe",
                "operand_contexts_per_pe",
                "active_blocks_per_pe",
            )
        ),
        "network_planes": "network_planes" in header_text
        and "instruction.network_plane" in source_text,
    }
    run_shape = {
        "record_count": len(run_manifest["records"]) == 78,
        "mode_counts": {
            mode: sum(item["mode"] == mode for item in run_manifest["records"])
            for mode in ("debug", "optimized", "asan", "ubsan")
        }
        == {"debug": 26, "optimized": 26, "asan": 13, "ubsan": 13},
        "manifest_checks": all(run_manifest["checks"].values()),
        "replays": all(run_manifest["replay_checks"].values()),
        "cross_builds": all(run_manifest["cross_build_checks"].values()),
        "legacy": run_manifest["checks"]["legacy"],
        "h52": run_manifest["checks"]["h52"],
    }
    target_free = (
        config["execution"]["paper_performance_targets_consumed"] is False
        and compile_manifest["paper_performance_targets_consumed"] is False
        and run_manifest["paper_performance_targets_consumed"] is False
    )
    integrity_checks = {
        "frozen_inputs": all(item["pass"] for item in frozen.values()),
        "compile_manifest": qualify(compile_path)["pass"],
        "run_manifest": qualify(run_path)["pass"],
        "compiled_contract": all(
            value
            for key, value in compiled.items()
            if not key.endswith("_checks")
        ),
        "semantic_scenarios": all(scenarios.values()) and len(scenarios) == 10,
        "run_shape": all(run_shape.values()),
        "source_files": all(item["pass"] for item in source_files.values()),
        "source_semantics": all(source_checks.values()),
        "reversible_patch": patch["pass"],
        "gem5_integration": gem5["pass"],
        "target_free": target_free,
    }
    integrity = all(integrity_checks.values())
    acceptance_gates = [
        *scenarios.values(),
        compiled["historical_fixtures"],
        run_shape["legacy"] and run_shape["h52"] and gem5["pass"],
    ]
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
        "paper_reproduction_claim": "none_target_free_architecture_contract_only",
        "frozen_inputs": frozen,
        "compile_manifest": qualify(compile_path),
        "run_manifest": qualify(run_path),
        "compile_checks": compiled,
        "scenario_checks": scenarios,
        "run_shape": run_shape,
        "source_files": source_files,
        "source_checks": source_checks,
        "patch_checks": patch,
        "gem5_checks": gem5,
        "summary": {
            "compiled_scenarios": int(config["execution"]["required_scenarios"]),
            "executions": len(run_manifest["records"]),
            "semantic_gates_passed": sum(scenarios.values()),
            "semantic_gates_total": len(scenarios),
            "acceptance_gates_passed": sum(acceptance_gates),
            "acceptance_gates_total": len(acceptance_gates),
            "sanitizer_executions": sum(
                item["mode"] in {"asan", "ubsan"}
                for item in run_manifest["records"]
            ),
            "h52_trace_semantics_exact": run_manifest["h52"]["pass"],
            "gem5_enabled_disabled_569_cycles": gem5["pass"],
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
        existing = json.loads(output.read_text(encoding="utf-8"))
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
