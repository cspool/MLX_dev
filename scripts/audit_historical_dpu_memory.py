#!/usr/bin/env python3
"""Audit H106's source-derived DPU DDR/DMA/SPM contract."""

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
from mlxsim.historical_dpu_memory import (
    invalid_relative_address_case,
    scenarios,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/historical_dpu_memory_v1.yaml"


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


def record_for(run: dict[str, Any], scenario: str) -> dict[str, Any]:
    records = [
        item
        for item in run["records"]
        if item["mode"] == "debug"
        and item["replay"] == 1
        and item["scenario"] == scenario
    ]
    if len(records) != 1:
        raise ValueError(f"missing unique H106 record: {scenario}")
    return records[0]


def load_trace(record: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    path_key = f"{kind}_trace_path"
    hash_key = f"{kind}_trace_sha256"
    path = PROJECT_ROOT / record[path_key]
    if sha256_file(path) != record[hash_key]:
        raise ValueError(f"H106 trace digest mismatch: {path}")
    return [json.loads(line) for line in path.read_text().splitlines()]


def ownership_and_mapping(
    trace: list[dict[str, Any]], half_bytes: int
) -> dict[str, bool]:
    owners = {0: "dma", 1: "dma"}
    ownership = True
    parity = True
    physical = True
    for event in trace:
        buffer = int(event["buffer"])
        tile = int(event["tile"])
        kind = event["kind"]
        if kind == "fill_queued":
            ownership &= owners[buffer] == "dma"
            owners[buffer] = "filling"
        elif kind == "fill_start":
            ownership &= owners[buffer] == "filling"
        elif kind == "fill_complete":
            ownership &= owners[buffer] == "filling"
            owners[buffer] = "pe"
        elif kind in {"pe_load", "pe_store"}:
            ownership &= owners[buffer] == "pe"
            parity &= buffer == tile % 2
            physical &= event["physical_address"] == (
                buffer * half_bytes + event["relative_address"]
            )
        elif kind == "pe_release":
            ownership &= owners[buffer] == "pe"
        elif kind == "drain_queued":
            ownership &= owners[buffer] == "pe"
            owners[buffer] = "draining"
        elif kind == "drain_start":
            ownership &= owners[buffer] == "draining"
        elif kind == "drain_complete":
            ownership &= owners[buffer] == "draining"
            owners[buffer] = "dma"
    return {
        "ownership_exclusive": ownership and owners == {0: "dma", 1: "dma"},
        "tile_parity": parity,
        "relative_mapping": physical,
    }


def compile_audit(
    config: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    expected = scenarios(config)
    canonical_checks = {}
    for name, item in expected.items():
        output = manifest["outputs"][name]
        overlay = PROJECT_ROOT / output["overlay"]["path"]
        memory = PROJECT_ROOT / output["memory"]["path"]
        canonical_checks[name] = (
            overlay.read_text() == canonical_json(item["overlay"])
            and memory.read_text() == canonical_json(item["memory"])
            and sha256_file(overlay) == output["overlay"]["sha256"]
            and sha256_file(memory) == output["memory"]["sha256"]
            and item["expected_failure"] == output["expected_failure"]
        )
    auxiliary = invalid_relative_address_case(config)
    auxiliary_output = manifest["auxiliary_outputs"]["invalid_relative_address"]
    auxiliary_overlay = PROJECT_ROOT / auxiliary_output["overlay"]["path"]
    auxiliary_memory = PROJECT_ROOT / auxiliary_output["memory"]["path"]
    auxiliary_exact = (
        auxiliary_overlay.read_text() == canonical_json(auxiliary["overlay"])
        and auxiliary_memory.read_text() == canonical_json(auxiliary["memory"])
        and auxiliary_output["expected_failure"] == auxiliary["expected_failure"]
    )
    fixtures = config["fixtures"]
    fixture_checks = {
        "dpu_2018": fixtures["dpu_2018"]
        == {
            "frequency_hz": 1_000_000_000,
            "dram_bandwidth_bytes_per_cycle": 64,
            "dram_capacity_bytes": None,
            "spm_bytes": 8_388_608,
            "spm_banks": 32,
            "spm_bank_width_bytes": 32,
            "dma_setup_cycles": None,
            "spm_response_cycles": None,
            "buffer_halves": 2,
        },
        "dpu_2019": fixtures["dpu_2019"]
        == {
            "frequency_hz": 1_000_000_000,
            "dram_capacity_bytes": 4_294_967_296,
            "spm_bytes": 3_145_728,
            "spm_banks": None,
            "operand_ram_slices_per_pe": 8,
            "noc_planes": 2,
            "noc_packet_bits": 64,
            "dma_setup_cycles": None,
        },
        "dpu_2022": fixtures["dpu_2022"]
        == {
            "frequency_hz": 1_000_000_000,
            "dram_capacity_bytes": 17_179_869_184,
            "data_spm_bytes": 4_194_304,
            "data_spm_banks": 16,
            "data_spm_bank_width_bytes": 32,
            "instruction_spm_bytes": 4_194_304,
            "instruction_spm_banks": 8,
            "instruction_spm_bank_width_bytes": 128,
            "noc_planes": 4,
            "noc_packet_bits": 64,
            "prefire_queue_entries": 8,
            "dma_setup_cycles": None,
            "spm_response_cycles": None,
        },
    }
    return {
        "scenario_count": len(expected)
        == int(config["execution"]["required_scenarios"]),
        "canonical_configs": all(canonical_checks.values()),
        "canonical_config_checks": canonical_checks,
        "auxiliary_relative_address": auxiliary_exact,
        "historical_fixtures": all(fixture_checks.values()),
        "historical_fixture_checks": fixture_checks,
        "paper_targets_absent": manifest["paper_performance_targets_consumed"]
        is False,
    }


def scenario_audit(
    config: dict[str, Any], run: dict[str, Any]
) -> tuple[dict[str, bool], dict[str, Any]]:
    non_stop_record = record_for(run, "non_stop_four_tiles")
    baseline_record = record_for(run, "baseline_four_tiles")
    same_record = record_for(run, "same_bank_pressure")
    split_record = record_for(run, "split_bank_traffic")
    queue_record = record_for(run, "queue_pressure")
    non_stop = non_stop_record["summary"]
    baseline = baseline_record["summary"]
    same = same_record["summary"]
    split = split_record["summary"]
    queue = queue_record["summary"]
    memory = non_stop["memory"]
    baseline_memory = baseline["memory"]
    trace = load_trace(non_stop_record, "memory")
    mapping = ownership_and_mapping(trace, int(memory["half_bytes"]))
    mechanism = config["mechanism_run"]
    expected_read = int(mechanism["tile_count"]) * int(
        mechanism["input_bytes_per_tile"]
    )
    expected_write = int(mechanism["tile_count"]) * int(
        mechanism["output_bytes_per_tile"]
    )
    expected_dma_cycles = int(mechanism["tile_count"]) * (
        (
            int(mechanism["input_bytes_per_tile"])
            + int(mechanism["dma_bytes_per_cycle"])
            - 1
        )
        // int(mechanism["dma_bytes_per_cycle"])
        + (
            int(mechanism["output_bytes_per_tile"])
            + int(mechanism["dma_bytes_per_cycle"])
            - 1
        )
        // int(mechanism["dma_bytes_per_cycle"])
    )
    invalid_capacity = [
        item
        for item in run["records"]
        if item["scenario"] == "invalid_half_capacity"
    ]
    invalid_relative = run["auxiliary_records"]
    same_work = all(
        non_stop["overlay"][key] == baseline["overlay"][key]
        for key in (
            "instructions_issued",
            "instructions_completed",
            "issued_by_pipeline",
            "external_memory_requests",
            "external_memory_completions",
            "productive_pe_cycles_by_fu_class",
        )
    )
    checks = {
        "ownership_exclusive": mapping["ownership_exclusive"]
        and memory["ownership_violations"] == 0,
        "tile_parity_relative_mapping": mapping["tile_parity"]
        and mapping["relative_mapping"],
        "offchip_byte_conservation": memory["offchip_read_bytes"] == expected_read
        and memory["offchip_write_bytes"] == expected_write
        and baseline_memory["offchip_read_bytes"] == expected_read
        and baseline_memory["offchip_write_bytes"] == expected_write,
        "pe_request_response_conservation": memory["requests"]
        == memory["responses"]
        == non_stop["overlay"]["external_memory_requests"]
        == non_stop["overlay"]["external_memory_completions"]
        == 8
        and memory["released_tiles"] == memory["drained_tiles"] == 4
        and memory["idle"],
        "array_episode_reduction": memory["array_fill_episodes"]
        == memory["array_drain_episodes"]
        == 1
        and baseline_memory["array_fill_episodes"]
        == baseline_memory["array_drain_episodes"]
        == 4
        and same_work,
        "non_stop_cycle_reduction": non_stop["end_to_end_cycles"]
        < baseline["end_to_end_cycles"],
        "dma_cycle_conservation": memory["dma_data_cycles"]
        == baseline_memory["dma_data_cycles"]
        == expected_dma_cycles
        and memory["dma_setup_cycles"] == baseline_memory["dma_setup_cycles"] == 0,
        "bank_pressure": same["memory"]["spad"]["bank_issue_stalls"] > 0
        and split["memory"]["spad"]["bank_issue_stalls"] == 0,
        "queue_pressure": queue["memory"]["spad"]["unavailable_checks"] > 0
        and queue["memory"]["spad"]["max_queue_entries"] == 1
        and queue["memory"]["requests"] == queue["memory"]["responses"] == 8
        and queue["memory"]["idle"],
        "invalid_capacity": len(invalid_capacity) == 6
        and all(item["pass"] for item in invalid_capacity),
        "invalid_relative_address": len(invalid_relative) == 4
        and all(item["pass"] for item in invalid_relative),
    }
    measurements = {
        "non_stop_end_to_end_cycles": non_stop["end_to_end_cycles"],
        "baseline_end_to_end_cycles": baseline["end_to_end_cycles"],
        "mechanism_speedup": baseline["end_to_end_cycles"]
        / non_stop["end_to_end_cycles"],
        "offchip_read_bytes": memory["offchip_read_bytes"],
        "offchip_write_bytes": memory["offchip_write_bytes"],
        "dma_data_cycles": memory["dma_data_cycles"],
        "same_bank_stalls": same["memory"]["spad"]["bank_issue_stalls"],
        "split_bank_stalls": split["memory"]["spad"]["bank_issue_stalls"],
        "queue_unavailable_checks": queue["memory"]["spad"]["unavailable_checks"],
    }
    return checks, measurements


def patch_audit(config: dict[str, Any]) -> dict[str, Any]:
    patch_path = PROJECT_ROOT / config["source_layout"]["patch"]
    header_path = PROJECT_ROOT / config["source_layout"]["overlay_header"]
    source_path = PROJECT_ROOT / config["source_layout"]["overlay_source"]
    h105 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h105"]["path"]).read_text()
    )
    report = {
        "patch": qualify(patch_path),
        "reverse_check": False,
        "h105_source_exact": False,
        "forward_check": False,
        "round_trip_exact": False,
        "newer_patch_stack": {},
    }
    if not patch_path.is_file():
        report["pass"] = False
        return report
    with tempfile.TemporaryDirectory(prefix="mlx-h106-patch-") as temporary:
        root = Path(temporary)
        target = root / "src/cpu/minor/ssim"
        target.mkdir(parents=True)
        header = target / "mlx_overlay.hh"
        source = target / "mlx_overlay.cc"
        shutil.copy2(header_path, header)
        shutil.copy2(source_path, source)
        newer_patches = [
            PROJECT_ROOT
            / "patches/dsagen/dsa-gem5-active-pipelined-scan-v1.patch",
            PROJECT_ROOT
            / "patches/dsagen/dsa-gem5-active-window-capacity-v1.patch",
            PROJECT_ROOT
            / "patches/dsagen/dsa-gem5-pipelined-block-contexts-v1.patch"
        ]
        applied_newer = []
        for newer in newer_patches:
            options = ["--unidiff-zero"] if "active-pipelined" in newer.name else []
            if not newer.is_file():
                continue
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
        report["h105_source_exact"] = (
            sha256_file(header)
            == h105["source_files"]["overlay_header"]["sha256"]
            and sha256_file(source)
            == h105["source_files"]["overlay_source"]["sha256"]
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
            "h105_source_exact",
            "forward_check",
            "round_trip_exact",
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
    disabled = (
        disabled_path.read_text(errors="replace") if disabled_path.is_file() else ""
    )
    overlay_cycles = int(specification["expected_overlay_cycles"])
    dsagen_cycles = int(specification["expected_dsagen_cycles"])
    checks = {
        "binary": binary["pass"],
        "enabled_log": enabled_file["pass"],
        "disabled_log": disabled_file["pass"],
        "enabled_overlay": "MLX_OVERLAY_SUMMARY" in enabled
        and f'"cycles":{overlay_cycles}' in enabled,
        "disabled_overlay": "MLX_OVERLAY_SUMMARY" not in disabled,
        "enabled_dsagen": f"Cycles: {dsagen_cycles}" in enabled
        and "sanity check passed successfully!" in enabled,
        "disabled_dsagen": f"Cycles: {dsagen_cycles}" in disabled
        and "sanity check passed successfully!" in disabled,
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
        name: qualify(PROJECT_ROOT / item["path"], item)
        for name, item in config["frozen_inputs"].items()
    }
    output_root = PROJECT_ROOT / config["output_root"]
    compile_path = output_root / "historical-dpu-memory-compile-manifest.json"
    run_path = output_root / "historical-dpu-memory-run-manifest.json"
    compile_manifest = json.loads(compile_path.read_text())
    run_manifest = json.loads(run_path.read_text())
    compile_checks = compile_audit(config, compile_manifest)
    scenario_checks, measurements = scenario_audit(config, run_manifest)
    patch_checks = patch_audit(config)
    gem5_checks = gem5_audit(config)
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    overlay_header = (
        PROJECT_ROOT / config["source_layout"]["overlay_header"]
    ).read_text()
    overlay_source = (
        PROJECT_ROOT / config["source_layout"]["overlay_source"]
    ).read_text()
    adapter_source = (
        PROJECT_ROOT / config["source_layout"]["adapter_source"]
    ).read_text()
    source_checks = {
        "backend": "DpuMemoryAdapter" in overlay_header
        and 'name == "dpu_memory"' in overlay_source,
        "tile_parity": "result.tile % config_.buffer_halves" in adapter_source,
        "relative_mapping": "result.buffer * half_bytes_ + result.relative"
        in adapter_source,
        "ownership": "Owner::Filling" in adapter_source
        and "Owner::Draining" in adapter_source,
        "dma_conservation": "offchip_read_bytes_" in adapter_source
        and "offchip_write_bytes_" in adapter_source,
        "h66_spad_reuse": "StandaloneSpadAdapter spad_" in (
            PROJECT_ROOT / config["source_layout"]["adapter_header"]
        ).read_text(),
    }
    mode_counts = {
        mode: sum(item["mode"] == mode for item in run_manifest["records"])
        for mode in ("debug", "optimized", "asan", "ubsan")
    }
    run_checks = {
        "records": len(run_manifest["records"]) == 36,
        "auxiliary_records": len(run_manifest["auxiliary_records"]) == 4,
        "mode_counts": mode_counts
        == {"debug": 12, "optimized": 12, "asan": 6, "ubsan": 6},
        "manifest": all(run_manifest["checks"].values()),
        "replays": all(run_manifest["replay_checks"].values()),
        "cross_builds": all(run_manifest["cross_build_checks"].values()),
        "h105_h52": run_manifest["h105"]["pass"],
        "legacy": run_manifest["legacy"]["pass"],
    }
    target_free = (
        config["execution"]["paper_performance_targets_consumed"] is False
        and compile_manifest["paper_performance_targets_consumed"] is False
        and run_manifest["paper_performance_targets_consumed"] is False
    )
    acceptance_gates = [
        compile_checks["historical_fixtures"],
        scenario_checks["ownership_exclusive"],
        scenario_checks["tile_parity_relative_mapping"],
        scenario_checks["offchip_byte_conservation"],
        scenario_checks["pe_request_response_conservation"],
        scenario_checks["array_episode_reduction"],
        scenario_checks["non_stop_cycle_reduction"],
        scenario_checks["dma_cycle_conservation"],
        scenario_checks["bank_pressure"],
        scenario_checks["queue_pressure"],
        scenario_checks["invalid_capacity"]
        and scenario_checks["invalid_relative_address"],
        run_checks["manifest"]
        and run_checks["h105_h52"]
        and run_checks["legacy"]
        and gem5_checks["pass"],
    ]
    integrity_checks = {
        "frozen_inputs": all(item["pass"] for item in frozen.values()),
        "compile_manifest": qualify(compile_path)["pass"],
        "run_manifest": qualify(run_path)["pass"],
        "compile_contract": all(
            value
            for key, value in compile_checks.items()
            if not key.endswith("_checks")
        ),
        "scenarios": all(scenario_checks.values()),
        "runs": all(run_checks.values()),
        "source_files": all(item["pass"] for item in source_files.values()),
        "source_semantics": all(source_checks.values()),
        "reversible_patch": patch_checks["pass"],
        "gem5": gem5_checks["pass"],
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
        "paper_reproduction_claim": "none_target_free_memory_contract_only",
        "frozen_inputs": frozen,
        "compile_manifest": qualify(compile_path),
        "run_manifest": qualify(run_path),
        "compile_checks": compile_checks,
        "scenario_checks": scenario_checks,
        "measurements": measurements,
        "run_checks": run_checks,
        "source_files": source_files,
        "source_checks": source_checks,
        "patch_checks": patch_checks,
        "gem5_checks": gem5_checks,
        "summary": {
            "main_executions": len(run_manifest["records"]),
            "auxiliary_executions": len(run_manifest["auxiliary_records"]),
            "sanitizer_executions": sum(
                item["mode"] in {"asan", "ubsan"}
                for item in run_manifest["records"]
            )
            + sum(
                item["mode"] in {"asan", "ubsan"}
                for item in run_manifest["auxiliary_records"]
            ),
            "acceptance_gates_passed": sum(acceptance_gates),
            "acceptance_gates_total": len(acceptance_gates),
            "non_stop_cycles": measurements["non_stop_end_to_end_cycles"],
            "baseline_cycles": measurements["baseline_end_to_end_cycles"],
            "mechanism_speedup": measurements["mechanism_speedup"],
            "offchip_bytes": measurements["offchip_read_bytes"]
            + measurements["offchip_write_bytes"],
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
            "measurements",
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
