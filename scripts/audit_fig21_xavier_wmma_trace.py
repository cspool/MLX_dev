#!/usr/bin/env python3
"""Audit H145's target-free NVBit/Accel-Sim WMMA trace attempt."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig21_xavier_wmma_trace_v1.yaml"


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    parent_names = ("h56_xavier", "h91_contract", "h144_failure")
    parents = {
        name: json.loads((PROJECT_ROOT / config["frozen_inputs"][name]["path"]).read_text())
        for name in parent_names
    }
    parent_checks = {
        name: parent["hypothesis_status"] == config["frozen_inputs"][name]["required_status"]
        and parent["audit_integrity"] is config["frozen_inputs"][name]["required_integrity"]
        for name, parent in parents.items()
    }
    h144_check = (
        parents["h144_failure"]["failure_class"]
        == "gpgpusim_functional_ptx_wmma_post_enqueue_crash"
        and parents["h144_failure"]["summary"]["projection_estimates"] == 0
    )
    manifest_path = PROJECT_ROOT / config["output_root"] / "xavier-wmma-trace-run-manifest-r16.json"
    manifest = json.loads(manifest_path.read_text())
    generated_inputs = {"run_manifest": qualify(manifest_path)}
    device = manifest["device"]
    device_check = (
        device["name"] == config["trace_capture"]["expected_device_name"]
        and device["uuid"] == config["trace_capture"]["expected_device_uuid"]
    )
    capture = manifest["captures"].get("wmma-r16", {})
    log_path = PROJECT_ROOT / capture["log"]["path"]
    log_text = log_path.read_text(errors="replace")
    capture_failure_checks = {
        "single_anchor": set(manifest["captures"]) == {"wmma-r16"},
        "returncode": capture["returncode"] == 1,
        "nvbit_version": capture["nvbit_banner"] is True
        and "NVidia Binary Instrumentation Tool v1.7.3" in log_text,
        "unsupported": capture["cuda_error_not_supported"] is True
        and "CUDA_ERROR_NOT_SUPPORTED" in log_text,
        "application_not_started": capture["application_summary_present"] is False,
        "no_trace": capture["trace_files"] == [],
        "no_replay": manifest["replays"] == {},
        "stopping_rule": manifest["stopping_rule_applied"]
        == config["acceptance"]["foundational_failure_stop"],
    }
    trace_gate = len(manifest["captures"]) == int(config["acceptance"]["required_traces"]) and all(
        item["pass"] for item in manifest["captures"].values()
    )
    replay_gate = len(manifest["replays"]) == int(config["acceptance"]["required_replays"]) and all(
        item["pass"] for item in manifest["replays"].values()
    )
    work_gate = trace_gate and replay_gate
    model_gate = False
    holdout_gate = False
    projection_estimates: dict[str, Any] = {}
    estimates_gate = len(projection_estimates) == int(
        config["acceptance"]["required_projection_shapes"]
    )
    source_files = {
        name: qualify(PROJECT_ROOT / path) for name, path in config["source_layout"].items()
    }
    source_text = "\n".join(
        (PROJECT_ROOT / path).read_text(errors="replace")
        for path in config["source_layout"].values()
    )
    forbidden = (
        "fig21-target" + "s-run094.json",
        "target" + "_efficiency_factor",
        "post_result" + "_trace_selection",
    )
    target_free_check = (
        manifest["paper_performance_targets_consumed"] is False
        and config["acceptance"]["targets_consumed"] is False
        and not any(token in source_text for token in forbidden)
    )
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        device_check and h144_check,
        trace_gate,
        replay_gate,
        work_gate,
        model_gate,
        holdout_gate,
        estimates_gate,
        target_free_check and all(item["pass"] for item in source_files.values()),
        all(capture_failure_checks.values())
        and len(projection_estimates) == 0
        and config["trace_capture"]["trace_isa"] == "SM89"
        and config["xavier_replay"]["timing_isa"] == "SM70_Volta_proxy",
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "h144": h144_check,
        "device": device_check,
        "capture_failure": all(capture_failure_checks.values()),
        "no_trace_or_replay": capture["trace_files"] == [] and manifest["replays"] == {},
        "no_estimates": len(projection_estimates) == 0,
        "source": target_free_check and all(item["pass"] for item in source_files.values()),
        "acceptance_evaluated": len(acceptance_gates) == 10
        and all(isinstance(value, bool) for value in acceptance_gates),
    }
    integrity = all(integrity_checks.values())
    supported = integrity and all(acceptance_gates)
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "classification": config["classification"],
        "validation_eligible": config["validation_eligible"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_commit": git_commit(),
        "hypothesis_status": "supported" if supported else "rejected",
        "audit_integrity": integrity,
        "paper_performance_targets_consumed": False,
        "paper_reproduction_claim": "none_nvbit_capture_unsupported",
        "failure_class": "nvbit_1_7_3_cuda_error_not_supported_on_driver_595",
        "recommended_successor": "source_derived_hmma_traceg_generator",
        "cross_isa_proxy": "SM89_trace_capture_failed_before_SM70_timing_replay",
        "frozen_inputs": frozen,
        "generated_inputs": generated_inputs,
        "parent_checks": parent_checks,
        "device_check": device_check,
        "h144_check": h144_check,
        "capture_failure_checks": capture_failure_checks,
        "projection_estimates": projection_estimates,
        "source_files": source_files,
        "target_free_check": target_free_check,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "planned_captures": int(config["acceptance"]["required_traces"]),
            "attempted_captures": len(manifest["captures"]),
            "successful_captures": sum(item["pass"] for item in manifest["captures"].values()),
            "trace_files": sum(len(item["trace_files"]) for item in manifest["captures"].values()),
            "replays": len(manifest["replays"]),
            "capture_returncode": capture["returncode"],
            "driver_version": device["driver_version"],
            "projection_estimates": len(projection_estimates),
            "figure21_dense_projection_complete": False,
            "active_simulator_figures_reproduced": 3,
            "active_simulator_figures_total": 8,
            "acceptance_gates_passed": sum(acceptance_gates),
            "acceptance_gates_total": len(acceptance_gates),
        },
        "integrity_checks": integrity_checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    report = build_audit(config)
    output = PROJECT_ROOT / config["result_path"]
    if args.verify_existing:
        existing = json.loads(output.read_text())
        keys = (
            "hypothesis_status",
            "audit_integrity",
            "failure_class",
            "capture_failure_checks",
            "projection_estimates",
            "acceptance_gates",
            "summary",
            "integrity_checks",
        )
        matches = all(
            json.dumps(existing.get(key), sort_keys=True)
            == json.dumps(report.get(key), sort_keys=True)
            for key in keys
        )
        print(json.dumps({"existing_matches": matches, **report}, indent=2))
        return 0 if matches else 1
    if args.preflight_only:
        print(json.dumps(report, indent=2))
        return 0 if report["audit_integrity"] and not output.exists() else 1
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["hypothesis_status"], **report["summary"]}, indent=2))
    return 0 if report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
