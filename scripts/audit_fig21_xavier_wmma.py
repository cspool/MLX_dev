#!/usr/bin/env python3
"""Audit H144's target-free Xavier WMMA functional-PTX attempt."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig21_xavier_wmma_projection_v1.yaml"


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    parents = {
        name: json.loads((PROJECT_ROOT / spec["path"]).read_text())
        for name, spec in config["frozen_inputs"].items()
    }
    parent_checks = {
        name: parent["hypothesis_status"] == spec["required_status"]
        and parent["audit_integrity"] is spec["required_integrity"]
        for name, parent in parents.items()
        for spec in [config["frozen_inputs"][name]]
    }
    h56 = parents["h56_xavier"]
    xavier_config_record = h56["config_derivation"]["derived"]
    xavier_config_path = PROJECT_ROOT / xavier_config_record["path"]
    xavier_config_text = xavier_config_path.read_text()
    xavier_config_check = (
        hashlib.sha256(xavier_config_path.read_bytes()).hexdigest()
        == xavier_config_record["sha256"]
        and "-gpgpu_tensor_core_avail 1" in xavier_config_text
        and "-gpgpu_num_tensor_core_units 4" in xavier_config_text
        and int(config["xavier"]["clock_hz"]) == 1_377_000_000
    )
    manifest_path = PROJECT_ROOT / config["output_root"] / "xavier-wmma-run-manifest-r16.json"
    manifest = json.loads(manifest_path.read_text())
    generated_inputs = {"run_manifest": qualify(manifest_path)}
    ptx_path = PROJECT_ROOT / manifest["ptx"]["path"]
    ptx_text = ptx_path.read_text()
    ptx_checks = {
        "qualified": hashlib.sha256(ptx_path.read_bytes()).hexdigest() == manifest["ptx"]["sha256"],
        "wmma_load": "wmma.load.a.sync" in ptx_text and "wmma.load.b.sync" in ptx_text,
        "wmma_mma": "wmma.mma.sync" in ptx_text,
        "wmma_store": "wmma.store.d.sync" in ptx_text,
        "compute_70": ".target sm_70" in ptx_text,
    }
    record = manifest["records"].get("wmma-r16", {})
    measurement_path = PROJECT_ROOT / record["path"]
    measurement = json.loads(measurement_path.read_text())
    log_path = PROJECT_ROOT / measurement["run"]["log_path"]
    log_text = log_path.read_text(errors="replace")
    failure_checks = {
        "single_anchor": set(manifest["records"]) == {"wmma-r16"},
        "returncode": record["returncode"] == measurement["returncode"] == 139,
        "stage": record["failure_stage"]
        == measurement["failure_stage"]
        == "post_kernel_enqueue_crash",
        "ptx_parsed": "finished parsing EMBEDDED .ptx file" in log_text,
        "kernel_enqueued": "pushing kernel" in log_text,
        "no_cycles": record["cycles"] == 0 and measurement["run"]["cycles"] is None,
        "no_summary": measurement["run"]["summary"] is None,
        "abnormal_exit": measurement["run"]["normal_exit"] is False,
        "stopping_rule": manifest["stopping_rule_applied"]
        == config["acceptance"]["foundational_failure_stop"],
    }
    expected_repeats = [
        *config["wmma_workload"]["fit_repeats"],
        *config["wmma_workload"]["holdout_repeats"],
    ]
    attempted_runs = len(manifest["records"])
    successful_runs = sum(item["pass"] for item in manifest["records"].values())
    projection_estimates: dict[str, Any] = {}
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
        "sparse" + "_projection_substitute",
    )
    target_free_check = (
        manifest["paper_performance_targets_consumed"] is False
        and config["acceptance"]["targets_consumed"] is False
        and not any(token in source_text for token in forbidden)
    )
    execution_gate = attempted_runs == 4 and successful_runs == 4
    work_gate = execution_gate and all(
        item["fma_equivalents"] == int(config["wmma_workload"]["tiles"]) * repeat * 4096
        for repeat, item in zip(expected_repeats, manifest["records"].values(), strict=True)
    )
    model_gate = False
    holdout_gate = False
    estimates_gate = len(projection_estimates) == int(
        config["acceptance"]["required_projection_shapes"]
    )
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        xavier_config_check,
        all(ptx_checks.values()),
        execution_gate,
        work_gate,
        model_gate,
        holdout_gate,
        estimates_gate,
        target_free_check and all(item["pass"] for item in source_files.values()),
        all(failure_checks.values()) and len(projection_estimates) == 0 and attempted_runs == 1,
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "xavier_config": xavier_config_check,
        "ptx": all(ptx_checks.values()),
        "failure_captured": all(failure_checks.values()),
        "stopping_rule": attempted_runs == 1 and len(expected_repeats) == 4,
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
        "paper_reproduction_claim": "none_wmma_functional_ptx_unsupported",
        "failure_class": "gpgpusim_functional_ptx_wmma_post_enqueue_crash",
        "recommended_successor": "accelsim_trace_driven_tensor_core",
        "frozen_inputs": frozen,
        "generated_inputs": generated_inputs,
        "parent_checks": parent_checks,
        "xavier_config_check": xavier_config_check,
        "ptx_checks": ptx_checks,
        "failure_checks": failure_checks,
        "projection_estimates": projection_estimates,
        "source_files": source_files,
        "target_free_check": target_free_check,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "planned_runs": len(expected_repeats),
            "attempted_runs": attempted_runs,
            "successful_runs": successful_runs,
            "ptx_wmma_present": all(ptx_checks.values()),
            "failure_stage": record["failure_stage"],
            "returncode": record["returncode"],
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
            "ptx_checks",
            "failure_checks",
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
