#!/usr/bin/env python3
"""Audit H173's detailed Xavier-class end-to-end functional execution."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/xavier_e2e_functional_v1.yaml"


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    h56 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h56"]["path"]).read_text()
    )
    mlx = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["mlx_e2e"]["path"]).read_text()
    )
    parent_checks = {
        "h56": h56["hypothesis_status"]
        == config["frozen_inputs"]["h56"]["required_status"]
        and h56["audit_integrity"]
        is config["frozen_inputs"]["h56"]["required_integrity"],
        "mlx": mlx["hypothesis_status"]
        == config["frozen_inputs"]["mlx_e2e"]["required_status"]
        and mlx["audit_integrity"]
        is config["frozen_inputs"]["mlx_e2e"]["required_integrity"],
        "target_free": h56["paper_performance_targets_consumed"] is False
        and mlx["paper_performance_targets_consumed"] is False,
    }
    output_root = PROJECT_ROOT / config["output_root"]
    manifest_path = output_root / "xavier-e2e-run-manifest.json"
    manifest_file = qualify(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    expected_tokens = [int(value) for value in config["model"]["token_counts"]]
    record_checks: dict[str, bool] = {}
    functional_checks: dict[str, bool] = {}
    execution_checks: dict[str, bool] = {}
    artifact_checks: dict[str, bool] = {}
    for tokens in expected_tokens:
        key = f"N{tokens}"
        record = manifest["records"][key]
        summary = record["summary"]
        record_checks[key] = record["tokens"] == tokens and record["pass"] is True
        functional_checks[key] = (
            summary["operator"] == "dense_transformer"
            and summary["tokens"] == tokens
            and summary["hidden"] == int(config["model"]["hidden_dimension"])
            and summary["ffn"] == int(config["model"]["ffn_dimension"])
            and summary["layers"] == int(config["model"]["layers"])
            and summary["kernel_launches"]
            == int(config["execution"]["expected_kernel_launches_per_run"])
            and math.isfinite(float(summary["maximum_absolute_error"]))
            and float(summary["maximum_absolute_error"])
            <= float(config["execution"]["maximum_absolute_error"])
            and math.isfinite(float(summary["checksum"]))
            and math.isfinite(float(summary["reference"]))
        )
        execution_checks[key] = (
            record["returncode"] == 0
            and record["cycles"] > 0
            and record["instructions"] > 0
            and record["ctas"] > 0
            and record["detailed_mode"] is True
            and record["normal_exit"] is True
            and record["kernel_launches_observed"]
            == int(config["execution"]["expected_kernel_launches_per_run"])
        )
        artifact_checks[key] = qualify(
            PROJECT_ROOT / record["log"]["path"], record["log"]
        )["pass"]
    cycles = [manifest["records"][f"N{tokens}"]["cycles"] for tokens in expected_tokens]
    instructions = [
        manifest["records"][f"N{tokens}"]["instructions"]
        for tokens in expected_tokens
    ]
    scaling_checks = {
        "cycles": cycles == sorted(cycles) and len(set(cycles)) == len(cycles),
        "instructions": instructions == sorted(instructions)
        and len(set(instructions)) == len(instructions),
    }
    source_text = (
        PROJECT_ROOT / config["source_layout"]["cuda_source"]
    ).read_text()
    operator_checks = {
        name: token in source_text
        for name, token in {
            "rmsnorm": "rmsnorm_kernel",
            "dense": "dense_kernel",
            "rope": "rope_qk_kernel",
            "attention": "causal_attention_kernel",
            "residual": "residual_kernel",
            "silu": "silu_gate_kernel",
        }.items()
    }
    inventory_checks = {
        "count": len(config["operator_inventory"]) == 11,
        "operators": all(operator_checks.values()),
        "layers": int(config["model"]["layers"]) == 2,
        "causal": config["model"]["causal_attention"] is True,
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    target_free_check = (
        manifest["paper_performance_targets_consumed"] is False
        and config["execution"]["paper_performance_targets_consumed"] is False
    )
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        all(inventory_checks.values()),
        set(manifest["records"]) == {f"N{tokens}" for tokens in expected_tokens},
        all(record_checks.values()),
        all(execution_checks.values()),
        all(functional_checks.values()),
        all(scaling_checks.values()),
        manifest["checks"]["compile"]
        and qualify(PROJECT_ROOT / manifest["binary"]["path"], manifest["binary"])[
            "pass"
        ],
        mlx["summary"]["both_architectures_functionally_correct"] is True
        and mlx["summary"]["goal_complete"] is True,
        target_free_check
        and all(artifact_checks.values())
        and all(item["pass"] for item in source_files.values()),
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "manifest": manifest_file["pass"] and all(manifest["checks"].values()),
        "records": all(record_checks.values()),
        "functional": all(functional_checks.values()),
        "execution": all(execution_checks.values()),
        "artifacts": all(artifact_checks.values()),
        "scaling": all(scaling_checks.values()),
        "inventory": all(inventory_checks.values()),
        "source": all(item["pass"] for item in source_files.values()),
        "target_free": target_free_check,
        "acceptance_evaluated": len(acceptance_gates) == 10
        and all(isinstance(value, bool) for value in acceptance_gates),
    }
    integrity = all(integrity_checks.values())
    supported = integrity and all(acceptance_gates)
    maximum_error = max(
        float(manifest["records"][f"N{tokens}"]["summary"]["maximum_absolute_error"])
        for tokens in expected_tokens
    )
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
        "paper_reproduction_claim": "none_xavier_class_e2e_functional_only",
        "proxy_identity": "SM70_timing_resource_edited_Xavier_not_native_SM72",
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "generated_input": manifest_file,
        "record_checks": record_checks,
        "functional_checks": functional_checks,
        "execution_checks": execution_checks,
        "artifact_checks": artifact_checks,
        "scaling_checks": scaling_checks,
        "operator_checks": operator_checks,
        "inventory_checks": inventory_checks,
        "records": manifest["records"],
        "source_files": source_files,
        "target_free_check": target_free_check,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "runs": len(manifest["records"]),
            "token_counts": expected_tokens,
            "layers": int(config["model"]["layers"]),
            "operator_groups": len(config["operator_inventory"]),
            "kernel_launches_per_run": int(
                config["execution"]["expected_kernel_launches_per_run"]
            ),
            "maximum_absolute_error": maximum_error,
            "minimum_cycles": min(cycles),
            "maximum_cycles": max(cycles),
            "xavier_e2e_functional_complete": supported,
            "mlx_e2e_functional_parent_complete": mlx["summary"][
                "goal_complete"
            ],
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
            "record_checks",
            "functional_checks",
            "execution_checks",
            "scaling_checks",
            "records",
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
