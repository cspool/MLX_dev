#!/usr/bin/env python3
"""Audit H189 same-input numerical equivalence across lowered mappings."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/same_input_numerical_equivalence_v1.yaml"


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    parents = {
        name: json.loads((PROJECT_ROOT / spec["path"]).read_text())
        for name, spec in config["frozen_inputs"].items()
        if "required_status" in spec
    }
    parent_checks = {
        name: parent["hypothesis_status"] == config["frozen_inputs"][name]["required_status"]
        and parent["audit_integrity"]
        is config["frozen_inputs"][name]["required_integrity"]
        for name, parent in parents.items()
    }
    manifest_path = PROJECT_ROOT / config["execution_manifest"]
    manifest_file = qualify(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    acceptance = config["acceptance"]
    coverage_checks = {
        "graphs": manifest["summary"]["graphs"] == int(acceptance["required_graphs"]),
        "nodes": manifest["summary"]["nodes"] == int(acceptance["required_nodes"]),
        "runs": manifest["summary"]["runs"] == int(acceptance["required_runs"]),
        "boundaries": manifest["summary"]["boundary_comparisons"]
        == int(acceptance["required_boundary_comparisons"]),
        "finals": manifest["summary"]["final_comparisons"]
        == int(acceptance["required_final_comparisons"]),
        "invariance": manifest["summary"]["mapping_invariance_checks"]
        == int(acceptance["required_mapping_invariance_checks"]),
    }
    precision_checks: dict[str, bool] = {}
    boundary_passes = 0
    final_passes = 0
    event_passes = 0
    work_passes = 0
    for run in manifest["runs"]:
        precision = run["precision"]
        limits = config["test_contract"]["precisions"][precision]
        absolute_limit = float(limits["absolute_tolerance"])
        relative_limit = float(limits["relative_tolerance"])
        boundary_ok = all(
            values["maximum_absolute_error"] <= absolute_limit
            or values["maximum_relative_error"] <= relative_limit
            for values in run["comparison"]["boundaries"].values()
        )
        final_ok = (
            run["comparison"]["final_maximum_absolute_error"] <= absolute_limit
            or run["comparison"]["final_maximum_relative_error"] <= relative_limit
        )
        precision_checks[
            f"{run['graph_id']}-{run['seed']}-{precision}-{run['mapping']['name']}"
        ] = boundary_ok and final_ok
        boundary_passes += sum(run["boundary_passes"].values())
        final_passes += final_ok
        event_passes += run["comparison"]["event_order_identity"]
        work_passes += (
            run["comparison"]["operation_count_identity"]
            and run["comparison"]["tensor_element_identity"]
        )
    invariance_checks = {
        f"{item['graph_id']}-{item['seed']}-{item['precision']}-{item['mapping']}": item[
            "within_tolerance"
        ]
        for item in manifest["mapping_invariance"]
    }
    structure_checks = {
        "orders": all(
            run["topological_order"] == manifest["topological_orders"][run["graph_id"]]
            for run in manifest["runs"]
        ),
        "events": event_passes == int(acceptance["required_runs"]),
        "work": work_passes == int(acceptance["required_runs"]),
        "mapping_invariance": all(invariance_checks.values()),
        "parent_functional": parents["full_operator_parent"]["summary"][
            "mlx_full_operator_functional_complete"
        ]
        is True
        and parents["scalar_payload_parent"]["summary"][
            "integrated_scalar_functional_execution_complete"
        ]
        is True,
    }
    numerical_checks = {
        "boundaries": boundary_passes == int(acceptance["required_boundary_comparisons"]),
        "finals": final_passes == int(acceptance["required_final_comparisons"]),
        "precision": len(precision_checks) == int(acceptance["required_runs"])
        and all(precision_checks.values()),
        "finite": math.isfinite(float(manifest["summary"]["maximum_absolute_error"]))
        and math.isfinite(float(manifest["summary"]["maximum_relative_error"])),
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    source_text = "\n".join(
        (PROJECT_ROOT / path).read_text(errors="replace")
        for path in config["source_layout"].values()
    )
    target_free_checks = {
        "manifest": manifest["paper_performance_targets_consumed"] is False,
        "config": acceptance["paper_targets_consumed"] is False,
        "source": "paper_targets.yaml" not in source_text
        and "target_speedup" not in source_text,
    }
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        manifest_file["pass"] and all(manifest["checks"].values()),
        coverage_checks["graphs"] and coverage_checks["nodes"],
        coverage_checks["runs"],
        coverage_checks["boundaries"] and coverage_checks["finals"],
        numerical_checks["boundaries"] and numerical_checks["finals"],
        numerical_checks["precision"] and numerical_checks["finite"],
        structure_checks["orders"] and structure_checks["events"],
        structure_checks["work"]
        and structure_checks["mapping_invariance"]
        and structure_checks["parent_functional"],
        all(target_free_checks.values()) and all(item["pass"] for item in source_files.values()),
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": len(parent_checks) == 3,
        "manifest": manifest_file["pass"],
        "coverage": len(coverage_checks) == 6,
        "precision": len(precision_checks) == 72,
        "invariance": len(invariance_checks) == 54,
        "structure": len(structure_checks) == 5,
        "numerical": len(numerical_checks) == 4,
        "target_free": len(target_free_checks) == 3,
        "source": all(item["pass"] for item in source_files.values()),
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
        "paper_reproduction_claim": "none_same_input_functional_equivalence_only",
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "execution_manifest": manifest_file,
        "coverage_checks": coverage_checks,
        "precision_checks": precision_checks,
        "invariance_checks": invariance_checks,
        "structure_checks": structure_checks,
        "numerical_checks": numerical_checks,
        "target_free_checks": target_free_checks,
        "source_files": source_files,
        "acceptance_gates": acceptance_gates,
        "summary": {
            **manifest["summary"],
            "boundary_passes": boundary_passes,
            "final_passes": final_passes,
            "event_order_passes": event_passes,
            "work_identity_passes": work_passes,
            "mapping_invariance_passes": sum(invariance_checks.values()),
            "same_input_numerical_equivalence_complete": supported,
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
            "coverage_checks",
            "precision_checks",
            "invariance_checks",
            "structure_checks",
            "numerical_checks",
            "target_free_checks",
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
