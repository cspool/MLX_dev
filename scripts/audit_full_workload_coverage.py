#!/usr/bin/env python3
"""Audit H192 complete shape, operator, and multi-layer coverage."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/full_workload_coverage_v1.yaml"


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    documents: dict[str, Any] = {}
    for name, spec in config["frozen_inputs"].items():
        path = PROJECT_ROOT / spec["path"]
        documents[name] = (
            yaml.safe_load(path.read_text())
            if path.suffix in {".yaml", ".yml"}
            else json.loads(path.read_text())
        )
    parent_checks = {
        name: document["hypothesis_status"] == spec["required_status"]
        and document["audit_integrity"] is spec["required_integrity"]
        for name, document in documents.items()
        for spec in [config["frozen_inputs"][name]]
        if "required_status" in spec
    }
    coverage_path = PROJECT_ROOT / config["coverage_manifest"]
    execution_path = PROJECT_ROOT / config["execution_manifest"]
    coverage_file = qualify(coverage_path)
    execution_file = qualify(execution_path)
    coverage = json.loads(coverage_path.read_text())
    execution = json.loads(execution_path.read_text())
    contract = config["coverage_contract"]
    units = coverage["units"]
    category_checks = {
        "figure23": coverage["format_counts"]["mlx_overlay_json"]
        == int(contract["figure23_units"]),
        "figure19": coverage["format_counts"]["mlx_dpu_memory_json"]
        == int(contract["figure19_units"]),
        "figure20": coverage["format_counts"]["analytical_kernel_profile_json"]
        == int(contract["figure20_units"]),
        "compositions": coverage["format_counts"]["multi_layer_composition_json"]
        == int(contract["composition_units"]),
        "total": len(units) == int(contract["executable_units"]),
    }
    fig23_units = [unit for unit in units if unit["graph_id"] == "figure23_scalability_grid"]
    fig19_units = [unit for unit in units if unit["graph_id"] == "figure19_component_grid" and unit["execution_format"] != "multi_layer_composition_json"]
    fig20_units = [unit for unit in units if unit["graph_id"] == "figure20_kernel_grid" and unit["execution_format"] != "multi_layer_composition_json"]
    coverage_checks = {
        "figure23": {
            (
                unit["sequence_length"],
                unit["active_window"],
                unit["hardware_name"],
            )
            for unit in fig23_units
        }
        == {
            (sequence, window, hardware)
            for sequence in contract["figure23_sequence_lengths"]
            for window in (2, 4)
            for hardware in ("baseline", "simd32_4x4", "simd8_8x8", "simd32_8x8")
        },
        "figure19": {
            (unit["sequence_length"], unit["node_id"]) for unit in fig19_units
        }
        == {
            (sequence, node)
            for sequence in contract["figure19_sequence_lengths"]
            for node in ("fft2d_attention", "global_ffn1", "global_ffn2")
        },
        "figure20": {
            (unit["sequence_length"], unit["node_id"]) for unit in fig20_units
        }
        == {
            (sequence, node)
            for sequence in contract["figure20_sequence_lengths"]
            for node in ("qkv", "attention", "ffn1", "ffn2")
        },
    }
    schema_checks: dict[str, bool] = {}
    replay_checks: dict[str, bool] = {}
    composition_checks: dict[str, bool] = {}
    for unit in units:
        replay_checks[unit["unit_id"]] = all(
            artifact["identical"]
            and artifact["primary"]["sha256"] == artifact["replay"]["sha256"]
            for artifact in unit["artifacts"].values()
        )
        if unit["execution_format"] == "mlx_overlay_json":
            document = json.loads(
                (PROJECT_ROOT / unit["artifacts"]["overlay"]["primary"]["path"]).read_text()
            )
            schema_checks[unit["unit_id"]] = bool(document.get("blocks")) and (
                document.get("physical_timing", {}).get("enabled") is True
            )
        elif unit["execution_format"] == "mlx_dpu_memory_json":
            overlay = json.loads(
                (PROJECT_ROOT / unit["artifacts"]["overlay"]["primary"]["path"]).read_text()
            )
            memory = json.loads(
                (PROJECT_ROOT / unit["artifacts"]["memory"]["primary"]["path"]).read_text()
            )
            schema_checks[unit["unit_id"]] = bool(overlay.get("blocks")) and (
                memory.get("mode") == "non_stop" and int(memory.get("spm_bytes", 0)) > 0
            )
        elif unit["execution_format"] == "analytical_kernel_profile_json":
            profile = json.loads(
                (PROJECT_ROOT / unit["artifacts"]["profile"]["primary"]["path"]).read_text()
            )["profile"]
            schema_checks[unit["unit_id"]] = (
                float(profile["operations"]) > 0
                and float(profile["offchip_bytes"]) > 0
                and bool(profile["stages"])
            )
        else:
            plan = json.loads(
                (PROJECT_ROOT / unit["artifacts"]["plan"]["primary"]["path"]).read_text()
            )
            expected_layers = (
                int(contract["llama_layers"])
                if plan["composition_id"] == "llama2_32_layer"
                else int(contract["fabnet_layers"])
            )
            composition_checks[unit["unit_id"]] = (
                int(plan["total_layers"]) == expected_layers
                and bool(plan["source_units"])
                and set(plan["source_units"])
                <= {candidate["unit_id"] for candidate in units}
            )
            schema_checks[unit["unit_id"]] = composition_checks[unit["unit_id"]]
    execution_checks = {
        "checks": all(execution["checks"].values()),
        "records": len(execution["records"]) == int(contract["replay_executions"]),
        "passes": all(record["pass"] for record in execution["records"]),
        "replays": all(execution["replay_checks"].values()),
        "units": set(execution["replay_checks"])
        == {unit["unit_id"] for unit in units},
        "finite": all(
            record["returncode"] == 0
            and record["summary"] is not None
            for record in execution["records"]
        ),
    }
    composition_execution_checks: dict[str, bool] = {}
    for unit in units:
        if unit["execution_format"] != "multi_layer_composition_json":
            continue
        records = [record for record in execution["records"] if record["unit_id"] == unit["unit_id"]]
        composition_execution_checks[unit["unit_id"]] = len(records) == 2 and all(
            int(record["summary"]["total_layers"]) == int(unit["metadata"]["total_layers"])
            and all(
                entry["total_cycles"]
                == entry["single_layer_cycles"] * int(unit["metadata"]["total_layers"])
                and entry["total_operations"]
                == entry["single_layer_operations"] * int(unit["metadata"]["total_layers"])
                for entry in record["summary"]["per_sequence"].values()
            )
            for record in records
        )
    lineage_checks = {
        "all_units": all(
            unit["graph_id"] and unit["node_id"] and unit["unit_id"]
            for unit in units
        ),
        "shapes": all(
            int(unit.get("sequence_length", 1)) > 0 for unit in units
        ),
        "functional_parent": documents["same_input_parent"]["summary"][
            "same_input_numerical_equivalence_complete"
        ]
        is True,
        "frontend_parent": documents["frontend_parent"]["summary"][
            "automatic_model_frontend_complete"
        ]
        is True,
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    cli_text = (PROJECT_ROOT / config["source_layout"]["cli"]).read_text()
    entrypoint_checks = {
        "single": coverage["single_entrypoint"] is True,
        "no_subprocess": "subprocess" not in cli_text,
        "expander": "expand_coverage" in cli_text,
    }
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        coverage_file["pass"] and all(coverage["checks"].values()),
        all(category_checks.values()),
        all(coverage_checks.values()),
        all(schema_checks.values()),
        len(composition_checks) == 2 and all(composition_checks.values()),
        execution_file["pass"]
        and execution_checks["records"]
        and execution_checks["passes"],
        execution_checks["checks"]
        and execution_checks["replays"]
        and execution_checks["units"],
        all(composition_execution_checks.values()) and all(lineage_checks.values()),
        all(entrypoint_checks.values()) and all(item["pass"] for item in source_files.values()),
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": len(parent_checks) == 3,
        "coverage_manifest": coverage_file["pass"],
        "categories": len(category_checks) == 5,
        "coverage": len(coverage_checks) == 3,
        "schemas": len(schema_checks) == 62,
        "lowering_replays": len(replay_checks) == 62,
        "compositions": len(composition_checks) == 2,
        "execution_manifest": execution_file["pass"],
        "execution": len(execution_checks) == 6,
        "composition_execution": len(composition_execution_checks) == 2,
        "lineage": len(lineage_checks) == 4,
        "entrypoint": len(entrypoint_checks) == 3,
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
        "paper_performance_targets_consumed": True,
        "paper_reproduction_claim": "full_shape_toolchain_coverage_not_author_compiler",
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "coverage_manifest": coverage_file,
        "category_checks": category_checks,
        "coverage_checks": coverage_checks,
        "schema_checks": schema_checks,
        "replay_checks": replay_checks,
        "composition_checks": composition_checks,
        "execution_manifest": execution_file,
        "execution_checks": execution_checks,
        "composition_execution_checks": composition_execution_checks,
        "lineage_checks": lineage_checks,
        "entrypoint_checks": entrypoint_checks,
        "source_files": source_files,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "graphs": int(contract["graphs"]),
            "operator_nodes": int(contract["operator_nodes"]),
            "figure23_units": len(fig23_units),
            "figure19_units": len(fig19_units),
            "figure20_units": len(fig20_units),
            "composition_units": len(composition_checks),
            "executable_units": len(units),
            "lowering_replay_passes": sum(replay_checks.values()),
            "executions": len(execution["records"]),
            "execution_replay_passes": sum(execution["replay_checks"].values()),
            "llama_layers": int(contract["llama_layers"]),
            "fabnet_layers": int(contract["fabnet_layers"]),
            "single_entrypoint": entrypoint_checks["single"],
            "full_workload_coverage_complete": supported,
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
            "category_checks",
            "coverage_checks",
            "schema_checks",
            "replay_checks",
            "composition_checks",
            "execution_checks",
            "composition_execution_checks",
            "lineage_checks",
            "entrypoint_checks",
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
