#!/usr/bin/env python3
"""Audit H187 unified workload lowering, execution, replay, and lineage."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from mlxsim.workload_lowering import validate_suite
from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/unified_workload_lowering_v1.yaml"


def load_document(path: Path) -> Any:
    return yaml.safe_load(path.read_text()) if path.suffix in {".yaml", ".yml"} else json.loads(path.read_text())


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    documents = {
        name: load_document(PROJECT_ROOT / spec["path"])
        for name, spec in config["frozen_inputs"].items()
    }
    parent_checks = {
        name: document["hypothesis_status"] == spec["required_status"]
        and document["audit_integrity"] is spec["required_integrity"]
        for name, document in documents.items()
        for spec in [config["frozen_inputs"][name]]
        if "required_status" in spec
    }
    spec = documents["workload_spec"]
    orders = validate_suite(spec)
    graph_checks = {
        "graphs": len(orders) == int(config["toolchain_contract"]["required_graphs"]),
        "nodes": sum(len(order) for order in orders.values())
        == int(config["toolchain_contract"]["required_graph_nodes"]),
        "complete": all(
            set(order) == {str(node["id"]) for node in spec["graphs"][graph]["operators"]}
            for graph, order in orders.items()
        ),
        "unique": len(
            {f"{graph_id}.{node}" for graph_id, order in orders.items() for node in order}
        )
        == sum(len(order) for order in orders.values()),
    }
    lowering_path = PROJECT_ROOT / config["lowering_manifest"]
    execution_path = PROJECT_ROOT / config["execution_manifest"]
    lowering_file = qualify(lowering_path)
    execution_file = qualify(execution_path)
    lowering = json.loads(lowering_path.read_text())
    execution = json.loads(execution_path.read_text())
    contract = config["toolchain_contract"]
    lowering_checks = {
        "checks": all(lowering["checks"].values()),
        "units": len(lowering["units"]) == int(contract["required_executable_units"]),
        "overlay_units": lowering["format_counts"]["mlx_overlay_json"]
        + lowering["format_counts"]["mlx_dpu_memory_json"]
        == int(contract["required_overlay_units"]),
        "memory": lowering["memory_configs"] == int(contract["required_memory_configs"]),
        "analytical": lowering["format_counts"]["analytical_kernel_profile_json"]
        == int(contract["required_analytical_profiles"]),
        "formats": set(lowering["format_counts"])
        == set(contract["execution_formats"]),
        "orders": lowering["topological_orders"] == orders,
    }
    lineage_checks: dict[str, bool] = {}
    unit_ids = {unit["unit_id"] for unit in lowering["units"]}
    for graph_id, order in orders.items():
        graph_lineage = lowering["lineage"][graph_id]
        lineage_checks[graph_id] = set(graph_lineage) == set(order) and all(
            references and set(references) <= unit_ids for references in graph_lineage.values()
        )
    detailed_schema_checks: dict[str, bool] = {}
    analytical_schema_checks: dict[str, bool] = {}
    artifact_replay_checks: dict[str, bool] = {}
    memory_count = 0
    for unit in lowering["units"]:
        artifact_replay_checks[unit["unit_id"]] = all(
            artifact["identical"]
            and artifact["primary"]["sha256"] == artifact["replay"]["sha256"]
            for artifact in unit["artifacts"].values()
        )
        if unit["execution_format"] in {"mlx_overlay_json", "mlx_dpu_memory_json"}:
            overlay = json.loads(
                (PROJECT_ROOT / unit["artifacts"]["overlay"]["primary"]["path"]).read_text()
            )
            blocks = overlay.get("blocks", [])
            tags = {int(block["tag"]) for block in blocks}
            event_edges = sum(len(block.get("wait_events", [])) for block in blocks) + sum(
                bool(instruction.get("emit_event"))
                for block in blocks
                for instruction in block.get("instructions", [])
            )
            detailed_schema_checks[unit["unit_id"]] = (
                bool(blocks)
                and bool(tags)
                and all(
                    len(block.get("pe", [])) == 2
                    and int(block.get("trip_count", 0)) > 0
                    and bool(block.get("instructions"))
                    for block in blocks
                )
                and event_edges > 0
                and overlay.get("memory_backend") in {"fixed", "dpu_memory"}
                and "routing" in overlay
            )
            if "memory" in unit["artifacts"]:
                memory_count += 1
                memory = json.loads(
                    (PROJECT_ROOT / unit["artifacts"]["memory"]["primary"]["path"]).read_text()
                )
                detailed_schema_checks[unit["unit_id"]] &= (
                    memory.get("mode") == "non_stop"
                    and int(memory.get("spm_bytes", 0)) > 0
                    and int(memory.get("dma_bytes_per_cycle", 0)) > 0
                    and int(memory.get("spad_ports", 0)) > 0
                )
        else:
            profile_doc = json.loads(
                (PROJECT_ROOT / unit["artifacts"]["profile"]["primary"]["path"]).read_text()
            )
            workload = profile_doc.get("workload", {})
            profile = profile_doc.get("profile", {})
            stages = profile.get("stages", [])
            analytical_schema_checks[unit["unit_id"]] = (
                int(workload.get("n", 0)) > 0
                and int(workload.get("d", 0)) > 0
                and float(profile.get("operations", 0)) > 0
                and float(profile.get("offchip_bytes", 0)) > 0
                and bool(stages)
                and all(
                    int(stage.get("tag", -1)) >= 0
                    and float(stage.get("operations", 0)) > 0
                    and bool(stage.get("kernel_class"))
                    for stage in stages
                )
            )
    schema_checks = {
        "detailed": len(detailed_schema_checks) == 4
        and all(detailed_schema_checks.values()),
        "analytical": len(analytical_schema_checks) == 8
        and all(analytical_schema_checks.values()),
        "memory": memory_count == int(contract["required_memory_configs"]),
        "replay": len(artifact_replay_checks) == 12
        and all(artifact_replay_checks.values()),
    }
    execution_checks = {
        "checks": all(execution["checks"].values()),
        "records": len(execution["records"]) == int(contract["required_executions"]),
        "passes": all(record["pass"] for record in execution["records"]),
        "replays": all(execution["replay_checks"].values()),
        "units": set(execution["replay_checks"]) == unit_ids,
    }
    representative_checks: dict[str, bool] = {}
    for unit in lowering["units"]:
        records = [record for record in execution["records"] if record["unit_id"] == unit["unit_id"]]
        first = records[0]["summary"]
        if unit["execution_format"] == "mlx_overlay_json":
            representative_checks[unit["unit_id"]] = (
                first["done"] is True
                and first["raw_cycles"] == 372
                and first["cycles"] == 209
                and first["latency_service"]["startup_credit_cycles"] == 163
            )
        elif unit["execution_format"] == "mlx_dpu_memory_json":
            representative_checks[unit["unit_id"]] = (
                first["overlay"]["done"] is True
                and first["memory"]["idle"] is True
                and first["end_to_end_cycles"] > 0
            )
        else:
            simulation = first["simulation"]
            representative_checks[unit["unit_id"]] = all(
                math.isfinite(float(simulation[field])) and float(simulation[field]) > 0
                for field in ("cycles", "latency_us", "operations", "offchip_bytes")
            )
    numerical_checks = {
        "figure23": documents["figure23_result"]["summary"][
            "figure23_numerically_reproduced_within_15pct"
        ]
        is True
        and documents["figure23_result"]["summary"]["max_relative_error"] <= 0.15,
        "figure19": documents["figure19_result"]["summary"][
            "figure19_numerically_reproduced_within_15pct"
        ]
        is True
        and documents["figure19_result"]["summary"]["max_relative_error"] <= 0.15,
        "figure20": documents["figure20_result"]["summary"][
            "figure20_numerically_reproduced_within_15pct"
        ]
        is True
        and documents["figure20_result"]["summary"]["max_relative_error"] <= 0.15,
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    documentation = (PROJECT_ROOT / config["source_layout"]["documentation"]).read_text()
    documentation_checks = {
        "pipeline": "model/operator graph YAML" in documentation
        and "lowering adapter" in documentation,
        "formats": all(value in documentation for value in contract["execution_formats"]),
        "boundary": "not claimed to be the paper authors' unpublished" in documentation,
    }
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        all(graph_checks.values()),
        all(lineage_checks.values()),
        lowering_file["pass"] and all(lowering_checks.values()),
        schema_checks["detailed"] and schema_checks["memory"],
        schema_checks["analytical"],
        schema_checks["replay"] and all(artifact_replay_checks.values()),
        execution_file["pass"] and all(execution_checks.values()),
        all(representative_checks.values()) and all(numerical_checks.values()),
        all(documentation_checks.values()) and all(item["pass"] for item in source_files.values()),
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": len(parent_checks) == 5,
        "graphs": len(graph_checks) == 4,
        "lineage": len(lineage_checks) == 3,
        "lowering": len(lowering_checks) == 7,
        "detailed_schema": len(detailed_schema_checks) == 4,
        "analytical_schema": len(analytical_schema_checks) == 8,
        "artifact_replay": len(artifact_replay_checks) == 12,
        "execution": len(execution_checks) == 5,
        "representatives": len(representative_checks) == 12,
        "numerical": len(numerical_checks) == 3,
        "documentation": len(documentation_checks) == 3,
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
        "paper_reproduction_claim": "unified_repository_toolchain_not_author_compiler",
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "graph_checks": graph_checks,
        "topological_orders": orders,
        "lineage_checks": lineage_checks,
        "lowering_manifest": lowering_file,
        "lowering_checks": lowering_checks,
        "detailed_schema_checks": detailed_schema_checks,
        "analytical_schema_checks": analytical_schema_checks,
        "artifact_replay_checks": artifact_replay_checks,
        "schema_checks": schema_checks,
        "execution_manifest": execution_file,
        "execution_checks": execution_checks,
        "representative_checks": representative_checks,
        "numerical_checks": numerical_checks,
        "documentation_checks": documentation_checks,
        "source_files": source_files,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "graphs": len(orders),
            "graph_nodes": sum(len(order) for order in orders.values()),
            "executable_units": len(lowering["units"]),
            "detailed_overlay_units": len(detailed_schema_checks),
            "memory_configs": memory_count,
            "analytical_profiles": len(analytical_schema_checks),
            "lineage_nodes": sum(len(value) for value in lowering["lineage"].values()),
            "lowering_replays": sum(artifact_replay_checks.values()),
            "executions": len(execution["records"]),
            "execution_replays": sum(execution["replay_checks"].values()),
            "numerically_complete_figures": sum(numerical_checks.values()),
            "unified_toolchain_complete": supported,
            "author_toolchain_claimed": False,
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
            "graph_checks",
            "topological_orders",
            "lineage_checks",
            "lowering_checks",
            "detailed_schema_checks",
            "analytical_schema_checks",
            "artifact_replay_checks",
            "schema_checks",
            "execution_checks",
            "representative_checks",
            "numerical_checks",
            "documentation_checks",
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
