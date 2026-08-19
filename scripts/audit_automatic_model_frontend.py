#!/usr/bin/env python3
"""Audit H190 automatic FX/ONNX import, planning, lowering, and execution."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import onnx
import torch
import yaml

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/automatic_model_frontend_v1.yaml"


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
    frontend_path = PROJECT_ROOT / config["frontend_manifest"]
    execution_path = PROJECT_ROOT / config["execution_manifest"]
    frontend_file = qualify(frontend_path)
    execution_file = qualify(execution_path)
    frontend = json.loads(frontend_path.read_text())
    execution = json.loads(execution_path.read_text())
    package_checks = {
        "torch_installed": bool(torch.__version__)
        and frontend["packages"]["torch"] == torch.__version__,
        "onnx_installed": bool(onnx.__version__)
        and frontend["packages"]["onnx"] == onnx.__version__,
        "fx_graph": frontend["fx_graph_module_type"] == "GraphModule",
        "onnx_model": qualify(
            PROJECT_ROOT / frontend["onnx_model"]["path"], frontend["onnx_model"]
        )["pass"],
    }
    graph_documents = {
        name: json.loads((PROJECT_ROOT / record["graph"]["path"]).read_text())
        for name, record in frontend["graphs"].items()
    }
    signatures = {
        name: record["canonical_signature"] for name, record in frontend["graphs"].items()
    }
    graph_checks = {
        "frontends": set(graph_documents) == {"pytorch_fx", "onnx"},
        "node_counts": all(
            len(graph["nodes"]) == int(config["acceptance"]["required_nodes_per_frontend"])
            for graph in graph_documents.values()
        ),
        "total": sum(len(graph["nodes"]) for graph in graph_documents.values())
        == int(config["acceptance"]["required_total_nodes"]),
        "signature": signatures["pytorch_fx"] == signatures["onnx"],
        "kinds": [node["kind"] for node in graph_documents["pytorch_fx"]["nodes"]]
        == config["model_contract"]["canonical_kinds"],
        "names": [node["id"] for node in graph_documents["pytorch_fx"]["nodes"]]
        == config["model_contract"]["canonical_names"],
        "shapes": all(
            node["shape"] == config["model_contract"]["input_shape"]
            for graph in graph_documents.values()
            for node in graph["nodes"]
        ),
        "dependencies": [node["depends_on"] for node in graph_documents["pytorch_fx"]["nodes"]]
        == [node["depends_on"] for node in graph_documents["onnx"]["nodes"]],
    }
    plan_checks: dict[str, bool] = {}
    profile_checks: dict[str, bool] = {}
    for name, record in frontend["graphs"].items():
        plan = json.loads((PROJECT_ROOT / record["plan"]["path"]).read_text())
        profiles = json.loads((PROJECT_ROOT / record["profiles"]["path"]).read_text())
        plan_checks[name] = all(record["plan_checks"].values()) and (
            len(plan["nodes"]) == int(config["acceptance"]["required_nodes_per_frontend"])
            and plan["peak_spm_bytes"] <= int(config["planning"]["spm_bytes"])
            and len({node["tag"] for node in plan["nodes"]}) == len(plan["nodes"])
        )
        profile_checks[name] = len(profiles) == 6 and all(
            float(profile["profile"]["operations"]) > 0
            and float(profile["profile"]["offchip_bytes"]) > 0
            and bool(profile["profile"]["stages"])
            for profile in profiles
        )
    lineage_checks = {
        "count": len(frontend["lineage"]) == int(config["acceptance"]["required_lineage_entries"]),
        "sources": all(
            item["source_node"]
            and item["source_op"]
            and item["canonical_node"] == item["profile_node"]
            for item in frontend["lineage"]
        ),
        "planning": all(
            item["tag"] > 0
            and len(item["pe"]) == 2
            and item["register"] >= 0
            and item["register_bank"] >= 0
            and item["spm_address"] % int(config["planning"]["dma_alignment_bytes"]) == 0
            for item in frontend["lineage"]
        ),
    }
    execution_checks = {
        "checks": all(execution["checks"].values()),
        "records": len(execution["records"]) == int(config["acceptance"]["required_executions"]),
        "passes": all(record["pass"] for record in execution["records"]),
        "replays": all(execution["replay_checks"].values()),
        "profiles": len(execution["replay_checks"])
        == int(config["acceptance"]["required_profiles"]),
        "finite": all(
            math.isfinite(float(record["simulation"][field]))
            and float(record["simulation"][field]) > 0
            for record in execution["records"]
            for field in ("cycles", "latency_us", "operations", "offchip_bytes")
        ),
    }
    replay_checks = {
        name: all(record["replay_checks"].values()) for name, record in frontend["graphs"].items()
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    frontend_source = (PROJECT_ROOT / config["source_layout"]["frontend"]).read_text()
    automatic_checks = {
        "shape_prop": "ShapeProp" in frontend_source,
        "fx_tracer": "MlxTracer" in frontend_source,
        "onnx_modelproto": "onnx.ModelProto" in frontend_source,
        "planner": all(
            token in frontend_source
            for token in ("cdc_id", "output_register", "spm_address", "memory_live_interval")
        ),
        "no_manual_graph_yaml": "operators:" not in frontend_source,
    }
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        frontend_file["pass"] and all(package_checks.values()),
        graph_checks["frontends"] and graph_checks["node_counts"] and graph_checks["total"],
        graph_checks["signature"] and graph_checks["kinds"] and graph_checks["names"],
        graph_checks["shapes"] and graph_checks["dependencies"],
        all(plan_checks.values()) and all(lineage_checks.values()),
        all(profile_checks.values()),
        execution_file["pass"]
        and execution_checks["records"]
        and execution_checks["passes"]
        and execution_checks["finite"],
        execution_checks["checks"]
        and execution_checks["replays"]
        and execution_checks["profiles"]
        and all(replay_checks.values()),
        all(automatic_checks.values()) and all(item["pass"] for item in source_files.values()),
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": len(parent_checks) == 2,
        "frontend_manifest": frontend_file["pass"],
        "packages": len(package_checks) == 4,
        "graphs": len(graph_checks) == 8,
        "plans": len(plan_checks) == 2,
        "profiles": len(profile_checks) == 2,
        "lineage": len(lineage_checks) == 3,
        "execution_manifest": execution_file["pass"],
        "execution": len(execution_checks) == 6,
        "replays": len(replay_checks) == 2,
        "automatic": len(automatic_checks) == 5,
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
        "paper_reproduction_claim": "none_automatic_frontend_toolchain_only",
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "frontend_manifest": frontend_file,
        "package_checks": package_checks,
        "graph_checks": graph_checks,
        "plan_checks": plan_checks,
        "profile_checks": profile_checks,
        "lineage_checks": lineage_checks,
        "execution_manifest": execution_file,
        "execution_checks": execution_checks,
        "replay_checks": replay_checks,
        "automatic_checks": automatic_checks,
        "source_files": source_files,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "frontends": len(graph_documents),
            "nodes_per_frontend": len(graph_documents["pytorch_fx"]["nodes"]),
            "total_source_nodes": sum(len(graph["nodes"]) for graph in graph_documents.values()),
            "canonical_matches": sum(
                left == right
                for left, right in zip(
                    signatures["pytorch_fx"], signatures["onnx"], strict=True
                )
            ),
            "lineage_entries": len(frontend["lineage"]),
            "profiles": sum(
                len(json.loads((PROJECT_ROOT / record["profiles"]["path"]).read_text()))
                for record in frontend["graphs"].values()
            ),
            "executions": len(execution["records"]),
            "execution_replays": sum(execution["replay_checks"].values()),
            "torch_version": frontend["packages"]["torch"],
            "onnx_version": frontend["packages"]["onnx"],
            "automatic_model_frontend_complete": supported,
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
            "package_checks",
            "graph_checks",
            "plan_checks",
            "profile_checks",
            "lineage_checks",
            "execution_checks",
            "replay_checks",
            "automatic_checks",
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
