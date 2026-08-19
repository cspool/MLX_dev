#!/usr/bin/env python3
"""Import real PyTorch FX and ONNX graphs and plan canonical MLX workloads."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import onnx
import torch
import yaml

from mlxsim.model_frontend import (
    build_onnx_model,
    canonical_signature,
    graph_digest,
    import_fx_graph,
    import_onnx_graph,
    plan_graph,
    profile_for_node,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/automatic_model_frontend_v1.yaml"


def write_json(path: Path, value: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    payload = path.read_bytes()
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def write_onnx(path: Path, model: onnx.ModelProto) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)
    payload = path.read_bytes()
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def legal_plan(plan: dict[str, Any], planning: dict[str, Any]) -> dict[str, bool]:
    mesh_x, mesh_y = (int(value) for value in planning["mesh"])
    return {
        "cdc": len(plan["cdcs"]) == 1
        and plan["cdcs"][0]["nodes"] == [node["id"] for node in plan["nodes"]],
        "tags": [node["tag"] for node in plan["nodes"]]
        == list(range(1, len(plan["nodes"]) + 1)),
        "pe": all(
            0 <= node["pe"][0] < mesh_x and 0 <= node["pe"][1] < mesh_y
            for node in plan["nodes"]
        ),
        "registers": all(
            0 <= node["output_register"] < int(planning["register_count"])
            and node["register_bank"]
            == node["output_register"] % int(planning["register_banks"])
            for node in plan["nodes"]
        ),
        "memory": plan["peak_spm_bytes"] <= int(planning["spm_bytes"])
        and all(
            node["spm_address"] % int(planning["dma_alignment_bytes"]) == 0
            for node in plan["nodes"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    model = config["model_contract"]
    input_shape = tuple(int(value) for value in model["input_shape"])
    hidden = int(model["hidden_dimension"])
    seed = 190
    fx_graph, fx_module = import_fx_graph(
        input_shape=input_shape, hidden_dimension=hidden, seed=seed
    )
    fx_replay, _ = import_fx_graph(
        input_shape=input_shape, hidden_dimension=hidden, seed=seed
    )
    onnx_model = build_onnx_model(
        input_shape=input_shape, hidden_dimension=hidden, seed=seed
    )
    onnx_replay_model = build_onnx_model(
        input_shape=input_shape, hidden_dimension=hidden, seed=seed
    )
    onnx_graph = import_onnx_graph(onnx_model)
    onnx_replay = import_onnx_graph(onnx_replay_model)
    graphs = {"pytorch_fx": fx_graph, "onnx": onnx_graph}
    graph_replays = {"pytorch_fx": fx_replay, "onnx": onnx_replay}
    output_root = PROJECT_ROOT / config["output_root"]
    graph_records: dict[str, Any] = {}
    lineage: list[dict[str, Any]] = []
    for frontend, graph in graphs.items():
        plan = plan_graph(graph, config["planning"])
        profiles = [
            {
                "node_id": node["id"],
                **profile_for_node(node, model),
            }
            for node in plan["nodes"]
        ]
        replay_plan = plan_graph(graph_replays[frontend], config["planning"])
        replay_profiles = [
            {
                "node_id": node["id"],
                **profile_for_node(node, model),
            }
            for node in replay_plan["nodes"]
        ]
        graph_file = write_json(output_root / f"{frontend}-graph.json", graph)
        plan_file = write_json(output_root / f"{frontend}-plan.json", plan)
        profile_file = write_json(output_root / f"{frontend}-profiles.json", profiles)
        graph_records[frontend] = {
            "graph": graph_file,
            "plan": plan_file,
            "profiles": profile_file,
            "canonical_signature": canonical_signature(graph),
            "canonical_sha256": graph_digest(graph),
            "node_count": len(graph["nodes"]),
            "plan_checks": legal_plan(plan, config["planning"]),
            "replay_checks": {
                "graph": canonical_signature(graph) == canonical_signature(graph_replays[frontend]),
                "plan": plan == replay_plan,
                "profiles": profiles == replay_profiles,
            },
        }
        for node, profile in zip(plan["nodes"], profiles, strict=True):
            lineage.append(
                {
                    "frontend": frontend,
                    "source_node": node["source"]["node_name"],
                    "source_op": node["source"]["node_op"],
                    "canonical_node": node["id"],
                    "canonical_kind": node["kind"],
                    "shape": node["shape"],
                    "cdc_id": node["cdc_id"],
                    "tag": node["tag"],
                    "pe": node["pe"],
                    "register": node["output_register"],
                    "register_bank": node["register_bank"],
                    "spm_address": node["spm_address"],
                    "profile_node": profile["node_id"],
                }
            )
    onnx_file = write_onnx(output_root / "auto-structured-block.onnx", onnx_model)
    expected_names = list(model["canonical_names"])
    expected_kinds = list(model["canonical_kinds"])
    canonical_match = canonical_signature(fx_graph) == canonical_signature(onnx_graph)
    checks = {
        "frontends": set(graph_records) == {"pytorch_fx", "onnx"},
        "nodes": all(
            record["node_count"] == int(config["acceptance"]["required_nodes_per_frontend"])
            for record in graph_records.values()
        ),
        "names": [node["id"] for node in fx_graph["nodes"]] == expected_names
        and [node["id"] for node in onnx_graph["nodes"]] == expected_names,
        "kinds": [node["kind"] for node in fx_graph["nodes"]] == expected_kinds
        and [node["kind"] for node in onnx_graph["nodes"]] == expected_kinds,
        "canonical": canonical_match,
        "plans": all(
            all(record["plan_checks"].values()) for record in graph_records.values()
        ),
        "profiles": all(
            json.loads((PROJECT_ROOT / record["profiles"]["path"]).read_text())
            for record in graph_records.values()
        ),
        "replay": all(
            all(record["replay_checks"].values()) for record in graph_records.values()
        ),
        "lineage": len(lineage) == int(config["acceptance"]["required_lineage_entries"]),
        "onnx": onnx_file["bytes"] > 0,
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "paper_performance_targets_consumed": False,
        "packages": {
            "torch": torch.__version__,
            "onnx": onnx.__version__,
        },
        "fx_graph_module_type": type(fx_module).__name__,
        "onnx_model": onnx_file,
        "graphs": graph_records,
        "lineage": lineage,
        "checks": checks,
    }
    path = PROJECT_ROOT / config["frontend_manifest"]
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "packages": manifest["packages"],
                "frontends": len(graph_records),
                "nodes": len(lineage),
                "checks": checks,
            },
            indent=2,
        )
    )
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
