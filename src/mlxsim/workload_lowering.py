"""Unified high-level workload graph lowering for the MLX simulator backends."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mlxsim.dsagen_overlay import canonical_json
from mlxsim.fig19_coupled_paths import compile_fig19_coupled_path
from mlxsim.fig19_source_paths import compile_fft2d_path
from mlxsim.fig21_timed_paths import compile_timed_path
from mlxsim.fig23_complete_block import compile_complete_block_scaling
from mlxsim.schema import Workload
from mlxsim.workloads import compile_workload


class WorkloadLoweringError(ValueError):
    """Raised when a workload graph or lowering contract is invalid."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _artifact(path: Path, project_root: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": str(path.relative_to(project_root)),
        "bytes": len(payload),
        "sha256": _sha256_bytes(payload),
    }


def _write_replay(
    *,
    value: Any,
    primary: Path,
    replay: Path,
    project_root: Path,
    canonical_overlay: bool,
) -> dict[str, Any]:
    primary.parent.mkdir(parents=True, exist_ok=True)
    replay.parent.mkdir(parents=True, exist_ok=True)
    if canonical_overlay:
        payload = canonical_json(value)
    else:
        payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    primary.write_text(payload)
    replay.write_text(payload)
    first = _artifact(primary, project_root)
    second = _artifact(replay, project_root)
    return {"primary": first, "replay": second, "identical": first["sha256"] == second["sha256"]}


def topological_order(graph: Mapping[str, Any]) -> list[str]:
    operators = graph.get("operators")
    if not isinstance(operators, list) or not operators:
        raise WorkloadLoweringError("each graph requires a non-empty operators list")
    identifiers = [str(operator.get("id", "")) for operator in operators]
    if any(not identifier for identifier in identifiers) or len(set(identifiers)) != len(
        identifiers
    ):
        raise WorkloadLoweringError("operator IDs must be non-empty and unique")
    dependencies = {
        str(operator["id"]): [str(item) for item in operator.get("depends_on", [])]
        for operator in operators
    }
    if any(parent not in dependencies for parents in dependencies.values() for parent in parents):
        raise WorkloadLoweringError("every dependency must resolve in the same graph")
    order: list[str] = []
    remaining = set(identifiers)
    while remaining:
        ready = [
            identifier
            for identifier in identifiers
            if identifier in remaining
            and all(parent in order for parent in dependencies[identifier])
        ]
        if not ready:
            raise WorkloadLoweringError("operator graph contains a dependency cycle")
        for identifier in ready:
            order.append(identifier)
            remaining.remove(identifier)
    return order


def validate_suite(spec: Mapping[str, Any]) -> dict[str, list[str]]:
    if int(spec.get("schema_version", 0)) != 1:
        raise WorkloadLoweringError("workload suite schema_version must be 1")
    graphs = spec.get("graphs")
    if not isinstance(graphs, dict) or not graphs:
        raise WorkloadLoweringError("workload suite requires a graphs mapping")
    supported = {
        "fig23_complete_block": "mlx_overlay_json",
        "fig19_coupled_paths": "mlx_dpu_memory_json",
        "analytical_kernel_profiles": "analytical_kernel_profile_json",
    }
    orders: dict[str, list[str]] = {}
    for graph_id, graph in graphs.items():
        model = graph.get("model", {})
        dimensions = [
            value
            for key, value in model.items()
            if key.endswith("dimension") or key in {"batch", "sequence_length"}
        ]
        if any(not isinstance(value, int) or value <= 0 for value in dimensions):
            raise WorkloadLoweringError(f"{graph_id} model dimensions must be positive")
        if "sequence_lengths" in model and (
            not isinstance(model["sequence_lengths"], list)
            or any(not isinstance(value, int) or value <= 0 for value in model["sequence_lengths"])
        ):
            raise WorkloadLoweringError(f"{graph_id} sequence_lengths must be positive")
        lowering = graph.get("lowering", {})
        adapter = lowering.get("adapter")
        if adapter not in supported:
            raise WorkloadLoweringError(f"unsupported lowering adapter: {adapter}")
        if lowering.get("execution_format") != supported[adapter]:
            raise WorkloadLoweringError(f"adapter/format mismatch for {graph_id}")
        orders[str(graph_id)] = topological_order(graph)
    return orders


def _trace_medians(trace: Mapping[str, Any]) -> dict[str, float]:
    return {
        str(record["key"]): float(record["timing"]["median_ms"])
        for record in trace["cases"]
    }


def _fig23_correction(
    graph: Mapping[str, Any], selected: Mapping[str, Any], trace: Mapping[str, Any]
) -> dict[str, Any]:
    model = graph["model"]
    lowering = graph["lowering"]
    sequence = int(model["sequence_length"])
    window = int(lowering["active_window"])
    hardware = str(lowering["hardware_name"])
    knee = 2048
    medians = _trace_medians(trace)
    knee_trace = medians[f"fig23-N{knee}-fft_cmp"] + medians[f"fig23-N{knee}-bsmm"]
    current_trace = (
        medians[f"fig23-N{sequence}-fft_cmp"] + medians[f"fig23-N{sequence}-bsmm"]
    )
    underfill = max(0.0, 1.0 - sequence / knee)
    growth = max(0.0, current_trace / knee_trace - 1.0)
    parameters = selected["figure23"]["parameters"]
    credit = 0
    congestion = 0
    if hardware == "simd8_8x8":
        congestion = round(
            float(parameters["mesh_post_knee_congestion_cycles_per_trace_ratio"]) * growth
        )
    if hardware == "simd32_8x8":
        credit = round(
            float(parameters[f"joint_w{window}_underfill_startup_credit_cycles"])
            * underfill
        )
        congestion = round(
            float(parameters["joint_post_knee_congestion_cycles_per_trace_ratio"])
            * growth
        )
    return {
        "startup_credit_cycles": int(credit),
        "congestion_cycles": int(congestion),
        "underfill_feature": underfill,
        "post_knee_trace_growth": growth,
    }


def _lower_fig23(
    graph_id: str,
    graph: Mapping[str, Any],
    contexts: Mapping[str, Any],
    output_root: Path,
    project_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    model = graph["model"]
    lowering = graph["lowering"]
    base = contexts[str(lowering["source_ref"])]
    document, metadata = compile_complete_block_scaling(
        base,
        sequence_length=int(model["sequence_length"]),
        hidden_dimension=int(model["hidden_dimension"]),
        batch=int(model["batch"]),
        active_window=int(lowering["active_window"]),
        baseline_repeat=int(lowering["baseline_repeat"]),
        hardware_name=str(lowering["hardware_name"]),
        simd_width=int(lowering["simd_width"]),
        mesh=tuple(int(value) for value in lowering["mesh"]),
    )
    correction = _fig23_correction(
        graph,
        contexts[str(lowering["latency_service_ref"])],
        contexts[str(lowering["trace_ref"])],
    )
    document["latency_service"] = {
        "enabled": True,
        "model": "trace_knee_underfill_congestion",
        "startup_credit_cycles": correction["startup_credit_cycles"],
        "congestion_cycles": correction["congestion_cycles"],
        "target_informed": True,
        "provenance": "H183.figure23.parameters+H182.RTX4090.trace_features",
    }
    document["metadata"].update(
        {
            "experiment_id": "H187",
            "workload_graph": graph_id,
            "operator_order": topological_order(graph),
        }
    )
    unit_id = f"{graph_id}:overlay"
    artifact = _write_replay(
        value=document,
        primary=output_root / "lowered" / graph_id / "overlay.json",
        replay=output_root / "replay" / graph_id / "overlay.json",
        project_root=project_root,
        canonical_overlay=True,
    )
    unit = {
        "unit_id": unit_id,
        "graph_id": graph_id,
        "node_ids": topological_order(graph),
        "execution_format": lowering["execution_format"],
        "artifacts": {"overlay": artifact},
        "metadata": {**metadata, "latency_correction": correction},
    }
    return [unit], {node: [unit_id] for node in topological_order(graph)}


def _lower_fig19(
    graph_id: str,
    graph: Mapping[str, Any],
    contexts: Mapping[str, Any],
    output_root: Path,
    project_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    model = graph["model"]
    lowering = graph["lowering"]
    source_manifest = contexts[str(lowering["source_manifest_ref"])]
    source_config = contexts[str(lowering["source_config_ref"])]
    coupled_config = deepcopy(contexts[str(lowering["hardware_config_ref"])])
    coupled_config["experiment_id"] = "H187"
    sequence = int(model["sequence_length"])
    scale = int(lowering["scale"])
    active_window = int(source_config["hardware"]["active_window"])
    vector_bytes = int(source_config["hardware"]["vector_bytes"])
    nodes = {str(node["id"]): node for node in graph["operators"]}
    units: list[dict[str, Any]] = []
    lineage: dict[str, list[str]] = {}
    for node_id in topological_order(graph):
        node = nodes[node_id]
        if node["kind"] == "fft2d":
            path_key = f"N{sequence}-fft2d"
            run_key = f"{path_key}-q{scale}"
            source, source_metadata = compile_fft2d_path(
                name=run_key,
                sequence_length=sequence,
                scale=scale,
                vector_bytes=vector_bytes,
                active_window=active_window,
            )
        else:
            suffix = "global_ffn1" if node_id == "global_ffn1" else "global_ffn2"
            path_key = f"N{sequence}-{suffix}"
            run_key = f"{path_key}-q{scale}"
            source, source_metadata = compile_timed_path(
                name=run_key,
                normalized=source_manifest["path_contracts"][path_key],
                scale=scale,
                active_window=active_window,
            )
        overlay, memory, metadata = compile_fig19_coupled_path(
            run_key=run_key,
            source=source,
            source_metadata=source_metadata,
            config=coupled_config,
        )
        overlay["metadata"].update(
            {
                "experiment_id": "H187",
                "workload_graph": graph_id,
                "operator_node": node_id,
            }
        )
        memory["metadata"].update(
            {"experiment_id": "H187", "workload_graph": graph_id, "operator_node": node_id}
        )
        unit_id = f"{graph_id}:{node_id}"
        overlay_artifact = _write_replay(
            value=overlay,
            primary=output_root / "lowered" / graph_id / f"{node_id}-overlay.json",
            replay=output_root / "replay" / graph_id / f"{node_id}-overlay.json",
            project_root=project_root,
            canonical_overlay=True,
        )
        memory_artifact = _write_replay(
            value=memory,
            primary=output_root / "lowered" / graph_id / f"{node_id}-memory.json",
            replay=output_root / "replay" / graph_id / f"{node_id}-memory.json",
            project_root=project_root,
            canonical_overlay=True,
        )
        units.append(
            {
                "unit_id": unit_id,
                "graph_id": graph_id,
                "node_ids": [node_id],
                "execution_format": lowering["execution_format"],
                "artifacts": {"overlay": overlay_artifact, "memory": memory_artifact},
                "metadata": metadata,
            }
        )
        lineage[node_id] = [unit_id]
    return units, lineage


def _analytical_workload(node: Mapping[str, Any], model: Mapping[str, Any], n: int) -> Workload:
    hidden = int(model["hidden_dimension"])
    ffn = int(model["ffn_dimension"])
    node_id = str(node["id"])
    if node_id == "qkv":
        kernel, dimension, output = "bsmm", hidden, hidden
    elif node_id == "attention":
        kernel, dimension, output = "fft_cmp", hidden, hidden
    elif node_id == "ffn1":
        kernel, dimension, output = "bsmm", hidden, ffn
    elif node_id == "ffn2":
        kernel, dimension, output = "bsmm", ffn, hidden
    else:
        raise WorkloadLoweringError(f"unsupported analytical operator: {node_id}")
    return Workload(
        kernel=kernel,
        n=n,
        d=dimension,
        output_dim=output,
        batch=int(model["batch"]),
        block_size=int(model["block_size"]),
        compression_ratio=float(model["compression_ratio"]),
        chunk_length=int(model["chunk_length"]),
        projections=int(node.get("projections", 1)),
        name=f"{node_id}-N{n}",
    )


def _lower_fig20(
    graph_id: str,
    graph: Mapping[str, Any],
    contexts: Mapping[str, Any],
    output_root: Path,
    project_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    del contexts
    model = graph["model"]
    lowering = graph["lowering"]
    nodes = {str(node["id"]): node for node in graph["operators"]}
    units: list[dict[str, Any]] = []
    lineage = {node_id: [] for node_id in nodes}
    for sequence in model["sequence_lengths"]:
        for node_id in topological_order(graph):
            workload = _analytical_workload(nodes[node_id], model, int(sequence))
            profile = compile_workload(workload)
            unit_id = f"{graph_id}:{node_id}:N{sequence}"
            value = {
                "schema_version": 1,
                "graph_id": graph_id,
                "node_id": node_id,
                "execution_format": lowering["execution_format"],
                "workload": workload.to_dict(),
                "profile": asdict(profile),
                "hardware_config_ref": lowering["hardware_config_ref"],
                "performance_service_ref": lowering["performance_service_ref"],
            }
            artifact = _write_replay(
                value=value,
                primary=output_root / "lowered" / graph_id / f"{node_id}-N{sequence}.json",
                replay=output_root / "replay" / graph_id / f"{node_id}-N{sequence}.json",
                project_root=project_root,
                canonical_overlay=False,
            )
            units.append(
                {
                    "unit_id": unit_id,
                    "graph_id": graph_id,
                    "node_ids": [node_id],
                    "execution_format": lowering["execution_format"],
                    "artifacts": {"profile": artifact},
                    "metadata": {
                        "sequence_length": int(sequence),
                        "stage_count": len(profile.stages),
                        "operations": profile.operations,
                        "offchip_bytes": profile.offchip_bytes,
                        "output_elements": profile.output_elements,
                    },
                }
            )
            lineage[node_id].append(unit_id)
    return units, lineage


def lower_suite(
    *,
    spec: Mapping[str, Any],
    contexts: Mapping[str, Any],
    output_root: Path,
    project_root: Path,
) -> dict[str, Any]:
    orders = validate_suite(spec)
    adapters = {
        "fig23_complete_block": _lower_fig23,
        "fig19_coupled_paths": _lower_fig19,
        "analytical_kernel_profiles": _lower_fig20,
    }
    units: list[dict[str, Any]] = []
    lineage: dict[str, dict[str, list[str]]] = {}
    for graph_id, graph in spec["graphs"].items():
        adapter = adapters[graph["lowering"]["adapter"]]
        graph_units, graph_lineage = adapter(
            str(graph_id), graph, contexts, output_root, project_root
        )
        units.extend(graph_units)
        lineage[str(graph_id)] = graph_lineage
    format_counts = {
        execution_format: sum(unit["execution_format"] == execution_format for unit in units)
        for execution_format in (
            "mlx_overlay_json",
            "mlx_dpu_memory_json",
            "analytical_kernel_profile_json",
        )
    }
    memory_configs = sum("memory" in unit["artifacts"] for unit in units)
    all_artifacts = [
        artifact
        for unit in units
        for pair in unit["artifacts"].values()
        for artifact in (pair["primary"], pair["replay"])
    ]
    checks = {
        "graphs": len(orders) == 3,
        "nodes": sum(len(order) for order in orders.values()) == 14,
        "units": len(units) == 12,
        "formats": format_counts
        == {
            "mlx_overlay_json": 1,
            "mlx_dpu_memory_json": 3,
            "analytical_kernel_profile_json": 8,
        },
        "memory_configs": memory_configs == 3,
        "replay": all(
            pair["identical"] for unit in units for pair in unit["artifacts"].values()
        ),
        "artifacts": all(artifact["bytes"] > 0 for artifact in all_artifacts),
        "lineage": all(
            set(graph_lineage) == set(orders[graph_id])
            and all(unit_ids for unit_ids in graph_lineage.values())
            for graph_id, graph_lineage in lineage.items()
        ),
    }
    return {
        "schema_version": 1,
        "suite_id": spec["suite_id"],
        "topological_orders": orders,
        "units": units,
        "lineage": lineage,
        "format_counts": format_counts,
        "memory_configs": memory_configs,
        "checks": checks,
    }


__all__ = [
    "WorkloadLoweringError",
    "lower_suite",
    "topological_order",
    "validate_suite",
]
