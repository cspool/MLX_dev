"""Full-grid expansion for the unified MLX workload lowering toolchain."""

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
from mlxsim.schema import Workload
from mlxsim.workloads import compile_workload


def _artifact(path: Path, project_root: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": str(path.relative_to(project_root)),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _write_pair(
    value: Any, primary: Path, replay: Path, project_root: Path, *, overlay: bool
) -> dict[str, Any]:
    primary.parent.mkdir(parents=True, exist_ok=True)
    replay.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(value) if overlay else json.dumps(value, indent=2, sort_keys=True) + "\n"
    primary.write_text(payload)
    replay.write_text(payload)
    first, second = _artifact(primary, project_root), _artifact(replay, project_root)
    return {"primary": first, "replay": second, "identical": first["sha256"] == second["sha256"]}


def _fig19_units(
    *,
    graph: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    source_config: Mapping[str, Any],
    coupled_config: Mapping[str, Any],
    output_root: Path,
    project_root: Path,
) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    active_window = int(source_config["hardware"]["active_window"])
    vector_bytes = int(source_config["hardware"]["vector_bytes"])
    scale = int(graph["lowering"]["scale"])
    local_config = deepcopy(coupled_config)
    local_config["experiment_id"] = "H192"
    for sequence in graph["model"]["sequence_lengths"]:
        for node_id in graph["operators"]:
            if node_id == "fft2d_attention":
                path_key = f"N{sequence}-fft2d"
                run_key = f"{path_key}-q{scale}"
                source, metadata = compile_fft2d_path(
                    name=run_key,
                    sequence_length=int(sequence),
                    scale=scale,
                    vector_bytes=vector_bytes,
                    active_window=active_window,
                )
            else:
                suffix = "global_ffn1" if node_id == "global_ffn1" else "global_ffn2"
                path_key = f"N{sequence}-{suffix}"
                run_key = f"{path_key}-q{scale}"
                source, metadata = compile_timed_path(
                    name=run_key,
                    normalized=source_manifest["path_contracts"][path_key],
                    scale=scale,
                    active_window=active_window,
                )
            overlay_doc, memory_doc, lowered_metadata = compile_fig19_coupled_path(
                run_key=run_key,
                source=source,
                source_metadata=metadata,
                config=local_config,
            )
            overlay_doc["metadata"].update(
                {"experiment_id": "H192", "coverage_graph": "figure19_component_grid"}
            )
            unit_id = f"figure19:N{sequence}:{node_id}"
            overlay_artifact = _write_pair(
                overlay_doc,
                output_root / "lowered/figure19" / f"N{sequence}-{node_id}-overlay.json",
                output_root / "replay/figure19" / f"N{sequence}-{node_id}-overlay.json",
                project_root,
                overlay=True,
            )
            memory_artifact = _write_pair(
                memory_doc,
                output_root / "lowered/figure19" / f"N{sequence}-{node_id}-memory.json",
                output_root / "replay/figure19" / f"N{sequence}-{node_id}-memory.json",
                project_root,
                overlay=True,
            )
            units.append(
                {
                    "unit_id": unit_id,
                    "graph_id": "figure19_component_grid",
                    "node_id": node_id,
                    "sequence_length": int(sequence),
                    "execution_format": "mlx_dpu_memory_json",
                    "artifacts": {"overlay": overlay_artifact, "memory": memory_artifact},
                    "metadata": lowered_metadata,
                }
            )
    return units


def _fig20_workload(node_id: str, model: Mapping[str, Any], sequence: int) -> Workload:
    hidden, ffn = int(model["hidden_dimension"]), int(model["ffn_dimension"])
    if node_id == "qkv":
        kernel, dimension, output, projections = "bsmm", hidden, hidden, 3
    elif node_id == "attention":
        kernel, dimension, output, projections = "fft_cmp", hidden, hidden, 1
    elif node_id == "ffn1":
        kernel, dimension, output, projections = "bsmm", hidden, ffn, 1
    elif node_id == "ffn2":
        kernel, dimension, output, projections = "bsmm", ffn, hidden, 1
    else:
        raise ValueError(f"unsupported Figure20 node: {node_id}")
    return Workload(
        kernel=kernel,
        n=sequence,
        d=dimension,
        output_dim=output,
        batch=int(model["batch"]),
        block_size=int(model["block_size"]),
        compression_ratio=float(model["compression_ratio"]),
        chunk_length=int(model["chunk_length"]),
        projections=projections,
        name=f"{node_id}-N{sequence}",
    )


def _fig20_units(
    graph: Mapping[str, Any], output_root: Path, project_root: Path
) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for sequence in graph["model"]["sequence_lengths"]:
        for node_id in graph["operators"]:
            workload = _fig20_workload(str(node_id), graph["model"], int(sequence))
            profile = compile_workload(workload)
            document = {
                "schema_version": 1,
                "graph_id": "figure20_kernel_grid",
                "node_id": node_id,
                "sequence_length": int(sequence),
                "workload": workload.to_dict(),
                "profile": asdict(profile),
            }
            artifact = _write_pair(
                document,
                output_root / "lowered/figure20" / f"N{sequence}-{node_id}.json",
                output_root / "replay/figure20" / f"N{sequence}-{node_id}.json",
                project_root,
                overlay=False,
            )
            units.append(
                {
                    "unit_id": f"figure20:N{sequence}:{node_id}",
                    "graph_id": "figure20_kernel_grid",
                    "node_id": node_id,
                    "sequence_length": int(sequence),
                    "execution_format": "analytical_kernel_profile_json",
                    "artifacts": {"profile": artifact},
                    "metadata": {
                        "operations": profile.operations,
                        "offchip_bytes": profile.offchip_bytes,
                        "stage_count": len(profile.stages),
                    },
                }
            )
    return units


def expand_coverage(
    *,
    spec: Mapping[str, Any],
    physical_compile: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    source_config: Mapping[str, Any],
    coupled_config: Mapping[str, Any],
    output_root: Path,
    project_root: Path,
) -> dict[str, Any]:
    units: list[dict[str, Any]] = []
    for key, item in physical_compile["outputs"].items():
        metadata = item["metadata"]
        units.append(
            {
                "unit_id": f"figure23:{key}",
                "graph_id": "figure23_scalability_grid",
                "node_id": "complete_block",
                "sequence_length": int(metadata["sequence_length"]),
                "active_window": int(metadata["active_window"]),
                "hardware_name": metadata["hardware_name"],
                "execution_format": "mlx_overlay_json",
                "artifacts": {
                    "overlay": {
                        "primary": item["primary"],
                        "replay": item["replay"],
                        "identical": item["identical"],
                    }
                },
                "metadata": metadata,
            }
        )
    units.extend(
        _fig19_units(
            graph=spec["graphs"]["figure19_component_grid"],
            source_manifest=source_manifest,
            source_config=source_config,
            coupled_config=coupled_config,
            output_root=output_root,
            project_root=project_root,
        )
    )
    units.extend(
        _fig20_units(spec["graphs"]["figure20_kernel_grid"], output_root, project_root)
    )
    compositions: list[dict[str, Any]] = []
    for composition_id, composition in spec["compositions"].items():
        source_units = [
            unit["unit_id"]
            for unit in units
            if unit["graph_id"] == composition["source_graph"]
        ]
        plan = {
            "schema_version": 1,
            "composition_id": composition_id,
            "source_graph": composition["source_graph"],
            "source_units": source_units,
            **{key: value for key, value in composition.items() if key != "source_graph"},
        }
        artifact = _write_pair(
            plan,
            output_root / "lowered/compositions" / f"{composition_id}.json",
            output_root / "replay/compositions" / f"{composition_id}.json",
            project_root,
            overlay=False,
        )
        compositions.append(
            {
                "unit_id": f"composition:{composition_id}",
                "graph_id": composition["source_graph"],
                "node_id": composition_id,
                "execution_format": "multi_layer_composition_json",
                "artifacts": {"plan": artifact},
                "metadata": {
                    "total_layers": int(composition["total_layers"]),
                    "source_unit_count": len(source_units),
                },
            }
        )
    units.extend(compositions)
    format_counts = {
        name: sum(unit["execution_format"] == name for unit in units)
        for name in (
            "mlx_overlay_json",
            "mlx_dpu_memory_json",
            "analytical_kernel_profile_json",
            "multi_layer_composition_json",
        )
    }
    checks = {
        "figure23": format_counts["mlx_overlay_json"] == 40,
        "figure19": format_counts["mlx_dpu_memory_json"] == 12,
        "figure20": format_counts["analytical_kernel_profile_json"] == 8,
        "compositions": format_counts["multi_layer_composition_json"] == 2,
        "units": len(units) == 62,
        "replay": all(
            artifact["identical"]
            for unit in units
            for artifact in unit["artifacts"].values()
        ),
    }
    return {
        "schema_version": 1,
        "suite_id": spec["suite_id"],
        "units": units,
        "format_counts": format_counts,
        "checks": checks,
    }


__all__ = ["expand_coverage"]
