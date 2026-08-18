"""Recompile H102 exact paths with H109 pipelined iteration contexts."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from mlxsim.fig24_25_full_mesh_paths import (
    compile_full_mesh_fft_cmp_path,
    compile_full_mesh_swa_path,
    compile_full_mesh_timed_path,
)


def recompile_h102_path(
    *,
    run_key: str,
    contract: dict[str, Any],
    scale: int,
    active_window: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    family = contract["family"]
    case, operator = contract["case"], contract["operator"]
    if family == "qkv_bsmm":
        return compile_full_mesh_timed_path(
            name=run_key,
            normalized=contract["normalized"],
            scale=scale,
            active_window=active_window,
        )
    if family == "fft":
        return compile_full_mesh_fft_cmp_path(
            name=run_key,
            sequence_length=int(case["n"]),
            hidden_dimension=int(case["d"]),
            batch=int(case["batch"]),
            scale=scale,
        )
    if family == "swa":
        return compile_full_mesh_swa_path(
            name=run_key,
            sequence_length=int(case["n"]),
            hidden_dimension=int(case["d"]),
            batch=int(case["batch"]),
            window=int(operator["window"]),
            query_tile=int(operator["query_tile"]),
            scale=scale,
        )
    raise ValueError(f"unknown H110 family: {family}")


def convert_to_pipelined(
    *,
    document: dict[str, Any],
    metadata: dict[str, Any],
    contexts: int,
    operand_contexts_per_pe: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    converted = deepcopy(document)
    converted["pe_dependency_model"] = "dpu_pipelined"
    converted["dpu"] = {
        "instruction_slots_per_pe": 0,
        "operand_contexts_per_pe": operand_contexts_per_pe,
        "active_blocks_per_pe": 0,
        "iteration_contexts_per_block": contexts,
    }
    converted_metadata = deepcopy(metadata)
    converted_metadata.update(
        {
            "experiment_id": "H110",
            "parent_experiment_id": "H102",
            "pe_dependency_model": "dpu_pipelined",
            "iteration_contexts_per_block": contexts,
            "operand_contexts_per_pe": operand_contexts_per_pe,
            "paper_performance_targets_consumed": False,
        }
    )
    converted["metadata"] = converted_metadata
    return converted, converted_metadata


def compile_pipelined_path(
    *,
    run_key: str,
    contract: dict[str, Any],
    scale: int,
    active_window: int,
    contexts: int,
    operand_contexts_per_pe: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    original, original_metadata = recompile_h102_path(
        run_key=run_key,
        contract=contract,
        scale=scale,
        active_window=active_window,
    )
    converted, metadata = convert_to_pipelined(
        document=original,
        metadata=original_metadata,
        contexts=contexts,
        operand_contexts_per_pe=operand_contexts_per_pe,
    )
    return converted, metadata, original


__all__ = [
    "compile_pipelined_path",
    "convert_to_pipelined",
    "recompile_h102_path",
]

