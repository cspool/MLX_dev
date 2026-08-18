"""Executable one-layer work contracts for Figure 21 Llama2 shapes."""

from __future__ import annotations

import math
from typing import Any

from mlxsim.attention_signature import (
    attention_work_signature,
    compressed_attention_signature,
)
from mlxsim.dsagen_combined_attention import compile_combined_attention

FFT_TEMPLATE = {
    "fma_per_pair": 4,
    "add_per_pair": 6,
    "analytical_flops_per_pair": 10,
    "shuffle_per_retained_element": 1,
}


def elementwise_signature(
    *, sequence_length: int, batch: int, hidden_dimension: int, ffn_dimension: int
) -> dict[str, Any]:
    tokens = batch * sequence_length
    counts = {
        "mul": tokens * (8 * hidden_dimension + 2 * ffn_dimension),
        "add": tokens * (6 * hidden_dimension + ffn_dimension),
        "frsqrt": tokens * 2,
        "fexp": tokens * ffn_dimension,
        "fdiv": tokens * ffn_dimension,
        "shuffle": tokens * 2 * hidden_dimension,
    }
    return {
        "evidence": "inferred_llama2_semantics",
        "tokens": tokens,
        "fu_instruction_instances": counts,
        "operation_count": sum(counts.values()),
    }


def _combined_attention_fu(signature: dict[str, Any]) -> dict[str, int]:
    fft = signature["fft_compression"]["fu_instruction_instances"]
    attention = signature["compressed_attention"]["fu_instruction_instances"]
    return {
        "fma": int(fft["fma"] + attention["fma"]),
        "add": int(fft["alu_add"] + attention["alu_add"]),
        "shuffle": int(fft["shuffle"]),
        "fmax": int(attention["fmax"]),
        "fexp": int(attention["fexp"]),
        "fdiv": int(attention["fdiv"]),
    }


def build_shape_contract(
    *,
    sequence_length: int,
    batch: int,
    hidden_dimension: int,
    ffn_dimension: int,
    simd_width: int,
    vector_bytes: int,
    active_window: int,
    logical_profile: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    retained = sequence_length // 2
    if sequence_length % 2 or hidden_dimension % sequence_length:
        raise ValueError("Figure 21 shapes require even N dividing D")
    full_scale = batch * retained * retained // 128
    fft_scale_per_u = hidden_dimension // sequence_length
    attention_scale_per_u = 1
    document, unit_metadata = compile_combined_attention(
        name=f"fig21-N{sequence_length}-u1",
        sequence_length=sequence_length,
        retained_length=retained,
        hidden_dimension=hidden_dimension,
        forward_stages=int(math.log2(sequence_length)),
        inverse_stages=int(math.log2(retained)),
        fft_scale=fft_scale_per_u,
        attention_scale=attention_scale_per_u,
        vector_bytes=vector_bytes,
        active_window=active_window,
    )
    structured_attention = attention_work_signature(
        sequence_length=sequence_length,
        hidden_dimension=hidden_dimension,
        batch=batch,
        projections=3,
        compression_ratio=0.5,
        fft_template=FFT_TEMPLATE,
    )
    expected_fu = _combined_attention_fu(structured_attention)
    derived_fu = {
        operation: int(unit_metadata["operation_counts"][operation])
        * full_scale
        * simd_width
        for operation in expected_fu
    }
    input_bytes = batch * sequence_length * hidden_dimension * 3 * 2
    boundary_bytes = batch * retained * hidden_dimension * 3 * 2
    output_bytes = batch * retained * hidden_dimension * 2
    combined_offchip_bytes = input_bytes + output_bytes
    isolated_offchip_bytes = int(logical_profile["structured"]["attention"]["offchip_bytes"])

    dense_attention = compressed_attention_signature(
        retained_length=sequence_length,
        hidden_dimension=hidden_dimension,
        batch=batch,
    )
    dense_attention_fu = dense_attention["fu_instruction_instances"]
    structured_components = {}
    dense_components = {}
    for component in ("qkv", "output", "ffn1", "ffn2"):
        structured_profile = logical_profile["structured"][component]
        dense_profile = logical_profile["dense"][component]
        structured_components[component] = {
            **structured_profile,
            "fma_equivalents": int(structured_profile["operations"] / 2),
            "bsmm_stage_count": 5,
            "execution_convention": "analytical_fma_equivalent",
        }
        dense_components[component] = {
            **dense_profile,
            "fma_equivalents": int(dense_profile["operations"] / 2),
            "gemm_stage_count": 1,
            "execution_convention": "analytical_fma_equivalent",
        }
    structured_components["attention"] = {
        **logical_profile["structured"]["attention"],
        "fu_instruction_instances": expected_fu,
        "combined_offchip_bytes": combined_offchip_bytes,
        "isolated_offchip_bytes": isolated_offchip_bytes,
        "removed_roundtrip_bytes": 2 * boundary_bytes,
        "boundary_bytes": boundary_bytes,
    }
    dense_components["attention"] = {
        **logical_profile["dense"]["attention"],
        "fu_instruction_instances": {
            "fma": int(dense_attention_fu["fma"]),
            "fmax": int(dense_attention_fu["fmax"]),
            "fexp": int(dense_attention_fu["fexp"]),
            "add": int(dense_attention_fu["alu_add"]),
            "fdiv": int(dense_attention_fu["fdiv"]),
        },
    }
    elementwise = elementwise_signature(
        sequence_length=sequence_length,
        batch=batch,
        hidden_dimension=hidden_dimension,
        ffn_dimension=ffn_dimension,
    )
    checks = {
        "attention_fu": derived_fu == expected_fu,
        "attention_analytical_operations": structured_attention[
            "analytical_operations_excluding_fdiv"
        ]
        == logical_profile["structured"]["attention"]["operations"],
        "dense_attention_operations": dense_attention[
            "analytical_operations_excluding_fdiv"
        ]
        == logical_profile["dense"]["attention"]["operations"],
        "combined_offchip": unit_metadata["offchip_bytes"] * full_scale
        == combined_offchip_bytes,
        "boundary": unit_metadata["boundary_bytes"] * full_scale
        == boundary_bytes,
        "isolated_roundtrip": isolated_offchip_bytes - combined_offchip_bytes
        == 2 * boundary_bytes,
        "instruction_footprint": unit_metadata[
            "max_active_instruction_footprint_per_pe"
        ]
        <= 32,
        "output_projection": "output" in structured_components
        and "output" in dense_components,
        "elementwise_positive": all(
            value > 0 for value in elementwise["fu_instruction_instances"].values()
        ),
    }
    contract = {
        "sequence_length": sequence_length,
        "batch": batch,
        "retained_length": retained,
        "full_scale": full_scale,
        "fft_scale_per_u": fft_scale_per_u,
        "attention_scale_per_u": attention_scale_per_u,
        "structured_components": structured_components,
        "dense_components": dense_components,
        "elementwise": elementwise,
        "unit_attention_metadata": unit_metadata,
        "checks": checks,
    }
    return document, contract


__all__ = ["FFT_TEMPLATE", "build_shape_contract", "elementwise_signature"]
