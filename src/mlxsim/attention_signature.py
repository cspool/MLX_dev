"""Target-free per-FU work signatures for MLX compressed attention."""

from __future__ import annotations

import math
from typing import Any


def _log2(value: int) -> int:
    if value < 1 or value & (value - 1):
        raise ValueError(f"expected a positive power of two, got {value}")
    return value.bit_length() - 1


def fft_compression_signature(
    *,
    sequence_length: int,
    hidden_dimension: int,
    batch: int,
    projections: int,
    compression_ratio: float,
    fma_per_pair: int,
    add_per_pair: int,
    analytical_flops_per_pair: int,
    shuffle_per_retained_element: int,
) -> dict[str, Any]:
    retained = int(sequence_length * compression_ratio)
    if not math.isclose(retained, sequence_length * compression_ratio):
        raise ValueError("compression must produce an integral retained length")
    forward_stages = _log2(sequence_length)
    inverse_stages = _log2(retained)
    vectors = batch * hidden_dimension * projections
    forward_pairs = vectors * (sequence_length // 2) * forward_stages
    inverse_pairs = vectors * (retained // 2) * inverse_stages
    pairs = forward_pairs + inverse_pairs
    fma = pairs * fma_per_pair
    add = pairs * add_per_pair
    shuffle = vectors * retained * shuffle_per_retained_element
    return {
        "sequence_length": sequence_length,
        "retained_length": retained,
        "vectors": vectors,
        "forward_stages": forward_stages,
        "inverse_stages": inverse_stages,
        "tagged_stage_count": forward_stages + 1 + inverse_stages,
        "analytical_stage_count_excluding_shuffle": forward_stages
        + inverse_stages,
        "forward_butterfly_pairs": forward_pairs,
        "inverse_butterfly_pairs": inverse_pairs,
        "butterfly_pairs": pairs,
        "analytical_operations": pairs * analytical_flops_per_pair,
        "execution_weighted_flops_excluding_shuffle": 2 * fma + add,
        "fu_instruction_instances": {
            "fma": fma,
            "alu_add": add,
            "shuffle": shuffle,
        },
    }


def compressed_attention_signature(
    *, retained_length: int, hidden_dimension: int, batch: int
) -> dict[str, Any]:
    score_elements = batch * retained_length * retained_length
    output_elements = batch * retained_length * hidden_dimension
    fma = 2 * score_elements * hidden_dimension
    return {
        "retained_length": retained_length,
        "hidden_dimension": hidden_dimension,
        "tagged_stage_count": 4,
        "score_elements": score_elements,
        "output_elements": output_elements,
        "analytical_operations_excluding_fdiv": (
            2 * fma + score_elements + 4 * score_elements + score_elements
        ),
        "fu_instruction_instances": {
            "fma": fma,
            "fmax": score_elements,
            "fexp": score_elements,
            "alu_add": score_elements,
            "fdiv": output_elements,
        },
    }


def attention_work_signature(
    *,
    sequence_length: int,
    hidden_dimension: int,
    batch: int,
    projections: int,
    compression_ratio: float,
    fft_template: dict[str, int],
) -> dict[str, Any]:
    fft = fft_compression_signature(
        sequence_length=sequence_length,
        hidden_dimension=hidden_dimension,
        batch=batch,
        projections=projections,
        compression_ratio=compression_ratio,
        **fft_template,
    )
    attention = compressed_attention_signature(
        retained_length=fft["retained_length"],
        hidden_dimension=hidden_dimension,
        batch=batch,
    )
    return {
        "sequence_length": sequence_length,
        "fft_compression": fft,
        "compressed_attention": attention,
        "analytical_operations_excluding_fdiv": fft["analytical_operations"]
        + attention["analytical_operations_excluding_fdiv"],
        "required_fdiv_instructions": attention["fu_instruction_instances"]["fdiv"],
    }


__all__ = [
    "attention_work_signature",
    "compressed_attention_signature",
    "fft_compression_signature",
]
