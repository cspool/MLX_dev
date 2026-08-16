from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AttentionShape:
    """Shape needed for a QKV-plus-attention operation-count audit."""

    name: str
    sequence_length: int
    hidden_size: int
    query_heads: int
    key_value_heads: int

    def __post_init__(self) -> None:
        for field_name in ("sequence_length", "hidden_size", "query_heads", "key_value_heads"):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.query_heads % self.key_value_heads != 0:
            raise ValueError("query_heads must be divisible by key_value_heads")

    @property
    def qkv_projection_coefficient(self) -> float:
        """Dense D-by-output-D factors for Q plus grouped K and V."""

        return 1.0 + 2.0 * self.key_value_heads / self.query_heads

    @property
    def dense_qkv_operations(self) -> float:
        return self.qkv_projection_coefficient * self.sequence_length * self.hidden_size**2

    @property
    def dense_attention_operations(self) -> float:
        # QK^T and AV; the common multiply/add convention cancels in all ratios.
        return 2.0 * self.sequence_length**2 * self.hidden_size

    @property
    def dense_operations(self) -> float:
        return self.dense_qkv_operations + self.dense_attention_operations


def hierarchical_bsmm_density(block_size: int) -> float:
    """Return Eq. (2)'s 2*log2(B)/B nonzero/operation ratio."""

    if block_size < 2 or block_size & (block_size - 1):
        raise ValueError("block_size must be a power of two greater than or equal to two")
    return 2.0 * math.log2(block_size) / block_size


def hybrid_compute_remaining(
    shape: AttentionShape,
    *,
    block_size: int,
    compression_ratio: float,
) -> float:
    """Analytical remaining QKV+attention operations for the MLX hybrid method.

    The disclosed leading terms are used directly: hierarchical BSMM scales QKV
    projection by 2*log2(B)/B and semantic compression scales the quadratic
    attention term by s^2. The paper calls the FFT term minor and does not disclose
    enough implementation detail to count it, so it is excluded and reported as a
    protocol limitation by the figure runner.
    """

    if not 0.0 < compression_ratio <= 1.0:
        raise ValueError("compression_ratio must be in (0, 1]")
    structured_qkv = hierarchical_bsmm_density(block_size) * shape.dense_qkv_operations
    compressed_attention = compression_ratio**2 * shape.dense_attention_operations
    return (structured_qkv + compressed_attention) / shape.dense_operations


def mixed_layer_compute_remaining(
    modified_layer_ratio: float,
    *,
    total_layers: int,
    modified_layers: int,
) -> float:
    """Blend dense and modified layers for the BERT last-k-layer sweep."""

    if total_layers <= 0:
        raise ValueError("total_layers must be positive")
    if not 0 <= modified_layers <= total_layers:
        raise ValueError("modified_layers must be between zero and total_layers")
    modified_fraction = modified_layers / total_layers
    return 1.0 - modified_fraction * (1.0 - modified_layer_ratio)
