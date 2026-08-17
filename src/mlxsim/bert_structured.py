"""Inferred BERT wiring for MLX-style compressed attention and tiled QKV."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .structured import (
    FFTCompressionContext,
    HierarchicalButterflyLinear,
    chunked_fft_compress,
    chunked_fft_decompress,
)


class CompressedBertSelfAttention(nn.Module):
    """BERT encoder self-attention under the frozen H15 inferred semantics."""

    def __init__(
        self,
        original: nn.Module,
        *,
        compression_ratio: float,
        chunk_length: int,
        block_size: int,
    ) -> None:
        super().__init__()
        if getattr(original, "position_embedding_type", "absolute") != "absolute":
            raise ValueError("H15 only supports BERT absolute position embeddings")
        self.num_attention_heads = int(original.num_attention_heads)
        self.attention_head_size = int(original.attention_head_size)
        self.all_head_size = int(original.all_head_size)
        self.scaling = float(original.scaling)
        self.dropout = original.dropout
        self.compression_ratio = compression_ratio
        self.chunk_length = chunk_length
        self.block_size = block_size
        self.query = HierarchicalButterflyLinear(
            original.query.in_features,
            original.query.out_features,
            block_size=block_size,
            bias=original.query.bias is not None,
        )
        self.key = HierarchicalButterflyLinear(
            original.key.in_features,
            original.key.out_features,
            block_size=block_size,
            bias=original.key.bias is not None,
        )
        self.value = HierarchicalButterflyLinear(
            original.value.in_features,
            original.value.out_features,
            block_size=block_size,
            bias=original.value.bias is not None,
        )

    @staticmethod
    def _key_validity(attention_mask: torch.Tensor) -> torch.Tensor:
        key_mask = attention_mask[:, 0, 0, :]
        if key_mask.dtype == torch.bool:
            return key_mask
        return key_mask >= 0

    def compressed_attention_mask(
        self, attention_mask: torch.Tensor | None, context: FFTCompressionContext
    ) -> torch.Tensor | None:
        if attention_mask is None:
            return None
        valid = self._key_validity(attention_mask)
        if valid.shape[-1] != context.original_length:
            raise ValueError("attention mask token length does not match QKV")
        if context.padded_length > context.original_length:
            valid = torch.nn.functional.pad(
                valid, (0, context.padded_length - context.original_length), value=False
            )
        chunk_valid = valid.reshape(valid.shape[0], context.num_chunks, context.chunk_length).any(
            dim=-1
        )
        compressed_valid = chunk_valid.repeat_interleave(
            context.compressed_chunk_length, dim=-1
        )
        compressed_tokens = compressed_valid.shape[-1]
        return compressed_valid[:, None, None, :].expand(
            -1, 1, compressed_tokens, compressed_tokens
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if past_key_values is not None:
            raise ValueError("H15 BERT encoder reconstruction does not support KV caching")
        if kwargs.get("encoder_hidden_states") is not None:
            raise ValueError("H15 reconstruction does not support cross-attention")

        batch_size, token_count, _ = hidden_states.shape
        head_shape = (batch_size, token_count, self.num_attention_heads, self.attention_head_size)
        query = self.query(hidden_states).view(head_shape).transpose(1, 2)
        key = self.key(hidden_states).view(head_shape).transpose(1, 2)
        value = self.value(hidden_states).view(head_shape).transpose(1, 2)
        query, context = chunked_fft_compress(
            query,
            chunk_length=self.chunk_length,
            compression_ratio=self.compression_ratio,
            dim=-2,
        )
        key, key_context = chunked_fft_compress(
            key,
            chunk_length=self.chunk_length,
            compression_ratio=self.compression_ratio,
            dim=-2,
        )
        value, value_context = chunked_fft_compress(
            value,
            chunk_length=self.chunk_length,
            compression_ratio=self.compression_ratio,
            dim=-2,
        )
        if key_context != context or value_context != context:
            raise RuntimeError("Q/K/V compression contexts diverged")

        scores = torch.matmul(query, key.transpose(-1, -2)) * self.scaling
        compressed_mask = self.compressed_attention_mask(attention_mask, context)
        if compressed_mask is not None:
            scores = scores.masked_fill(~compressed_mask, torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores, dim=-1)
        probabilities = self.dropout(probabilities)
        head_mask = kwargs.get("head_mask")
        if head_mask is not None:
            probabilities = probabilities * head_mask
        output = torch.matmul(probabilities, value)
        output = output.transpose(1, 2).reshape(batch_size, -1, self.all_head_size)
        output = chunked_fft_decompress(output, context, dim=1).contiguous()
        return output, probabilities


def inject_structured_bert_layers(
    model: nn.Module,
    *,
    modified_last_k_layers: int,
    compression_ratio: float,
    chunk_length: int,
    block_size: int,
    fit_steps: int,
    fit_learning_rate: float,
    fit_seed_base: int,
) -> list[dict[str, Any]]:
    layers = model.bert.encoder.layer
    layer_count = len(layers)
    if not 1 <= modified_last_k_layers <= layer_count:
        raise ValueError("modified_last_k_layers is outside the encoder depth")
    first_layer = layer_count - modified_last_k_layers
    fit_reports: list[dict[str, Any]] = []
    for layer_index in range(first_layer, layer_count):
        original = layers[layer_index].attention.self
        replacement = CompressedBertSelfAttention(
            original,
            compression_ratio=compression_ratio,
            chunk_length=chunk_length,
            block_size=block_size,
        ).to(device=original.query.weight.device, dtype=original.query.weight.dtype)
        for projection_index, projection_name in enumerate(("query", "key", "value")):
            dense_projection = getattr(original, projection_name)
            structured_projection = getattr(replacement, projection_name)
            seed = fit_seed_base + layer_index * 10 + projection_index
            fit = structured_projection.fit_to_dense_(
                dense_projection,
                steps=fit_steps,
                learning_rate=fit_learning_rate,
                seed=seed,
            )
            fit_reports.append(
                {
                    "layer_index": layer_index,
                    "projection": projection_name,
                    "seed": seed,
                    "structured_weight_count": structured_projection.structured_weight_count,
                    "dense_weight_count": dense_projection.weight.numel(),
                    "analytical_density": structured_projection.analytical_density,
                    **fit,
                }
            )
        layers[layer_index].attention.self = replacement
    return fit_reports


def structured_parameter_summary(model: nn.Module) -> dict[str, int | float]:
    modules = [module for module in model.modules() if isinstance(module, HierarchicalButterflyLinear)]
    structured_weights = sum(module.structured_weight_count for module in modules)
    replaced_dense_weights = sum(
        module.in_features * module.out_features for module in modules
    )
    return {
        "structured_projection_count": len(modules),
        "structured_weight_count": structured_weights,
        "replaced_dense_weight_count": replaced_dense_weights,
        "weight_density": structured_weights / replaced_dense_weights,
        "expected_b32_density": 2 * math.log2(32) / 32,
    }

