"""Inferred post-RoPE chunk-FFT attention wiring for Llama models."""

from __future__ import annotations

from typing import Any

import torch
from peft import LoraConfig, TaskType
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from .structured import chunked_fft_compress, chunked_fft_decompress


class CompressedLlamaAttention(nn.Module):
    """Llama attention with inferred post-RoPE chunk compression/decompression."""

    def __init__(
        self,
        source: nn.Module,
        *,
        chunk_length: int,
        compression_ratio: float,
    ) -> None:
        super().__init__()
        self.config = source.config
        self.layer_idx = source.layer_idx
        self.head_dim = source.head_dim
        self.num_key_value_groups = source.num_key_value_groups
        self.scaling = source.scaling
        self.attention_dropout = source.attention_dropout
        self.q_proj = source.q_proj
        self.k_proj = source.k_proj
        self.v_proj = source.v_proj
        self.o_proj = source.o_proj
        self.chunk_length = chunk_length
        self.compression_ratio = compression_ratio

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        del attention_mask, kwargs
        if past_key_values is not None:
            raise RuntimeError("CompressedLlamaAttention does not support KV cache")
        if position_embeddings is None:
            raise RuntimeError("position_embeddings are required")
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        query, context = chunked_fft_compress(
            query,
            chunk_length=self.chunk_length,
            compression_ratio=self.compression_ratio,
            dim=2,
        )
        key, key_context = chunked_fft_compress(
            key,
            chunk_length=self.chunk_length,
            compression_ratio=self.compression_ratio,
            dim=2,
        )
        value, value_context = chunked_fft_compress(
            value,
            chunk_length=self.chunk_length,
            compression_ratio=self.compression_ratio,
            dim=2,
        )
        if context != key_context or context != value_context:
            raise RuntimeError("Q/K/V compression contexts diverged")

        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)
        dropout = self.attention_dropout if self.training else 0.0
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=dropout,
            is_causal=True,
            scale=self.scaling,
        )
        output = output.transpose(1, 2).contiguous().reshape(
            hidden_states.shape[0], -1, hidden_states.shape[-1]
        )
        output = chunked_fft_decompress(output, context, dim=1).contiguous()
        if output.shape[:-1] != input_shape:
            raise RuntimeError(
                f"decompressed attention shape {output.shape} does not match {input_shape}"
            )
        return self.o_proj(output), None


def _decoder_layers(model: nn.Module) -> nn.ModuleList:
    candidate = model
    while hasattr(candidate, "model"):
        candidate = candidate.model
    if not hasattr(candidate, "layers"):
        raise TypeError("could not locate Llama decoder layers")
    return candidate.layers


def install_compressed_attention(
    model: nn.Module,
    *,
    layer_indices: list[int],
    chunk_length: int,
    compression_ratio: float,
) -> list[int]:
    layers = _decoder_layers(model)
    installed: list[int] = []
    for index in layer_indices:
        if not 0 <= index < len(layers):
            raise ValueError(f"layer index out of range: {index}")
        source = layers[index].self_attn
        if isinstance(source, CompressedLlamaAttention):
            raise TypeError(f"layer {index} is already compressed")
        layers[index].self_attn = CompressedLlamaAttention(
            source,
            chunk_length=chunk_length,
            compression_ratio=compression_ratio,
        )
        installed.append(index)
    return installed


def make_lora_config(config: dict[str, Any]) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(config["rank"]),
        lora_alpha=int(config["alpha"]),
        lora_dropout=float(config["dropout"]),
        target_modules=list(config["target_modules"]),
        bias=config["bias"],
        layers_to_transform=list(config["layers_to_transform"]),
        layers_pattern=config["layers_pattern"],
        init_lora_weights=True,
    )


def audit_trainable_parameters(
    model: nn.Module, *, expected_layers: list[int], maximum_fraction: float
) -> dict[str, Any]:
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    total_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(parameter.numel() for _, parameter in trainable)
    unexpected = [name for name, _ in trainable if "lora_" not in name]
    missing_layer_names = [
        index
        for index in expected_layers
        if not any(f".layers.{index}." in name for name, _ in trainable)
    ]
    outside_layer_names = [
        name
        for name, _ in trainable
        if not any(f".layers.{index}." in name for index in expected_layers)
    ]
    fraction = trainable_count / total_count
    return {
        "trainable_parameters": trainable_count,
        "total_parameters": total_count,
        "trainable_fraction": fraction,
        "trainable_tensor_count": len(trainable),
        "unexpected_non_lora_parameters": unexpected,
        "missing_expected_layers": missing_layer_names,
        "outside_expected_layers": outside_layer_names,
        "pass": not unexpected
        and not missing_layer_names
        and not outside_layer_names
        and fraction <= maximum_fraction,
    }
