"""Same-input golden and mapping-aware numerical execution for lowered MLX graphs."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MappingConfig:
    name: str
    simd_width: int
    mesh: tuple[int, int]

    @property
    def pe_count(self) -> int:
        return self.mesh[0] * self.mesh[1]


def _quantize(value: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "float16":
        return value.astype(np.float16).astype(np.float32)
    if dtype == "float32":
        return value.astype(np.float32)
    raise ValueError(f"unsupported numerical-equivalence dtype: {dtype}")


def _block_weight(
    rng: np.random.Generator, input_dim: int, output_dim: int, block_size: int, dtype: str
) -> np.ndarray:
    if input_dim % block_size or output_dim % block_size:
        raise ValueError("block weight dimensions must divide the block size")
    result = np.zeros((input_dim, output_dim), dtype=np.float32)
    input_blocks = input_dim // block_size
    output_blocks = output_dim // block_size
    for output_block in range(output_blocks):
        input_block = output_block % input_blocks
        row = slice(input_block * block_size, (input_block + 1) * block_size)
        column = slice(output_block * block_size, (output_block + 1) * block_size)
        result[row, column] = rng.normal(0.0, 0.08, (block_size, block_size))
    return _quantize(result, dtype)


def _rmsnorm_golden(value: np.ndarray) -> np.ndarray:
    variance = np.mean(value * value, axis=-1, keepdims=True)
    return value / np.sqrt(variance + 1.0e-5)


def _rmsnorm_lowered(value: np.ndarray, mapping: MappingConfig) -> np.ndarray:
    result = np.empty_like(value, dtype=np.float32)
    for shard in np.array_split(np.arange(value.shape[0]), mapping.pe_count):
        for row in shard:
            total = 0.0
            for start in range(0, value.shape[1], mapping.simd_width):
                tile = value[row, start : start + mapping.simd_width]
                total += float(np.sum(tile.astype(np.float64) ** 2))
            scale = 1.0 / math.sqrt(total / value.shape[1] + 1.0e-5)
            result[row] = value[row] * scale
    return result


def _rope_angles(sequence: int, dimension: int) -> tuple[np.ndarray, np.ndarray]:
    positions = np.arange(sequence, dtype=np.float32)[:, None]
    frequencies = 1.0 / (
        10000.0 ** (np.arange(0, dimension, 2, dtype=np.float32) / dimension)
    )
    angle = positions * frequencies[None, :]
    return np.cos(angle), np.sin(angle)


def _rope_golden(value: np.ndarray) -> np.ndarray:
    cosine, sine = _rope_angles(*value.shape)
    even = value[:, 0::2]
    odd = value[:, 1::2]
    result = np.empty_like(value)
    result[:, 0::2] = even * cosine - odd * sine
    result[:, 1::2] = even * sine + odd * cosine
    return result


def _rope_lowered(value: np.ndarray, mapping: MappingConfig) -> np.ndarray:
    cosine, sine = _rope_angles(*value.shape)
    result = np.empty_like(value)
    pair_tile = max(1, mapping.simd_width // 2)
    for shard in np.array_split(np.arange(value.shape[0]), mapping.pe_count):
        for row in shard:
            for pair_start in range(0, value.shape[1] // 2, pair_tile):
                for pair in range(pair_start, min(value.shape[1] // 2, pair_start + pair_tile)):
                    even = float(value[row, 2 * pair])
                    odd = float(value[row, 2 * pair + 1])
                    result[row, 2 * pair] = even * cosine[row, pair] - odd * sine[row, pair]
                    result[row, 2 * pair + 1] = even * sine[row, pair] + odd * cosine[row, pair]
    return result


def _block_linear_lowered(
    value: np.ndarray, weight: np.ndarray, block_size: int, mapping: MappingConfig
) -> np.ndarray:
    result = np.zeros((value.shape[0], weight.shape[1]), dtype=np.float32)
    input_blocks = weight.shape[0] // block_size
    output_blocks = weight.shape[1] // block_size
    for shard in np.array_split(np.arange(value.shape[0]), mapping.pe_count):
        for row in shard:
            for output_block in range(output_blocks):
                input_block = output_block % input_blocks
                input_slice = slice(input_block * block_size, (input_block + 1) * block_size)
                output_slice = slice(
                    output_block * block_size, (output_block + 1) * block_size
                )
                for lane_start in range(0, block_size, mapping.simd_width):
                    lane_stop = min(block_size, lane_start + mapping.simd_width)
                    result[row, output_slice] += np.matmul(
                        value[row, input_slice][lane_start:lane_stop],
                        weight[input_slice, output_slice][lane_start:lane_stop],
                    )
    return result


def _fft_cmp_golden(value: np.ndarray) -> np.ndarray:
    spectrum = np.fft.fft(value, axis=0)
    retained = max(1, value.shape[0] // 2)
    spectrum[retained:] = 0
    return np.fft.ifft(spectrum, axis=0).real.astype(np.float32)


def _fft_cmp_lowered(value: np.ndarray, mapping: MappingConfig) -> np.ndarray:
    sequence = value.shape[0]
    indices = np.arange(sequence, dtype=np.float64)
    forward = np.exp(-2j * np.pi * np.outer(indices, indices) / sequence)
    inverse = np.conjugate(forward).T / sequence
    retained = max(1, sequence // 2)
    result = np.empty_like(value, dtype=np.float32)
    for start in range(0, value.shape[1], mapping.simd_width):
        stop = min(value.shape[1], start + mapping.simd_width)
        spectrum = forward @ value[:, start:stop].astype(np.float64)
        spectrum[retained:] = 0
        result[:, start:stop] = (inverse @ spectrum).real.astype(np.float32)
    return result


def _fft2d_golden(value: np.ndarray) -> np.ndarray:
    spectrum = np.fft.fft2(value)
    spectrum[value.shape[0] // 2 :] = 0
    spectrum[:, value.shape[1] // 2 :] = 0
    return np.fft.ifft2(spectrum).real.astype(np.float32)


def _fft2d_lowered(value: np.ndarray, mapping: MappingConfig) -> np.ndarray:
    rows, columns = value.shape
    row_index = np.arange(rows, dtype=np.float64)
    column_index = np.arange(columns, dtype=np.float64)
    row_forward = np.exp(-2j * np.pi * np.outer(row_index, row_index) / rows)
    col_forward = np.exp(-2j * np.pi * np.outer(column_index, column_index) / columns)
    row_inverse = np.conjugate(row_forward).T / rows
    col_inverse = np.conjugate(col_forward).T / columns
    spectrum = np.zeros((rows, columns), dtype=np.complex128)
    for start in range(0, columns, mapping.simd_width):
        stop = min(columns, start + mapping.simd_width)
        spectrum[:, start:stop] = row_forward @ value[:, start:stop]
    spectrum = spectrum @ col_forward.T
    spectrum[rows // 2 :] = 0
    spectrum[:, columns // 2 :] = 0
    return (row_inverse @ spectrum @ col_inverse.T).real.astype(np.float32)


def _softmax_golden(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / np.sum(exponential, axis=-1, keepdims=True)


def _attention_golden(value: np.ndarray) -> np.ndarray:
    scores = value @ value.T / math.sqrt(value.shape[1])
    return _softmax_golden(scores) @ value


def _attention_lowered(value: np.ndarray, mapping: MappingConfig) -> np.ndarray:
    result = np.zeros_like(value, dtype=np.float32)
    scale = math.sqrt(value.shape[1])
    for shard in np.array_split(np.arange(value.shape[0]), mapping.pe_count):
        for row in shard:
            scores = np.empty(value.shape[0], dtype=np.float64)
            for key_row in range(value.shape[0]):
                score = 0.0
                for start in range(0, value.shape[1], mapping.simd_width):
                    stop = min(value.shape[1], start + mapping.simd_width)
                    score += float(
                        np.dot(value[row, start:stop], value[key_row, start:stop])
                    )
                scores[key_row] = score / scale
            scores -= np.max(scores)
            weights = np.exp(scores)
            weights /= np.sum(weights)
            for start in range(0, value.shape[1], mapping.simd_width):
                stop = min(value.shape[1], start + mapping.simd_width)
                result[row, start:stop] = weights @ value[:, start:stop]
    return result


def _silu(value: np.ndarray) -> np.ndarray:
    return value / (1.0 + np.exp(-value))


def _array_digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    denominator = np.maximum(np.abs(expected), 1.0e-8)
    return float(np.max(np.abs(actual - expected) / denominator))


def prepare_tensors(
    graph_id: str, contract: Mapping[str, Any], seed: int, dtype: str
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    sequence = int(contract["sequence_length"])
    hidden = int(contract["hidden_dimension"])
    ffn = int(contract["ffn_dimension"])
    block = int(contract["block_size"])
    tensors: dict[str, Any] = {
        "input": _quantize(rng.normal(0.0, 0.2, (sequence, hidden)), dtype),
        "weights": {},
    }
    shapes: dict[str, tuple[int, int]]
    if graph_id == "figure23_complete_block":
        shapes = {
            "qkv": (hidden, hidden),
            "output": (hidden, hidden),
            "ffn_up": (hidden, ffn),
            "ffn_down": (ffn, hidden),
        }
    elif graph_id in {"figure19_fabnet_block", "figure20_llama_kernels"}:
        shapes = {
            "qkv": (hidden, hidden),
            "global_ffn1": (hidden, ffn),
            "global_ffn2": (ffn, hidden),
            "ffn1": (hidden, ffn),
            "ffn2": (ffn, hidden),
        }
    else:
        raise ValueError(f"unsupported graph: {graph_id}")
    for name, (input_dim, output_dim) in shapes.items():
        tensors["weights"][name] = _block_weight(
            rng, input_dim, output_dim, block, dtype
        )
    return tensors


def _execute_node(
    *,
    graph_id: str,
    node_id: str,
    value: np.ndarray,
    tensors: Mapping[str, Any],
    block_size: int,
    dtype: str,
    mapping: MappingConfig | None,
) -> tuple[np.ndarray, int]:
    lowered = mapping is not None
    if node_id == "rmsnorm":
        output = _rmsnorm_lowered(value, mapping) if lowered else _rmsnorm_golden(value)
        operations = value.size * 4
    elif node_id == "rope":
        output = _rope_lowered(value, mapping) if lowered else _rope_golden(value)
        operations = value.size * 3
    elif node_id in {"qkv", "output", "global_ffn1", "global_ffn2", "ffn1", "ffn2"}:
        weight_name = node_id
        weight = tensors["weights"][weight_name]
        output = (
            _block_linear_lowered(value, weight, block_size, mapping)
            if lowered
            else value @ weight
        )
        operations = 2 * value.shape[0] * int(np.count_nonzero(weight))
    elif node_id in {"fft_cmp", "attention"} and graph_id != "figure19_fabnet_block":
        if node_id == "fft_cmp":
            output = _fft_cmp_lowered(value, mapping) if lowered else _fft_cmp_golden(value)
            operations = value.size * int(math.log2(value.shape[0])) * 10
        else:
            output = (
                _attention_lowered(value, mapping) if lowered else _attention_golden(value)
            )
            operations = 4 * value.shape[0] * value.shape[0] * value.shape[1]
    elif node_id == "fft2d_attention":
        output = _fft2d_lowered(value, mapping) if lowered else _fft2d_golden(value)
        operations = value.size * (
            int(math.log2(value.shape[0])) + int(math.log2(value.shape[1]))
        ) * 10
    elif node_id == "ffn":
        up_weight = tensors["weights"]["ffn_up"]
        down_weight = tensors["weights"]["ffn_down"]
        if lowered:
            hidden = _block_linear_lowered(value, up_weight, block_size, mapping)
            hidden = _silu(hidden)
            output = _block_linear_lowered(hidden, down_weight, block_size, mapping)
        else:
            hidden = _silu(value @ up_weight)
            output = hidden @ down_weight
        operations = 2 * value.shape[0] * (
            int(np.count_nonzero(up_weight)) + int(np.count_nonzero(down_weight))
        )
    else:
        raise ValueError(f"unsupported node {graph_id}.{node_id}")
    return _quantize(np.asarray(output), dtype), operations


def execute_graph(
    *,
    graph_id: str,
    graph: Mapping[str, Any],
    order: list[str],
    contract: Mapping[str, Any],
    seed: int,
    dtype: str,
    mapping: MappingConfig | None,
) -> dict[str, Any]:
    tensors = prepare_tensors(graph_id, contract, seed, dtype)
    original = tensors["input"]
    by_id = {str(node["id"]): node for node in graph["operators"]}
    boundaries: dict[str, np.ndarray] = {}
    events: list[str] = []
    operation_counts: dict[str, int] = {}
    for node_id in order:
        dependencies = [str(item) for item in by_id[node_id].get("depends_on", [])]
        value = boundaries[dependencies[-1]] if dependencies else original
        output, operations = _execute_node(
            graph_id=graph_id,
            node_id=node_id,
            value=value,
            tensors=tensors,
            block_size=int(contract["block_size"]),
            dtype=dtype,
            mapping=mapping,
        )
        boundaries[node_id] = output
        events.append(node_id)
        operation_counts[node_id] = operations
    depended = {
        str(parent)
        for node in graph["operators"]
        for parent in node.get("depends_on", [])
    }
    sinks = [node_id for node_id in order if node_id not in depended]
    final = np.concatenate([boundaries[node_id].reshape(-1) for node_id in sinks])
    return {
        "boundaries": boundaries,
        "events": events,
        "operation_counts": operation_counts,
        "tensor_elements": {
            node_id: int(value.size) for node_id, value in boundaries.items()
        },
        "sinks": sinks,
        "final": final,
        "final_sha256": _array_digest(final),
    }


def compare_execution(
    actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, Any]:
    boundaries = {
        node_id: {
            "maximum_absolute_error": float(
                np.max(np.abs(actual["boundaries"][node_id] - expected_value))
            ),
            "maximum_relative_error": _relative_error(
                actual["boundaries"][node_id], expected_value
            ),
            "actual_sha256": _array_digest(actual["boundaries"][node_id]),
            "expected_sha256": _array_digest(expected_value),
        }
        for node_id, expected_value in expected["boundaries"].items()
    }
    return {
        "boundaries": boundaries,
        "final_maximum_absolute_error": float(
            np.max(np.abs(actual["final"] - expected["final"]))
        ),
        "final_maximum_relative_error": _relative_error(actual["final"], expected["final"]),
        "event_order_identity": actual["events"] == expected["events"],
        "operation_count_identity": actual["operation_counts"]
        == expected["operation_counts"],
        "tensor_element_identity": actual["tensor_elements"]
        == expected["tensor_elements"],
    }


__all__ = [
    "MappingConfig",
    "compare_execution",
    "execute_graph",
    "prepare_tensors",
]
