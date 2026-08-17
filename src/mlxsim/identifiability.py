"""Source-bound ambiguity witnesses for compressed-LLM reconstruction."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from .llama_perplexity import sha256_file
from .structured import chunked_fft_compress, chunked_fft_decompress


def qualify_file(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    actual_bytes = path.stat().st_size if path.is_file() else None
    actual_sha256 = sha256_file(path) if path.is_file() else None
    checks = {
        "is_file": path.is_file(),
        "bytes": actual_bytes == int(expected["bytes"]),
        "sha256": actual_sha256 == expected["sha256"],
    }
    return {
        "path": str(path),
        "actual_bytes": actual_bytes,
        "expected_bytes": int(expected["bytes"]),
        "actual_sha256": actual_sha256,
        "expected_sha256": expected["sha256"],
        "checks": checks,
        "pass": all(checks.values()),
    }


def line_segment_bytes(path: Path, start_line: int, end_line: int) -> bytes:
    if start_line <= 0 or end_line < start_line:
        raise ValueError("invalid one-indexed inclusive line range")
    lines = path.read_text(encoding="utf-8").splitlines()
    if end_line > len(lines):
        raise ValueError("line range exceeds source")
    return ("\n".join(lines[start_line - 1 : end_line]) + "\n").encode()


def qualify_line_segment(
    path: Path, name: str, expected: Mapping[str, Any]
) -> dict[str, Any]:
    start_line = int(expected["start_line"])
    end_line = int(expected["end_line"])
    content = line_segment_bytes(path, start_line, end_line)
    actual_sha256 = hashlib.sha256(content).hexdigest()
    checks = {
        "bytes": len(content) == int(expected["bytes"]),
        "sha256": actual_sha256 == expected["sha256"],
    }
    return {
        "name": name,
        "start_line": start_line,
        "end_line": end_line,
        "actual_bytes": len(content),
        "expected_bytes": int(expected["bytes"]),
        "actual_sha256": actual_sha256,
        "expected_sha256": expected["sha256"],
        "checks": checks,
        "pass": all(checks.values()),
    }


def deterministic_signal(length: int) -> torch.Tensor:
    if length <= 0:
        raise ValueError("signal length must be positive")
    index = torch.arange(length, dtype=torch.float64)
    return (
        torch.sin(2.0 * torch.pi * index / 7.0)
        + 0.1 * index
        + 0.25 * torch.cos(2.0 * torch.pi * index / 5.0)
    )


def fft_ambiguity_witness(
    *,
    chunk_length: int,
    compression_ratio: float,
    perturbation_index: int,
    perturbation_delta: float,
) -> dict[str, Any]:
    if not 0 <= perturbation_index < chunk_length:
        raise ValueError("perturbation index is outside the chunk")
    compressed_length_float = chunk_length * compression_ratio
    compressed_length = round(compressed_length_float)
    if not math.isclose(compressed_length_float, compressed_length):
        raise ValueError("compression ratio does not produce an integral length")

    signal = deterministic_signal(chunk_length)
    real_compressed, context = chunked_fft_compress(
        signal,
        chunk_length=chunk_length,
        compression_ratio=compression_ratio,
        dim=-1,
    )
    real_restored = chunked_fft_decompress(real_compressed, context, dim=-1)
    perturbed = signal.clone()
    perturbed[perturbation_index] += perturbation_delta
    perturbed_compressed, perturbed_context = chunked_fft_compress(
        perturbed,
        chunk_length=chunk_length,
        compression_ratio=compression_ratio,
        dim=-1,
    )
    perturbed_restored = chunked_fft_decompress(
        perturbed_compressed, perturbed_context, dim=-1
    )
    earlier_delta = (
        perturbed_restored[:perturbation_index] - real_restored[:perturbation_index]
    ).abs()

    full_spectrum = torch.fft.fft(signal)
    literal_prefix = torch.fft.ifft(full_spectrum[:compressed_length], n=compressed_length)
    interpretation_delta = literal_prefix - real_compressed.to(torch.complex128)
    return {
        "chunk_length": chunk_length,
        "compression_ratio": compression_ratio,
        "compressed_length": compressed_length,
        "perturbation_index": perturbation_index,
        "perturbation_delta": perturbation_delta,
        "earlier_positions": perturbation_index,
        "changed_earlier_positions": int(torch.count_nonzero(earlier_delta)),
        "maximum_earlier_absolute_change": float(earlier_delta.max()),
        "l2_earlier_change": float(torch.linalg.vector_norm(earlier_delta)),
        "literal_prefix_maximum_imaginary": float(literal_prefix.imag.abs().max()),
        "literal_prefix_l2_imaginary": float(torch.linalg.vector_norm(literal_prefix.imag)),
        "interpretation_maximum_absolute_difference": float(
            interpretation_delta.abs().max()
        ),
        "interpretation_l2_difference": float(
            torch.linalg.vector_norm(interpretation_delta)
        ),
        "real_compressed_first_four": real_compressed[:4].tolist(),
        "literal_prefix_first_four": [
            {"real": float(value.real), "imag": float(value.imag)}
            for value in literal_prefix[:4]
        ],
    }


def layer_plan_combinatorics(
    *,
    total_layers: int,
    minimum_modified_layers: int,
    chunk_lengths: Sequence[int],
) -> dict[str, Any]:
    if not 0 <= minimum_modified_layers <= total_layers:
        raise ValueError("invalid modified-layer range")
    if not chunk_lengths or len(set(chunk_lengths)) != len(chunk_lengths):
        raise ValueError("chunk lengths must be nonempty and unique")
    layer_subsets = sum(
        math.comb(total_layers, count)
        for count in range(minimum_modified_layers, total_layers + 1)
    )
    minimum_chunk_assignments = len(chunk_lengths) ** minimum_modified_layers
    return {
        "total_layers": total_layers,
        "minimum_modified_layers": minimum_modified_layers,
        "admissible_layer_subsets": layer_subsets,
        "chunk_length_option_count": len(chunk_lengths),
        "chunk_lengths": list(chunk_lengths),
        "minimum_chunk_assignments_at_minimum_layers": minimum_chunk_assignments,
    }


def missing_field_audit(
    fields: Sequence[Mapping[str, Any]], required_domains: Sequence[str]
) -> dict[str, Any]:
    domain_totals = Counter(str(field["domain"]) for field in fields)
    missing = [field for field in fields if not bool(field["disclosed"])]
    domain_missing = Counter(str(field["domain"]) for field in missing)
    required = list(required_domains)
    checks = {
        "all_required_domains_represented": all(domain_totals[domain] for domain in required),
        "all_required_domains_have_omission": all(
            domain_missing[domain] for domain in required
        ),
        "all_registered_fields_undisclosed": len(missing) == len(fields),
    }
    return {
        "registered_fields": [dict(field) for field in fields],
        "field_count": len(fields),
        "missing_field_count": len(missing),
        "missing_fraction": len(missing) / len(fields),
        "domain_field_counts": dict(sorted(domain_totals.items())),
        "domain_missing_counts": dict(sorted(domain_missing.items())),
        "required_domains": required,
        "checks": checks,
        "pass": all(checks.values()),
    }
