from __future__ import annotations

import math


def contiguous_window_ranges(num_tokens: int, sequence_length: int) -> list[tuple[int, int]]:
    """Partition a token stream into contiguous causal-LM evaluation windows."""

    if num_tokens < 2:
        raise ValueError("at least two tokens are required")
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least two")
    ranges = [
        (start, min(start + sequence_length, num_tokens))
        for start in range(0, num_tokens - 1, sequence_length)
    ]
    return [(start, end) for start, end in ranges if end - start >= 2]


def perplexity_from_nll(total_negative_log_likelihood: float, predicted_tokens: int) -> float:
    if predicted_tokens <= 0:
        raise ValueError("predicted_tokens must be positive")
    return math.exp(total_negative_log_likelihood / predicted_tokens)
