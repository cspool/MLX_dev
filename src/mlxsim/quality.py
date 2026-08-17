from __future__ import annotations

import math
import re
import string
from collections import Counter
from collections.abc import Mapping, Sequence


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


def normalize_squad_answer(text: str) -> str:
    """Apply the normalization used by the official SQuAD 1.1 scorer."""

    without_punctuation = "".join(character for character in text.lower() if character not in string.punctuation)
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def _squad_exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_squad_answer(prediction) == normalize_squad_answer(ground_truth))


def _squad_f1(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_squad_answer(prediction).split()
    ground_truth_tokens = normalize_squad_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    overlap = sum(common.values())
    if not prediction_tokens or not ground_truth_tokens:
        return float(prediction_tokens == ground_truth_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(ground_truth_tokens)
    return 2.0 * precision * recall / (precision + recall)


def squad_metrics(
    predictions: Mapping[str, str],
    references: Mapping[str, Sequence[str]],
) -> dict[str, float]:
    """Return official-style SQuAD exact-match and token-F1 percentages."""

    if predictions.keys() != references.keys():
        missing = sorted(set(references) - set(predictions))
        extra = sorted(set(predictions) - set(references))
        raise ValueError(f"prediction/reference IDs differ; missing={missing[:3]}, extra={extra[:3]}")
    if not references:
        raise ValueError("at least one SQuAD reference is required")
    exact = 0.0
    f1 = 0.0
    for example_id, answers in references.items():
        if not answers:
            raise ValueError(f"example {example_id!r} has no reference answer")
        prediction = predictions[example_id]
        exact += max(_squad_exact_match(prediction, answer) for answer in answers)
        f1 += max(_squad_f1(prediction, answer) for answer in answers)
    scale = 100.0 / len(references)
    return {"exact_match": exact * scale, "f1": f1 * scale}
