import math

import pytest

from mlxsim.quality import (
    contiguous_window_ranges,
    normalize_squad_answer,
    perplexity_from_nll,
    squad_metrics,
)


def test_contiguous_windows_cover_all_usable_tokens() -> None:
    assert contiguous_window_ranges(10, 4) == [(0, 4), (4, 8), (8, 10)]


def test_single_token_tail_is_not_a_prediction_window() -> None:
    assert contiguous_window_ranges(9, 4) == [(0, 4), (4, 8)]


def test_weighted_perplexity() -> None:
    assert perplexity_from_nll(math.log(4.0) * 10, 10) == pytest.approx(4.0)


def test_squad_normalization_matches_official_rules() -> None:
    assert normalize_squad_answer("The, Quick Fox!") == "quick fox"


def test_squad_metrics_use_best_reference_and_token_overlap() -> None:
    metrics = squad_metrics(
        {"one": "Denver Broncos", "two": "blue whale"},
        {"one": ["the Denver Broncos", "Broncos"], "two": ["whale shark"]},
    )
    assert metrics["exact_match"] == pytest.approx(50.0)
    assert metrics["f1"] == pytest.approx(75.0)


def test_squad_metrics_reject_mismatched_ids() -> None:
    with pytest.raises(ValueError, match="IDs differ"):
        squad_metrics({"prediction": "x"}, {"reference": ["x"]})
