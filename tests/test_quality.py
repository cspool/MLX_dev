import math

import pytest

from mlxsim.quality import contiguous_window_ranges, perplexity_from_nll


def test_contiguous_windows_cover_all_usable_tokens() -> None:
    assert contiguous_window_ranges(10, 4) == [(0, 4), (4, 8), (8, 10)]


def test_single_token_tail_is_not_a_prediction_window() -> None:
    assert contiguous_window_ranges(9, 4) == [(0, 4), (4, 8)]


def test_weighted_perplexity() -> None:
    assert perplexity_from_nll(math.log(4.0) * 10, 10) == pytest.approx(4.0)
