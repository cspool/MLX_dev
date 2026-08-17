"""Affine steady-state folding for repeated deterministic CDC schedules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AffineCycleModel:
    intercept: float
    slope: float

    def predict(self, repeats: float) -> float:
        if repeats < 0:
            raise ValueError("repeats must be nonnegative")
        return self.intercept + self.slope * repeats


def fit_affine(
    first_repeats: float,
    first_cycles: float,
    second_repeats: float,
    second_cycles: float,
) -> AffineCycleModel:
    if first_repeats < 0 or second_repeats <= first_repeats:
        raise ValueError("fit repeats must be ordered and nonnegative")
    slope = (second_cycles - first_cycles) / (second_repeats - first_repeats)
    if slope <= 0:
        raise ValueError("cycle slope must be positive")
    return AffineCycleModel(first_cycles - slope * first_repeats, slope)


def relative_error(actual: float, expected: float) -> float:
    return abs(actual - expected) / abs(expected)


__all__ = ["AffineCycleModel", "fit_affine", "relative_error"]
