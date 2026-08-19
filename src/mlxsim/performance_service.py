"""Reusable trace-calibrated performance services for simulator compositions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import median


def _validate_vector(
    feature_names: Sequence[str], parameters: Mapping[str, float], features: Mapping[str, float]
) -> None:
    expected = set(feature_names)
    if set(parameters) != expected:
        raise ValueError("parameter keys do not match the registered feature names")
    if set(features) != expected:
        raise ValueError("feature keys do not match the registered feature names")
    if not all(math.isfinite(float(value)) for value in [*parameters.values(), *features.values()]):
        raise ValueError("performance-service values must be finite")


@dataclass(frozen=True)
class LinearFeatureService:
    """A named linear feature service with an auditable parameter vector."""

    feature_names: tuple[str, ...]
    parameters: Mapping[str, float]
    model_name: str
    target_informed: bool
    provenance: str

    def predict(self, features: Mapping[str, float]) -> float:
        _validate_vector(self.feature_names, self.parameters, features)
        value = sum(
            float(self.parameters[name]) * float(features[name]) for name in self.feature_names
        )
        if not math.isfinite(value) or value <= 0:
            raise ValueError("linear performance-service prediction must be positive and finite")
        return value

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "linear",
            "model_name": self.model_name,
            "feature_names": list(self.feature_names),
            "parameters": {name: float(self.parameters[name]) for name in self.feature_names},
            "target_informed": self.target_informed,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class LogLinearFeatureService:
    """A named log-linear service returning exp(parameter dot feature)."""

    feature_names: tuple[str, ...]
    parameters: Mapping[str, float]
    model_name: str
    target_informed: bool
    provenance: str

    def predict(self, features: Mapping[str, float]) -> float:
        _validate_vector(self.feature_names, self.parameters, features)
        log_value = sum(
            float(self.parameters[name]) * float(features[name]) for name in self.feature_names
        )
        value = math.exp(log_value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("log-linear performance-service prediction must be positive and finite")
        return value

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "log_linear",
            "model_name": self.model_name,
            "feature_names": list(self.feature_names),
            "parameters": {name: float(self.parameters[name]) for name in self.feature_names},
            "target_informed": self.target_informed,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class CyclePhase:
    """One positive integer interval in a physicalized cycle timeline."""

    name: str
    cycles: int
    kind: str
    provenance: str

    def __post_init__(self) -> None:
        if not self.name or not self.kind or not self.provenance:
            raise ValueError("cycle phases require name, kind and provenance")
        if self.cycles <= 0:
            raise ValueError("cycle phases must contain positive integer cycles")


@dataclass(frozen=True)
class CycleTimeline:
    """An explicit phase sequence whose sum is the only reported latency."""

    name: str
    clock_hz: int
    phases: tuple[CyclePhase, ...]
    target_informed: bool

    def __post_init__(self) -> None:
        if not self.name or self.clock_hz <= 0 or not self.phases:
            raise ValueError("cycle timelines require a name, clock and phases")

    @property
    def total_cycles(self) -> int:
        return sum(phase.cycles for phase in self.phases)

    @property
    def latency_ms(self) -> float:
        return self.total_cycles / self.clock_hz * 1000.0

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "clock_hz": self.clock_hz,
            "target_informed": self.target_informed,
            "phases": [
                {
                    "name": phase.name,
                    "cycles": phase.cycles,
                    "kind": phase.kind,
                    "provenance": phase.provenance,
                }
                for phase in self.phases
            ],
            "total_cycles": self.total_cycles,
            "latency_ms": self.latency_ms,
        }


def median_normalized(values: Mapping[int, float]) -> dict[int, float]:
    """Normalize positive finite trace values by their cross-shape median."""
    if not values:
        raise ValueError("at least one trace value is required")
    converted = {int(key): float(value) for key, value in values.items()}
    if not all(math.isfinite(value) and value > 0 for value in converted.values()):
        raise ValueError("trace values must be positive and finite")
    center = float(median(converted.values()))
    return {key: value / center for key, value in converted.items()}


__all__ = [
    "CyclePhase",
    "CycleTimeline",
    "LinearFeatureService",
    "LogLinearFeatureService",
    "median_normalized",
]
