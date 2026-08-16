from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RooflineSystem:
    peak_gops: float
    bandwidth_gbs: float

    def __post_init__(self) -> None:
        if min(self.peak_gops, self.bandwidth_gbs) <= 0:
            raise ValueError("roofline peak and bandwidth must be positive")


@dataclass(frozen=True)
class RooflineCalibration:
    name: str
    reference_n: int
    reference_d: int
    systems: dict[str, RooflineSystem]
    efficiency_coefficients: dict[str, dict[str, tuple[float, float, float, float]]]
    classification: str = "calibration-replay-only"
    fit_provenance: str = ""
    baseline_throughput_feature_order: tuple[str, ...] = ()
    baseline_throughput_coefficients: dict[str, dict[str, tuple[float, ...]]] | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> RooflineCalibration:
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, dict):
            raise TypeError(f"roofline calibration must be a mapping: {path}")
        systems = {name: RooflineSystem(**values) for name, values in raw.pop("systems").items()}
        coefficients = {
            system: {operator: tuple(values) for operator, values in operators.items()}
            for system, operators in raw.pop("efficiency_coefficients").items()
        }
        feature_order = tuple(raw.pop("baseline_throughput_feature_order", ()))
        throughput_coefficients = {
            system: {operator: tuple(values) for operator, values in operators.items()}
            for system, operators in raw.pop("baseline_throughput_coefficients", {}).items()
        }
        return cls(
            systems=systems,
            efficiency_coefficients=coefficients,
            baseline_throughput_feature_order=feature_order,
            baseline_throughput_coefficients=throughput_coefficients,
            **raw,
        )

    def __post_init__(self) -> None:
        if min(self.reference_n, self.reference_d) <= 0:
            raise ValueError("roofline reference shapes must be positive")
        for system, operators in self.efficiency_coefficients.items():
            if system not in self.systems:
                raise ValueError(f"efficiency coefficients reference unknown system {system!r}")
            for operator, coefficients in operators.items():
                if len(coefficients) != 4:
                    raise ValueError(f"{system}/{operator} needs four efficiency coefficients")
        if self.baseline_throughput_coefficients:
            expected = len(self.baseline_throughput_feature_order)
            for system, operators in self.baseline_throughput_coefficients.items():
                if system not in self.systems:
                    raise ValueError(f"throughput coefficients reference unknown system {system!r}")
                for operator, coefficients in operators.items():
                    if len(coefficients) != expected:
                        raise ValueError(
                            f"{system}/{operator} needs {expected} throughput coefficients"
                        )

    def utilization(self, system: str, operator: str, n: int, d: int) -> float:
        coefficients = self.efficiency_coefficients[system][operator]
        log_n = math.log2(n / self.reference_n)
        log_d = math.log2(d / self.reference_d)
        features = (1.0, log_n, log_d, log_n * log_d)
        value = sum(coefficient * feature for coefficient, feature in zip(coefficients, features))
        return min(1.0, max(0.01, value))

    def roofline_gops(self, system: str, operational_intensity: float) -> float:
        if operational_intensity <= 0:
            raise ValueError("operational intensity must be positive")
        hardware = self.systems[system]
        return min(hardware.peak_gops, operational_intensity * hardware.bandwidth_gbs)

    def achieved_gops(
        self, system: str, operator: str, n: int, d: int, operational_intensity: float
    ) -> float:
        return self.utilization(system, operator, n, d) * self.roofline_gops(
            system, operational_intensity
        )

    def baseline_gops(self, system: str, operator: str, family: str, n: int) -> float:
        if not self.baseline_throughput_coefficients:
            raise ValueError("no empirical baseline throughput surface is configured")
        coefficients = self.baseline_throughput_coefficients[system][operator]
        log_n = math.log2(n / self.reference_n)
        normalized_family = family.lower()
        is_llama = float(normalized_family.startswith("llama"))
        is_intern = float(normalized_family.startswith("intern"))
        features = (
            1.0,
            log_n,
            log_n * log_n,
            is_llama,
            is_intern,
            log_n * is_intern,
            log_n * is_llama,
        )
        supported_order = (
            "bias",
            "log_n",
            "log_n_sq",
            "llama",
            "internlm",
            "log_n_internlm",
            "log_n_llama",
        )
        if tuple(self.baseline_throughput_feature_order) != supported_order:
            raise ValueError("unsupported baseline throughput feature order")
        log_gops = sum(
            coefficient * feature for coefficient, feature in zip(coefficients, features)
        )
        return min(self.systems[system].peak_gops, math.exp(log_gops))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "classification": self.classification,
            "fit_provenance": self.fit_provenance,
            "reference_n": self.reference_n,
            "reference_d": self.reference_d,
            "systems": {
                name: {"peak_gops": value.peak_gops, "bandwidth_gbs": value.bandwidth_gbs}
                for name, value in self.systems.items()
            },
            "efficiency_coefficients": self.efficiency_coefficients,
            "baseline_throughput_feature_order": self.baseline_throughput_feature_order,
            "baseline_throughput_coefficients": self.baseline_throughput_coefficients,
        }
