from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .schema import KernelProfile


@dataclass(frozen=True)
class SequenceEfficiency:
    sequence_lengths: tuple[int, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.sequence_lengths) != len(self.values) or not self.values:
            raise ValueError(
                "efficiency sequence lengths and values must have equal nonzero length"
            )
        if tuple(sorted(self.sequence_lengths)) != self.sequence_lengths:
            raise ValueError("efficiency sequence lengths must be sorted")
        if min(self.sequence_lengths) < 1 or not all(0 < value <= 1 for value in self.values):
            raise ValueError("efficiency anchors must have positive lengths and values in (0, 1]")

    def at(self, n: int) -> float:
        if n <= self.sequence_lengths[0]:
            return self.values[0]
        if n >= self.sequence_lengths[-1]:
            return self.values[-1]
        log_n = math.log2(n)
        for left in range(len(self.sequence_lengths) - 1):
            n0, n1 = self.sequence_lengths[left : left + 2]
            if n0 <= n <= n1:
                fraction = (log_n - math.log2(n0)) / (math.log2(n1) - math.log2(n0))
                return self.values[left] + fraction * (self.values[left + 1] - self.values[left])
        raise AssertionError("interpolation interval not found")


@dataclass(frozen=True)
class GpuBaselineConfig:
    name: str
    cuda_peak_gops: float
    tensor_peak_gops: float
    bandwidth_gbs: float
    power_w: float
    memory_capacity_gb: float
    cuda_efficiency: SequenceEfficiency
    tensor_efficiency: SequenceEfficiency
    provenance: dict[str, str]

    @classmethod
    def from_yaml(cls, path: str | Path) -> GpuBaselineConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, dict):
            raise TypeError(f"GPU baseline config must contain a mapping: {path}")
        raw["cuda_efficiency"] = SequenceEfficiency(
            tuple(raw["cuda_efficiency"]["sequence_lengths"]),
            tuple(raw["cuda_efficiency"]["values"]),
        )
        raw["tensor_efficiency"] = SequenceEfficiency(
            tuple(raw["tensor_efficiency"]["sequence_lengths"]),
            tuple(raw["tensor_efficiency"]["values"]),
        )
        return cls(**raw)

    def __post_init__(self) -> None:
        if (
            min(
                self.cuda_peak_gops,
                self.tensor_peak_gops,
                self.bandwidth_gbs,
                self.power_w,
                self.memory_capacity_gb,
            )
            <= 0
        ):
            raise ValueError("GPU peaks, bandwidth, power, and memory capacity must be positive")

    def predict(self, profile: KernelProfile, n: int, mode: str) -> dict[str, float | str]:
        if mode == "cuda":
            peak = self.cuda_peak_gops
            efficiency = self.cuda_efficiency.at(n)
        elif mode == "tensor":
            peak = self.tensor_peak_gops
            efficiency = self.tensor_efficiency.at(n)
        else:
            raise ValueError(f"GPU mode must be 'cuda' or 'tensor', got {mode!r}")
        operational_intensity = profile.operations / max(profile.offchip_bytes, 1.0)
        roofline_gops = min(peak, operational_intensity * self.bandwidth_gbs)
        achieved_gops = efficiency * roofline_gops
        latency_s = profile.operations / (achieved_gops * 1e9)
        return {
            "mode": mode,
            "operations": profile.operations,
            "offchip_bytes": profile.offchip_bytes,
            "operational_intensity": operational_intensity,
            "efficiency": efficiency,
            "roofline_gops": roofline_gops,
            "achieved_gops": achieved_gops,
            "latency_us": latency_s * 1e6,
            "energy_mj": self.power_w * latency_s * 1000.0,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
