from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class HardwareConfig:
    name: str
    frequency_ghz: float
    mesh_x: int
    mesh_y: int
    simd_width: int
    instructions_per_pe: int = 32
    active_tags: int = 16
    spm_bandwidth_gbs: float = 64.0
    noc_link_bytes_per_cycle: float = 8.0
    skip_distance: int = 4
    max_skip_hops: int = 2
    compute_issue_efficiency: float = 0.96
    launch_cycles: float = 18.0
    load_latency_cycles: float = 1.0
    compute_latency_cycles: float = 1.0
    xfer_latency_cycles: float = 1.0
    store_latency_cycles: float = 1.0
    core_power_w: float = 0.4338
    memory_power_w: float = 0.11
    idle_power_fraction: float = 0.12
    count_fma_as_two_ops: bool = True
    decoupled_pipelines: bool = True
    heterogeneous_compute: bool = True
    max_event_waves: int = 4096

    def __post_init__(self) -> None:
        if min(self.frequency_ghz, self.spm_bandwidth_gbs, self.noc_link_bytes_per_cycle) <= 0:
            raise ValueError("frequency and bandwidths must be positive")
        if min(self.mesh_x, self.mesh_y, self.simd_width, self.active_tags) < 1:
            raise ValueError("mesh, SIMD width, and active tag count must be positive")
        if not 0 < self.compute_issue_efficiency <= 1:
            raise ValueError("compute_issue_efficiency must be in (0, 1]")
        if not 0 <= self.idle_power_fraction <= 1:
            raise ValueError("idle_power_fraction must be in [0, 1]")

    @property
    def pe_count(self) -> int:
        return self.mesh_x * self.mesh_y

    @property
    def ops_per_lane_cycle(self) -> int:
        return 2 if self.count_fma_as_two_ops else 1

    @property
    def peak_ops_per_cycle(self) -> float:
        return self.pe_count * self.simd_width * self.ops_per_lane_cycle

    @property
    def peak_tops(self) -> float:
        return self.peak_ops_per_cycle * self.frequency_ghz / 1000.0

    @property
    def spm_bytes_per_cycle(self) -> float:
        return self.spm_bandwidth_gbs / self.frequency_ghz

    @property
    def noc_bytes_per_cycle(self) -> float:
        horizontal = self.mesh_y * max(1, self.mesh_x - 1)
        vertical = self.mesh_x * max(1, self.mesh_y - 1)
        skip = self.pe_count if self.skip_distance > 1 else 0
        return (horizontal + vertical + skip) * self.noc_link_bytes_per_cycle

    @property
    def full_power_w(self) -> float:
        return self.core_power_w + self.memory_power_w

    @classmethod
    def from_yaml(cls, path: str | Path) -> HardwareConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, dict):
            raise TypeError(f"hardware config must be a mapping: {path}")
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            pe_count=self.pe_count,
            peak_ops_per_cycle=self.peak_ops_per_cycle,
            peak_tops=self.peak_tops,
            spm_bytes_per_cycle=self.spm_bytes_per_cycle,
            noc_bytes_per_cycle=self.noc_bytes_per_cycle,
        )
        return result


@dataclass(frozen=True)
class CalibrationConfig:
    """Named mechanism-level parameters not disclosed by the target paper."""

    name: str = "identity"
    kernel_issue_scale: dict[str, float] = field(default_factory=dict)
    kernel_compute_setup_cycles: dict[str, float] = field(default_factory=dict)
    mesh_reference: int = 4
    mesh_reference_n: int = 512
    mesh_congestion_start_n: int = 2048
    mesh_base_penalty: float = 0.0
    mesh_fill_penalty: float = 0.0
    mesh_congestion_penalty: float = 0.0

    def __post_init__(self) -> None:
        if min(self.mesh_reference, self.mesh_reference_n, self.mesh_congestion_start_n) < 1:
            raise ValueError("mesh calibration reference values must be positive")
        if any(value <= 0 for value in self.kernel_issue_scale.values()):
            raise ValueError("kernel issue scales must be positive")
        if any(value < 0 for value in self.kernel_compute_setup_cycles.values()):
            raise ValueError("kernel compute setup cycles cannot be negative")

    @classmethod
    def from_yaml(cls, path: str | Path) -> CalibrationConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, dict):
            raise TypeError(f"calibration config must be a mapping: {path}")
        return cls(**raw)

    def issue_scale(self, kernel_class: str) -> float:
        return self.kernel_issue_scale.get(kernel_class, 1.0)

    def compute_setup_cycles(self, kernel_class: str, default: float) -> float:
        return self.kernel_compute_setup_cycles.get(kernel_class, default)

    def mesh_efficiency(self, hardware: HardwareConfig, n: int) -> float:
        largest_dimension = max(hardware.mesh_x, hardware.mesh_y)
        if largest_dimension <= self.mesh_reference:
            return 1.0
        scale = largest_dimension / self.mesh_reference - 1.0
        fill = self.mesh_fill_penalty / math.sqrt(max(n / self.mesh_reference_n, 1e-9))
        congestion_steps = max(0.0, math.log2(n / self.mesh_congestion_start_n))
        congestion = self.mesh_congestion_penalty * congestion_steps
        efficiency = 1.0 - scale * (self.mesh_base_penalty + fill + congestion)
        return min(1.0, max(0.25, efficiency))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Workload:
    kernel: str
    n: int
    d: int
    batch: int = 1
    block_size: int = 32
    compression_ratio: float = 0.5
    chunk_length: int = 64
    window: int = 128
    query_block: int = 32
    projections: int = 1
    name: str = ""

    def __post_init__(self) -> None:
        supported = {"bsmm", "fft", "fft_cmp", "gemm", "swa", "transformer"}
        if self.kernel not in supported:
            raise ValueError(
                f"unsupported kernel {self.kernel!r}; expected one of {sorted(supported)}"
            )
        if min(self.n, self.d, self.batch, self.projections) < 1:
            raise ValueError("n, d, batch, and projections must be positive")
        if self.block_size < 2 or self.block_size & (self.block_size - 1):
            raise ValueError("block_size must be a power of two >= 2")
        if self.chunk_length < 2 or self.chunk_length & (self.chunk_length - 1):
            raise ValueError("chunk_length must be a power of two >= 2")
        if not 0 < self.compression_ratio <= 1:
            raise ValueError("compression_ratio must be in (0, 1]")

    @property
    def label(self) -> str:
        return self.name or f"{self.kernel}-N{self.n}-D{self.d}-B{self.block_size}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StageSpec:
    tag: int
    name: str
    compute_resource: str
    operations: float
    load_bytes: float = 0.0
    transfer_bytes: float = 0.0
    store_bytes: float = 0.0
    route_distance: int = 0
    kernel_class: str = "generic"


@dataclass(frozen=True)
class KernelProfile:
    operations: float
    offchip_bytes: float
    output_elements: float
    stages: tuple[StageSpec, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationResult:
    hardware: str
    workload: str
    cycles: float
    latency_us: float
    operations: float
    offchip_bytes: float
    achieved_gops: float
    average_power_w: float
    energy_mj: float
    resource_busy_cycles: dict[str, float]
    resource_utilization: dict[str, float]
    compute_utilization: float
    waves: int
    event_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
