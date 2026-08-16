from dataclasses import replace
from pathlib import Path

import pytest

from mlxsim.schema import HardwareConfig, Workload
from mlxsim.simulator import MLXSimulator

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def reduced() -> HardwareConfig:
    return HardwareConfig.from_yaml(ROOT / "configs/hardware/mlx_reduced.yaml")


def test_simulation_is_deterministic(reduced: HardwareConfig) -> None:
    workload = Workload(kernel="bsmm", n=128, d=128, block_size=16)
    first = MLXSimulator(reduced).simulate(workload)
    second = MLXSimulator(reduced).simulate(workload)
    assert first.to_dict() == second.to_dict()


def test_resource_utilization_conserves_busy_cycles(reduced: HardwareConfig) -> None:
    result = MLXSimulator(reduced).simulate(
        Workload(kernel="fft_cmp", n=128, d=128, chunk_length=64)
    )
    for resource, busy in result.resource_busy_cycles.items():
        assert 0 < busy <= result.cycles
        assert result.resource_utilization[resource] == pytest.approx(busy / result.cycles)
    assert 0 < result.compute_utilization <= 1


def test_more_active_tags_do_not_slow_pipeline(reduced: HardwareConfig) -> None:
    workload = Workload(kernel="fft_cmp", n=256, d=128, chunk_length=64)
    serial = MLXSimulator(replace(reduced, active_tags=1)).simulate(workload)
    overlapped = MLXSimulator(replace(reduced, active_tags=16)).simulate(workload)
    assert overlapped.cycles <= serial.cycles


def test_decoupling_improves_or_matches_latency(reduced: HardwareConfig) -> None:
    workload = Workload(kernel="bsmm", n=256, d=256, block_size=32)
    unified = MLXSimulator(replace(reduced, decoupled_pipelines=False)).simulate(workload)
    decoupled = MLXSimulator(reduced).simulate(workload)
    assert decoupled.cycles <= unified.cycles


def test_trace_is_bounded(reduced: HardwareConfig) -> None:
    result = MLXSimulator(reduced, trace_limit=5).simulate(Workload(kernel="gemm", n=8, d=8))
    assert len(result.metadata["trace"]) == 5
