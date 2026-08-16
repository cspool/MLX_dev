from dataclasses import replace
from pathlib import Path

import pytest

from mlxsim.schema import HardwareConfig, Workload

ROOT = Path(__file__).resolve().parents[1]


def test_reduced_peak_matches_paper_configuration() -> None:
    hardware = HardwareConfig.from_yaml(ROOT / "configs/hardware/mlx_reduced.yaml")
    assert hardware.pe_count == 16
    assert hardware.peak_ops_per_cycle == 256
    assert hardware.peak_tops == pytest.approx(0.256)


def test_full_peak_is_one_top_class() -> None:
    hardware = HardwareConfig.from_yaml(ROOT / "configs/hardware/mlx_full.yaml")
    assert hardware.peak_ops_per_cycle == 1024
    assert hardware.peak_tops == pytest.approx(1.024)


def test_invalid_non_power_of_two_block_rejected() -> None:
    with pytest.raises(ValueError, match="power of two"):
        Workload(kernel="bsmm", n=512, d=512, block_size=24)


def test_replace_preserves_validation() -> None:
    hardware = HardwareConfig.from_yaml(ROOT / "configs/hardware/mlx_reduced.yaml")
    scaled = replace(hardware, mesh_x=8, mesh_y=8)
    assert scaled.pe_count == 64
    assert scaled.peak_tops == pytest.approx(1.024)
