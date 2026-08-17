from mlxsim.fig19_fused_fft2d import fuse_fft2d_profile, load_fusion_config
from mlxsim.fig19_mlx_transfer import load_transfer_config
from mlxsim.schema import CalibrationConfig, HardwareConfig, Workload
from mlxsim.simulator import MLXSimulator
from mlxsim.workloads import compile_workload


def test_simulate_profile_is_exact_public_entrypoint() -> None:
    hardware = HardwareConfig.from_yaml("configs/hardware/mlx_full.yaml")
    calibration = CalibrationConfig.from_yaml("configs/calibration/paper_v1.yaml")
    simulator = MLXSimulator(hardware, calibration)
    workload = Workload(kernel="fft", n=128, d=512, chunk_length=128)
    direct = simulator.simulate(workload).to_dict()
    precompiled = simulator.simulate_profile(workload, compile_workload(workload)).to_dict()
    assert precompiled == direct


def test_fusion_changes_only_registered_boundaries() -> None:
    fusion_config = load_fusion_config()
    base_config = load_transfer_config(fusion_config["base_transfer_config"])
    _, profile, invariants = fuse_fft2d_profile(
        base_config, 128, fusion_config["fusion"]
    )
    assert invariants["pass"]
    assert invariants["operations_before"] == invariants["operations_after"]
    assert invariants["handoff_bytes"] == 128 * 1024 * 4
    assert invariants["hidden_final_store_bytes_after"] == 0.0
    assert invariants["hidden_final_transfer_bytes_after"] == 128 * 1024 * 4
    assert invariants["token_initial_load_bytes_after"] == 0.0
    assert profile.metadata["launch_count"] == 1
