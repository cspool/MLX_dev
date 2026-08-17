from __future__ import annotations

from mlxsim.fig10_scaling import compile_scalability_config


def test_four_hardware_configs_conserve_lane_work() -> None:
    configs = {
        "baseline": (8, (4, 4)),
        "simd32_4x4": (32, (4, 4)),
        "simd8_8x8": (8, (8, 8)),
        "simd32_8x8": (32, (8, 8)),
    }
    work = []
    groups = {}
    for name, (simd, mesh) in configs.items():
        _, metadata = compile_scalability_config(
            sequence_length=512,
            batch=8,
            hidden_width=512,
            hardware_name=name,
            simd_width=simd,
            mesh=mesh,
        )
        work.append(metadata["lane_normalized_work"])
        groups[name] = metadata["outer_vector_groups"]
    assert all(value == work[0] for value in work)
    assert groups == {
        "baseline": 512,
        "simd32_4x4": 128,
        "simd8_8x8": 512,
        "simd32_8x8": 128,
    }


def test_mesh_scaling_changes_spatial_not_logical_work() -> None:
    _, small = compile_scalability_config(
        sequence_length=1024,
        batch=8,
        hidden_width=512,
        hardware_name="small",
        simd_width=8,
        mesh=(4, 4),
    )
    _, large = compile_scalability_config(
        sequence_length=1024,
        batch=8,
        hidden_width=512,
        hardware_name="large",
        simd_width=8,
        mesh=(8, 8),
    )
    assert small["outputs_per_pe_per_stage"] == 32
    assert large["outputs_per_pe_per_stage"] == 8
    assert small["output_instances"] == large["output_instances"]
    assert small["instruction_count"] == large["instruction_count"]
    assert small["lane_normalized_work"] == large["lane_normalized_work"]


def test_scaled_configs_remove_large_address_sequences() -> None:
    document, metadata = compile_scalability_config(
        sequence_length=8192,
        batch=8,
        hidden_width=512,
        hardware_name="baseline",
        simd_width=8,
        mesh=(4, 4),
    )
    assert metadata["outer_vector_groups"] == 8192
    assert all(
        "memory_address_sequence" not in instruction
        for block in document["blocks"]
        for instruction in block["instructions"]
    )
    assert metadata["paper_performance_targets_consumed"] is False
