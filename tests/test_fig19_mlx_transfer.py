import pytest

from mlxsim.fig19_mlx_transfer import (
    compare_mlx_transfer,
    load_transfer_config,
    mapped_workloads,
)


def test_symbolic_mapping_resolves_fabnet_large_shapes() -> None:
    config = load_transfer_config()
    workloads = mapped_workloads(config, 256)
    hidden_fft, token_fft = workloads["attention"]
    ffn1, ffn2 = workloads["ffn"]
    assert (hidden_fft.n, hidden_fft.d, hidden_fft.chunk_length) == (1024, 256, 1024)
    assert (token_fft.n, token_fft.d, token_fft.chunk_length) == (256, 1024, 256)
    assert (ffn1.n, ffn1.d, ffn1.resolved_output_dim, ffn1.block_size) == (
        256,
        1024,
        4096,
        1024,
    )
    assert (ffn2.n, ffn2.d, ffn2.resolved_output_dim, ffn2.block_size) == (
        256,
        4096,
        1024,
        4096,
    )


def test_component_gate_is_not_overridden_by_total_pass() -> None:
    targets = {
        "sequence_lengths": [128],
        "mlx": {
            "attention_latency_ms": [1.0],
            "ffn_latency_ms": [3.0],
            "total_latency_ms": [4.0],
        },
    }
    comparison = compare_mlx_transfer(
        targets,
        [
            {
                "sequence_length": 128,
                "attention_latency_ms": 1.5,
                "ffn_latency_ms": 2.5,
                "total_latency_ms": 4.0,
            }
        ],
        component_tolerance=0.10,
        total_tolerance=0.10,
    )
    assert comparison["total_summary"]["all_points_pass"]
    assert not comparison["component_summary"]["all_points_pass"]
    assert comparison["component_summary"]["max_absolute_relative_error"] == pytest.approx(
        0.5
    )
