import subprocess
from dataclasses import replace

import pytest

from mlxsim.tagged_compiler import compile_graph
from mlxsim.tagged_simulator import TaggedSimulator, execute_reference
from mlxsim.tagged_workloads import workload
from scripts.run_mlx_native_device import build_device, configuration, run_device


def test_native_device_relaunch_error_recovery_and_reset():
    binary = build_device().with_name("mlx-device-lifecycle-test")
    result = subprocess.run([str(binary)], text=True, capture_output=True, timeout=30, check=True)
    assert "MLX_DEVICE_LIFECYCLE_PASS" in result.stdout


@pytest.mark.parametrize("name", ["bsmm", "fft_cmp", "swa", "transformer_block"])
def test_binary_loading_dma_and_same_native_kernel(tmp_path, name):
    graph, _ = workload(name)
    program = compile_graph(graph)
    result = run_device(program, tmp_path, delay=5, period=3)
    assert result["error"] == 0
    assert result["complete"]
    assert {int(a): tuple(v) for a, v in result["host_outputs"].items()} == execute_reference(
        program
    )
    reference = TaggedSimulator(program).run()
    assert result["kernel_cycles"] == reference.cycles
    assert result["system_cycles"] == result["dma_cycles"] + result["kernel_cycles"]
    assert result["host_launch_wait_cycles"] >= result["system_cycles"]
    expected_bytes = (len(program.inputs) + len(program.outputs)) * program.hardware.lanes * 2
    assert result["dma_bytes"] == expected_bytes
    assert result["memory_requests"] == result["memory_responses"] == expected_bytes // 8
    assert result["config_commands"] == len(configuration(program))
    assert result["queried_status"][14] == 0x4D4C5802
    assert result["queried_status"][16] == reference.counters.get("overlap_pe_cycles", 0)


def test_input_comes_from_dma_not_configuration_or_a_golden(tmp_path):
    program = compile_graph(workload("bsmm")[0])
    first = min(program.inputs)
    mutated = replace(
        program, inputs={**program.inputs, first: (0x4400, *program.inputs[first][1:])}
    )
    result = run_device(program, tmp_path, first_half=0x4400)
    actual = {int(a): tuple(v) for a, v in result["host_outputs"].items()}
    assert actual == execute_reference(mutated)
    assert actual != execute_reference(program)


def test_memory_backpressure_changes_dma_but_not_kernel_schedule(tmp_path):
    program = compile_graph(workload("swa")[0])
    fast = run_device(program, tmp_path / "fast", delay=1, period=1)
    slow = run_device(program, tmp_path / "slow", delay=7, period=5)
    assert fast["kernel_cycles"] == slow["kernel_cycles"]
    assert fast["kernel"]["events"] == slow["kernel"]["events"]
    assert fast["system_cycles"] < slow["system_cycles"]
    assert fast["dma_cycles"] < slow["dma_cycles"]


@pytest.mark.parametrize("corruption", ["magic", "route", "missing", "extra"])
def test_invalid_image_is_rejected_before_dma_or_kernel(tmp_path, corruption):
    program = compile_graph(workload("bsmm")[0])
    words = configuration(program)
    if corruption == "magic":
        words[0] ^= 1
    elif corruption == "route":
        address = next(a for a, w in words.items() if 0x1000 <= a < 0x1200 and w >> 60 == 8)
        words[address] ^= 1 << 33
    elif corruption == "missing":
        del words[0x100]
    else:
        words[0x1100] = 0
    result = run_device(program, tmp_path, words=words)
    assert result["error"] != 0
    assert result["kernel_cycles"] == result["memory_requests"] == 0
    assert result["wait_status"] & 16


def test_dpi_signal_bridge_matches_direct_device_handshakes(tmp_path):
    program = compile_graph(workload("bsmm")[0])
    direct = run_device(program, tmp_path / "direct", delay=3, period=2)
    dpi = run_device(program, tmp_path / "dpi", delay=3, period=2, dpi=True)
    for key in (
        "error",
        "system_cycles",
        "dma_cycles",
        "kernel_cycles",
        "dma_bytes",
        "memory_requests",
        "memory_responses",
        "host_outputs",
    ):
        assert direct[key] == dpi[key], key
    assert direct["kernel"] == dpi["kernel"]
