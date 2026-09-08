import subprocess

import pytest

from mlxsim.tagged_mlir import compile_mlir, graph_to_mlir
from mlxsim.tagged_program import Block, Hardware, Program
from mlxsim.tagged_program import Instruction as I
from scripts.run_mlx_tagged import build_native
from scripts.run_mlx_tagged_control import PROTOCOL_CASES, build_control, run_control


def independent(iterations=4, waves=1, contexts=2, lanes=4):
    blocks = []
    for wave in range(waves):
        for slot in range(contexts):
            operations = (
                I("load", dst=0, spm=slot),
                I("exp" if slot % 2 == 0 else "mul", dst=1, src=(0,) if slot % 2 == 0 else (0, 0)),
                I("store", src=(1,), spm=8 + slot),
            )
            blocks.append(
                Block(
                    60000 + wave * contexts + slot,
                    40000 + wave * contexts + slot,
                    0,
                    wave,
                    2,
                    operations,
                    iterations,
                )
            )
    return Program(
        "rtl-control-independent",
        tuple(blocks),
        {
            slot: tuple(
                (0x3C00, 0x4000, 0x3800, 0x4200)[(slot + lane) % 4] for lane in range(lanes)
            )
            for slot in range(contexts)
        },
        tuple(range(8, 8 + contexts)),
        Hardware(rows=1, columns=1, lanes=lanes),
    )


def loopback_chain():
    return Program(
        "rtl-control-loopback",
        (
            Block(
                37,
                0,
                0,
                0,
                2,
                (
                    I("load", dst=0),
                    I("mul", dst=1, src=(0, 0)),
                    I("xfer", dst=0, src=(1,), target=82),
                ),
                5,
            ),
            Block(
                82,
                1,
                0,
                0,
                3,
                (
                    I("load", dst=1, spm=1),
                    I("add", dst=2, src=(0, 1)),
                    I("xfer", dst=0, src=(2,), target=99),
                ),
                5,
            ),
            Block(
                99,
                2,
                0,
                0,
                2,
                (I("mul", dst=1, src=(0, 0)), I("store", src=(1,), spm=2, stride=1)),
                5,
            ),
        ),
        {0: (0x4000,) * 4, 1: (0x4200,) * 4},
        tuple(range(2, 7)),
        Hardware(rows=1, columns=1, lanes=4),
    )


@pytest.mark.parametrize("contexts", [2, 4])
@pytest.mark.parametrize("memory_period", [1, 5])
@pytest.mark.parametrize("lanes", [4, 32])
def test_rtl_context_frontiers_match_cpp_each_cycle_with_real_issue_feedback(
    tmp_path, contexts, memory_period, lanes
):
    program = independent(contexts=contexts, lanes=lanes)
    overlap = run_control(
        program, tmp_path / "overlap", load_latency=5, memory_period=memory_period
    )
    serial = run_control(
        program, tmp_path / "serial", overlap=False, load_latency=5, memory_period=memory_period
    )
    assert overlap["cycles"] <= serial["cycles"]
    assert overlap["overlap_pe_cycles"] > 0
    assert serial["overlap_pe_cycles"] == 0
    assert overlap["outputs"] == serial["outputs"]
    assert overlap["issue"] == serial["issue"] == contexts * 4 * 3
    assert overlap["state_comparisons"] == overlap["cycles"]
    assert (
        overlap["classification"] == "rtl_pe_frontend_and_fu_with_cpp_memory_services_not_full_rtl"
    )
    assert overlap["fu_execution_clock_only"]


def test_rtl_contexts_reuse_slots_without_truncating_40_logical_block_ids(tmp_path):
    result = run_control(independent(iterations=2, waves=20), tmp_path)
    assert result["retired"] == 40
    assert len({event["block_id"] for event in result["events"]}) == 40
    assert max(event["epoch"] for event in result["events"]) == 19
    assert max(event["context_slot"] for event in result["events"]) == 1


def test_rtl_iteration_identity_survives_8bit_boundary(tmp_path):
    result = run_control(independent(iterations=257), tmp_path)
    assert max(event["iteration"] for event in result["events"]) == 256
    assert result["issue"] == result["complete"] == 257 * 2 * 3


def test_rtl_full_32_word_template_reaches_pc31_and_replays(tmp_path):
    instructions = (
        I("load", dst=0),
        *(I("mul", dst=0, src=(0, 0)) for _ in range(30)),
        I("store", src=(0,), spm=1),
    )
    program = Program(
        "full_template",
        (Block(65535, 65535, 0, 0, 1, instructions, 2),),
        {0: (0x3C00,) * 4},
        (1,),
        Hardware(rows=1, columns=1, lanes=4),
    )
    result = run_control(program, tmp_path)
    assert result["issue"] == result["complete"] == 64
    assert max(event["pc"] for event in result["events"]) == 31


def test_rtl_full_simd32_rf_preserves_every_lane():
    result = subprocess.run(
        [str(build_control(32)), "--protocol", "rf-data"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert "MLX_TAGGED_CONTROL_PROTOCOL_PASS rf-data" in result.stdout


def test_functional_rtl_fu_replaces_cpp_service_with_identical_events(tmp_path):
    program = independent(iterations=3, contexts=4)
    actual = run_control(program, tmp_path / "rtl", compute_ii=17, memory_period=5)
    baseline = run_control(
        program, tmp_path / "cpp", compute_ii=17, memory_period=5, cpp_fu_service=True
    )
    assert actual["classification"] != baseline["classification"]
    for key in ("outputs", "cycles", "issue", "complete", "events", "overlap_pe_cycles"):
        assert actual[key] == baseline[key]


@pytest.mark.parametrize("receive_period", [1, 7])
def test_rtl_mailbox_readiness_and_shared_writeback_match_cpp(tmp_path, receive_period):
    result = run_control(
        loopback_chain(), tmp_path, memory_period=5, receive_period=receive_period, compute_ii=17
    )
    assert result["issue"] == result["complete"] == 40
    assert result["retired"] == 3
    assert all(vector == [0x5220] * 4 for vector in result["outputs"].values())  # (2^2 + 3)^2 = 49


@pytest.mark.parametrize("lanes", [4, 32])
def test_mlir_compiled_three_operand_rf_reads_and_temporary_reuse(tmp_path, lanes):
    operations = [
        ("acc", "fma", ["a", "b", "c"]),
        ("maximum", "max", ["acc", "a"]),
        ("ratio", "div", ["maximum", "b"]),
        ("shuffled", "shuffle", ["ratio"]),
        ("out", "add", ["shuffled", "c"]),
    ]
    graph = {
        "schema_version": 2,
        "name": "three_port_rf",
        "iterations": 3,
        "inputs": {"a": [1.25] * lanes, "b": [2.0] * lanes, "c": [-0.5] * lanes},
        "operations": [
            {"id": name, "op": op, "inputs": inputs, "layer": 0, "region": "body"}
            for name, op, inputs in operations
        ],
        "outputs": ["out"],
    }
    hardware = Hardware(rows=1, columns=1, lanes=lanes)
    source = tmp_path / "source.mlir"
    source.write_text(graph_to_mlir(graph, hardware))
    program = compile_mlir(
        source, tmp_path / "mlir", build_native().with_name("mlx-mlir-front"), hardware
    )
    assert len(program.blocks) == 1
    result = run_control(program, tmp_path / "execution")
    assert list(result["outputs"].values()) == [
        [0x3800 if (lane ^ 1) < lanes // 4 else 0x3E00 for lane in range(lanes)]
    ]


@pytest.mark.parametrize("kind", PROTOCOL_CASES)
def test_rtl_admission_and_completion_protocol_rejects_corruption(kind):
    result = subprocess.run(
        [str(build_control()), "--protocol", kind],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert f"MLX_TAGGED_CONTROL_PROTOCOL_PASS {kind}" in result.stdout
