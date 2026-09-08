"""Behavioral regressions for the GPGPU-inspired MLX scheduling gap plan."""

import ctypes
import json
from collections import Counter
from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest

from mlxsim.tagged_compiler import compile_graph
from mlxsim.tagged_mlir import compile_mlir, graph_to_mlir
from mlxsim.tagged_program import (
    CODE_BASE,
    IMAGE_MAGIC,
    Block,
    Hardware,
    Program,
    ProgramError,
)
from mlxsim.tagged_program import (
    Instruction as I,
)
from mlxsim.tagged_simulator import TaggedSimulator, Timing, execute_reference
from mlxsim.tagged_workloads import workload
from scripts.run_mlx_tagged import BUILD, build_native, run_native


def bits(value, lanes=4):
    return tuple(int(x) for x in np.full(lanes, value, dtype=np.float16).view(np.uint16))


def independent(iterations=1, waves=1):
    hw = Hardware(rows=1, columns=1, lanes=4)
    blocks = []
    for wave in range(waves):
        blocks.extend(
            [
                Block(
                    100 + wave * 2,
                    wave * 2,
                    0,
                    wave,
                    2,
                    (I("load", dst=0), I("exp", dst=1, src=(0,)), I("store", src=(1,), spm=2)),
                    iterations,
                ),
                Block(
                    101 + wave * 2,
                    wave * 2 + 1,
                    0,
                    wave,
                    2,
                    (
                        I("load", dst=0, spm=1),
                        I("mul", dst=1, src=(0, 0)),
                        I("mul", dst=1, src=(1, 1)),
                        I("store", src=(1,), spm=3),
                    ),
                    iterations,
                ),
            ]
        )
    return Program("independent", tuple(blocks), {0: bits(1), 1: bits(2)}, (2, 3), hw)


def chain(iterations=3, columns=4):
    hw = Hardware(rows=1, columns=columns, lanes=4)
    return Program(
        "chain",
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
                iterations,
            ),
            Block(
                82,
                1,
                columns - 1,
                0,
                3,
                (
                    I("load", dst=1, spm=1),
                    I("add", dst=2, src=(0, 1)),
                    I("xfer", dst=0, src=(2,), target=99),
                ),
                iterations,
            ),
            Block(
                99,
                2,
                0,
                0,
                2,
                (I("mul", dst=1, src=(0, 0)), I("store", src=(1,), spm=2, stride=1)),
                iterations,
            ),
        ),
        {0: bits(2), 1: bits(3)},
        tuple(range(2, 2 + iterations)),
        hw,
    )


def checked(program, timing=None, **kwargs):
    result = TaggedSimulator(program, timing, **kwargs).run()
    assert result.outputs == execute_reference(program)
    issue = {e["uid"]: e for e in result.events if e["event"] == "issue"}
    complete = {e["uid"]: e for e in result.events if e["event"] == "complete"}
    assert len(issue) == result.counters["issue"]
    assert issue.keys() == complete.keys()
    for uid in issue:
        assert complete[uid]["cycle"] > issue[uid]["cycle"]
        assert all(
            issue[uid][key] == complete[uid][key]
            for key in ("pe", "block_id", "context_slot", "epoch", "iteration", "pc")
        )
    assert result.counters["admitted"] == result.counters["retired"] == len(program.blocks)
    writes = Counter((e["cycle"], e["pe"]) for e in result.events if e["event"] == "rf_write")
    assert max(writes.values(), default=0) <= 1
    return result


def test_t1_t2_t3_t12_local_contexts_overlap_and_hide_waits():
    program = independent(iterations=4)
    timing = Timing(latencies={**Timing().latencies, "load": 5})
    parallel = checked(program, timing)
    serial = checked(program, replace(timing, overlap=False))
    assert parallel.counters["overlap_pe_cycles"] > 0
    assert serial.counters.get("overlap_pe_cycles", 0) == 0
    assert parallel.cycles < serial.cycles
    assert parallel.counters["issue"] == serial.counters["issue"] == 28
    loads = [e for e in parallel.events if e["event"] == "issue" and e["op"] == "load"]
    completions = {e["uid"]: e["cycle"] for e in parallel.events if e["event"] == "complete"}
    computes = [e for e in parallel.events if e["event"] == "issue" and e["pipeline"] == "compute"]
    assert any(
        load["cycle"] < compute["cycle"] < completions[load["uid"]]
        and load["block_id"] != compute["block_id"]
        for load in loads
        for compute in computes
    )
    assert parallel.outputs[3] == bits(16)


def test_t4_compute_capacity_and_initiation_interval():
    timing = Timing(compute_ii=17)
    result = checked(independent(iterations=5), timing)
    issues = [
        e["cycle"] for e in result.events if e["event"] == "issue" and e["pipeline"] == "compute"
    ]
    assert all(b - a >= 17 for a, b in pairwise(issues))
    assert result.counters["stall_compute_capacity_or_ii"] > 0
    assert result.cycles > checked(independent(iterations=5)).cycles


@pytest.mark.parametrize("columns", [1, 2, 4])
def test_t5_t8_t9_iteration_mailboxes_and_network_backpressure(columns):
    program = chain(columns=columns)
    result = checked(
        program,
        memory_ready=lambda c: c % 5 == 0,
        link_ready=lambda c, source, target: c % 3 == 0,
        receive_ready=lambda c, target: c % 7 == 0,
    )
    assert all(vector == bits(49) for vector in result.outputs.values())
    assert result.counters["issue"] == 8 * 3
    assert result.counters["xfer_sent"] == result.counters["xfer_delivered"] == 6
    assert len({e["event_id"] for e in result.events if e["event"] == "xfer_receive"}) == 6
    assert result.counters["network_delivery_stall"] > 0
    if columns > 1:
        assert result.counters["routed_links"] > 0


def test_t7_logical_ids_do_not_alias_four_bit_slots():
    program = independent(waves=20)
    result = checked(program)
    assert result.counters["retired"] == 40
    assert result.counters["waves_admitted"] == 20
    admissions = [e for e in result.events if e["event"] == "admit"]
    assert {e["context_slot"] for e in admissions} == {0, 1}
    assert len({(e["epoch"], e["context_slot"]) for e in admissions}) == 40
    assert max(b.layer for b in program.blocks) > 16
    code, _ = program.layout()
    assert len(code[0]) == 7  # templates are reused, not 40 ROM copies


@pytest.mark.parametrize(
    "change",
    [
        {"abi_version": 1},
        {"numerics": "fast_math"},
        {"hardware": Hardware(rows=1, columns=1, lanes=4, contexts=1)},
        {"hardware": Hardware(rows=1, columns=1, lanes=4, registers=3)},
        {"hardware": Hardware(rows=1, columns=1, lanes=4, instruction_words=6)},
        {"outputs": (127,)},
    ],
)
def test_t6_t11_illegal_capacity_and_contracts_are_rejected(change):
    with pytest.raises(ProgramError):
        replace(independent(), **change).validate()


def test_reject_shared_memory_race_and_unconsumed_message():
    program = independent()
    block = program.blocks[1]
    bad = replace(block, instructions=(*block.instructions[:-1], I("store", src=(1,), spm=2)))
    with pytest.raises(ProgramError, match="race"):
        replace(program, blocks=(program.blocks[0], bad)).validate()
    program = chain()
    bad = replace(program.blocks[-1], instructions=(I("load", dst=1), I("store", src=(1,), spm=2)))
    with pytest.raises(ProgramError, match="never consumed"):
        replace(program, blocks=(*program.blocks[:-1], bad)).validate()


def test_t11_json_and_binary_image_roundtrip():
    original = chain()
    parsed = Program.from_dict(json.loads(json.dumps(original.to_dict())))
    assert parsed.digest() == original.digest()
    decoded = Program.from_image(
        parsed.image(), name=parsed.name, inputs=parsed.inputs, outputs=parsed.outputs
    )
    assert decoded == original
    assert checked(decoded).outputs == checked(original).outputs
    broken = parsed.image()
    broken[0] = IMAGE_MAGIC - 1
    with pytest.raises(ProgramError, match="magic"):
        Program.from_image(broken, name="bad", inputs=parsed.inputs, outputs=parsed.outputs)
    broken = parsed.image()
    broken[CODE_BASE] |= 1  # reserved instruction bit must not be ignored
    with pytest.raises(ProgramError, match="noncanonical"):
        Program.from_image(broken, name="bad", inputs=parsed.inputs, outputs=parsed.outputs)


def test_stuck_environment_reports_pending_contexts_not_a_pass():
    with pytest.raises(RuntimeError, match="pending contexts"):
        TaggedSimulator(independent(), Timing(max_cycles=30), memory_ready=lambda c: False).run()


def test_t5_stale_iteration_packet_is_rejected():
    sim = TaggedSimulator(chain(iterations=2))
    saved = None
    for _ in range(500):
        sim.step()
        if saved is None and sim.routers:
            saved = next(iter(sim.routers.values()))
        if saved is not None and sim.contexts[saved.target].iteration > saved.operation.iteration:
            sim.routers[saved.target_pe] = saved
            with pytest.raises(RuntimeError, match="late packet"):
                sim.step()
            return
    pytest.fail("fixture did not exercise iteration rollover")


@pytest.mark.parametrize("contexts", [1, 2, 4])
@pytest.mark.parametrize("columns", [1, 2, 4])
def test_t10_automatic_mapping_spills_and_routing(contexts, columns):
    graph = {
        "schema_version": 1,
        "name": "three_stage",
        "iterations": 3,
        "inputs": {"a": [2.0] * 4, "b": [3.0] * 4, "c": [4.0] * 4},
        "operations": [
            {"id": "x", "op": "mul", "inputs": ["a", "b"]},
            {"id": "y", "op": "add", "inputs": ["x", "c"]},
            {"id": "z", "op": "mul", "inputs": ["y", "x"]},
        ],
        "outputs": ["z"],
    }
    program = compile_graph(graph, Hardware(rows=1, columns=columns, lanes=4, contexts=contexts))
    result = checked(program)
    # Independent source-level mathematical expectation; does not call either
    # backend's arithmetic helper or interpret the compiler's generated program.
    assert list(result.outputs.values()) == [bits((2 * 3 + 4) * (2 * 3))]
    assert len(program.lineage["blocks"]) == 3
    assert program.lineage["operation_counts"] == {"mul": 2, "add": 1}
    if columns * contexts >= 3:
        assert result.counters["xfer_delivered"] == 9
    if columns == 1 and contexts == 1:
        assert result.counters["waves_admitted"] == 3


def test_t10_compiler_capacity_failure_is_not_silently_fitted():
    graph = {
        "schema_version": 1,
        "name": "too_big",
        "inputs": {"a": [1.0] * 4, "b": [2.0] * 4},
        "operations": [{"id": "sum", "op": "add", "inputs": ["a", "b"]}],
        "outputs": ["sum"],
    }
    with pytest.raises(ProgramError, match="SPM capacity"):
        compile_graph(graph, Hardware(lanes=4, spm_vectors=2))
    with pytest.raises(ProgramError, match="register"):
        compile_graph(graph, Hardware(lanes=4, registers=2))


def checked_native(program, output, **kwargs):
    result = run_native(program, output, **kwargs)
    timing = Timing(
        latencies={**Timing().latencies, "load": kwargs.get("memory_delay", 3)},
        compute_ii=kwargs.get("compute_ii", 1),
        overlap=kwargs.get("overlap", True),
    )
    reference = checked(
        program,
        timing,
        memory_ready=lambda c: c % kwargs.get("memory_period", 1) == 0,
        link_ready=lambda c, source, target: c % kwargs.get("link_period", 1) == 0,
        receive_ready=lambda c, target: c % kwargs.get("receive_period", 1) == 0,
    )
    assert result["cycles"] == reference.cycles
    for key in result["counters"].keys() | reference.counters.keys():
        assert result["counters"].get(key, 0) == reference.counters.get(key, 0), key
    strip_backend = lambda event: {k: v for k, v in event.items() if k != "backend"}
    assert [strip_backend(e) for e in result["events"]] == [
        strip_backend(e) for e in reference.events
    ]
    return result


@pytest.mark.parametrize("columns", [1, 2, 4])
def test_native_t3_t5_t8_t9_transport_and_iteration_ownership(tmp_path, columns):
    result = checked_native(
        chain(columns=columns),
        tmp_path,
        memory_delay=5,
        memory_period=3,
        link_period=3,
        receive_period=7,
    )
    assert result["counters"]["xfer_sent"] == result["counters"]["xfer_delivered"] == 6
    # Strong network throttling can remove overlap in a dependent chain. The
    # independent-context test below verifies latency hiding separately.
    assert result["counters"]["network_delivery_stall"] > 0


def test_native_t1_t2_t4_t12_overlap_with_unchanged_resources(tmp_path):
    program = independent(iterations=4)
    result = checked_native(program, tmp_path / "overlap", memory_delay=5)
    serial = checked_native(program, tmp_path / "serial", overlap=False, memory_delay=5)
    assert result["cycles"] < serial["cycles"]
    assert result["counters"]["overlap_pe_cycles"] > 0
    assert serial["counters"]["overlap_pe_cycles"] == 0
    slower = checked_native(program, tmp_path / "ii", compute_ii=23)
    issue = [
        e["cycle"] for e in slower["events"] if e["event"] == "issue" and e["pipeline"] == "compute"
    ]
    assert all(b - a >= 23 for a, b in pairwise(issue))


def test_native_t7_slot_reuse_across_40_logical_blocks(tmp_path):
    result = checked_native(independent(waves=20), tmp_path)
    admits = [e for e in result["events"] if e["event"] == "admit"]
    assert len({(e["context_slot"], e["epoch"]) for e in admits}) == 40
    assert {e["context_slot"] for e in admits} == {0, 1}


def test_native_t6_t11_reject_program_abi_and_capacity_without_python_validation(tmp_path):
    program = independent()
    invalid = program.to_dict()
    invalid["abi_version"] = 1
    assert run_native(
        program,
        tmp_path / "abi",
        program_override=invalid,
        expected_error="unsupported program ABI",
    )["rejected"]
    invalid = json.loads(json.dumps(program.to_dict()))
    invalid["blocks"][0]["registers"] = 16
    assert run_native(
        program,
        tmp_path / "capacity",
        program_override=invalid,
        expected_error="admission capacity",
    )["rejected"]


def test_native_t10_compiler_to_cpp_execution(tmp_path):
    graph = {
        "schema_version": 1,
        "name": "automatic",
        "iterations": 3,
        "inputs": {"a": [2.0] * 4, "b": [3.0] * 4, "c": [4.0] * 4},
        "operations": [
            {"id": "x", "op": "mul", "inputs": ["a", "b"]},
            {"id": "y", "op": "add", "inputs": ["x", "c"]},
            {"id": "z", "op": "mul", "inputs": ["y", "x"]},
        ],
        "outputs": ["z"],
    }
    program = compile_graph(graph, Hardware(rows=1, columns=4, lanes=4))
    result = checked_native(program, tmp_path)
    assert result["counters"]["xfer_sent"] == 9
    assert list(result["outputs"].values()) == [list(bits(60))]


def test_native_trace_disabled_keeps_cycles_and_resource_counts(tmp_path):
    program = chain(iterations=8)
    traced = run_native(program, tmp_path / "trace")
    fast = run_native(program, tmp_path / "fast", trace=False, repeat=100)
    assert fast["events"] == []
    assert fast["cycles"] == traced["cycles"]
    assert fast["counters"] == traced["counters"]
    assert fast["outputs"] == traced["outputs"]
    assert fast["native_elapsed_ns"] > 0


@pytest.mark.parametrize("name", ["bsmm", "fft_cmp", "swa", "transformer_block"])
def test_native_registered_workload_from_source_math(tmp_path, name):
    graph, golden = workload(name)
    program = compile_graph(graph)
    result = checked_native(program, tmp_path / "parallel")
    serial = checked_native(program, tmp_path / "serial", overlap=False)
    for execution in (result, serial):
        actual = {
            value: tuple(execution["outputs"][str(program.lineage["spm_values"][value])])
            for value in graph["outputs"]
        }
        assert actual == golden
    assert result["counters"]["issue"] == serial["counters"]["issue"]


def test_native_destination_credit_prevents_future_iteration_router_blocking(tmp_path):
    # This fan-out/fan-in workload deadlocked before endpoint credits and
    # simultaneous registered-buffer dequeue/enqueue were modeled.
    graph, _ = workload("bsmm", iterations=4)
    # Keep the original atomic fan-out/fan-in reproducer even though production
    # workload lowering now groups a complete butterfly into one tagged block.
    graph["schema_version"] = 1
    graph["operations"] = [
        {k: v for k, v in op.items() if k in ("id", "op", "inputs")} for op in graph["operations"]
    ]
    result = checked_native(compile_graph(graph), tmp_path)
    assert result["counters"]["stall_destination_credit"] > 0
    assert result["counters"]["retired"] == 16


@pytest.mark.parametrize("name", ["bsmm", "fft_cmp", "swa", "transformer_block"])
def test_source_regions_preserve_layer_identity_and_coarse_contexts(tmp_path, name):
    graph, golden = workload(name)
    program = compile_graph(graph)
    assert len(program.blocks) < sum(program.lineage["operation_counts"].values())
    assert any(sum(i.pipeline == "compute" for i in b.instructions) >= 4 for b in program.blocks)
    if name == "bsmm":
        assert [b.layer for b in program.blocks] == [0, 0, 1, 1]
        assert all(
            sum(i.pipeline == "compute" for i in b.instructions) == 4 for b in program.blocks
        )
    result = checked_native(program, tmp_path)
    events = [event for event in result["events"] if event["event"] == "admit"]
    assert [e["logical_layer_id"] for e in events] == [
        b.layer for b in sorted(program.blocks, key=lambda b: (b.wave, b.pe, b.layer, b.block_id))
    ]
    assert all(
        tuple(result["outputs"][str(program.lineage["spm_values"][value])]) == bits
        for value, bits in golden.items()
    )


def test_region_liveness_reuses_temporaries_without_dynamic_scoreboarding(tmp_path):
    operations = [{"id": "x0", "op": "add", "inputs": ["a", "b"], "region": "body", "layer": 19}]
    for index in range(1, 20):
        operations.append(
            {
                "id": f"x{index}",
                "op": "mul",
                "inputs": [f"x{index - 1}", "a"],
                "region": "body",
                "layer": 19,
            }
        )
    graph = {
        "schema_version": 2,
        "name": "reuse",
        "inputs": {"a": [1.0] * 4, "b": [2.0] * 4},
        "operations": operations,
        "outputs": ["x19"],
        "iterations": 3,
    }
    program = compile_graph(graph, Hardware(rows=1, columns=1, lanes=4, registers=3))
    assert len(program.blocks) == 1
    assert program.blocks[0].registers == 3
    assert {inst.dst for inst in program.blocks[0].instructions if inst.pipeline == "compute"} == {
        2
    }
    result = checked_native(program, tmp_path)
    assert list(result["outputs"].values()) == [list(bits(3))]


def test_region_contraction_rejects_a_hidden_cross_block_cycle():
    graph = {
        "schema_version": 2,
        "name": "cycle",
        "inputs": {"a": [1.0] * 4},
        "operations": [
            {"id": "x", "op": "mul", "inputs": ["a", "a"], "region": "A", "layer": 0},
            {"id": "y", "op": "add", "inputs": ["x", "a"], "region": "B", "layer": 0},
            {"id": "z", "op": "mul", "inputs": ["y", "a"], "region": "A", "layer": 0},
        ],
        "outputs": ["z"],
    }
    with pytest.raises(ProgramError, match="cyclic block graph"):
        compile_graph(graph, Hardware(lanes=4))


def test_c_api_tick_pause_poll_and_independent_instances(tmp_path):
    build_native()
    library = ctypes.CDLL(str(BUILD / "libmlx_tagged_c.so"))
    library.mlx_tagged_create.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    library.mlx_tagged_create.restype = ctypes.c_void_p
    library.mlx_tagged_tick.argtypes = [ctypes.c_void_p]
    library.mlx_tagged_cycles.argtypes = [ctypes.c_void_p]
    library.mlx_tagged_cycles.restype = ctypes.c_uint64
    library.mlx_tagged_destroy.argtypes = [ctypes.c_void_p]
    for name in ("snapshot", "result", "error"):
        method = getattr(library, f"mlx_tagged_{name}")
        method.argtypes = [ctypes.c_void_p]
        method.restype = ctypes.c_char_p
    program = chain(iterations=4)
    encoded = json.dumps(program.to_dict()).encode()
    error = ctypes.create_string_buffer(512)
    first = library.mlx_tagged_create(encoded, b'{"trace":false}', error, len(error))
    second = library.mlx_tagged_create(encoded, b'{"trace":false}', error, len(error))
    assert first and second, error.value
    try:
        assert library.mlx_tagged_cycles(first) == library.mlx_tagged_cycles(second) == 0
        assert library.mlx_tagged_result(first) is None
        assert b"before model completion" in library.mlx_tagged_error(first)
        for step in range(7):
            assert library.mlx_tagged_tick(first) == 0
            assert library.mlx_tagged_cycles(first) == step + 1
        assert library.mlx_tagged_cycles(second) == 0
        snapshot = json.loads(library.mlx_tagged_snapshot(first))
        assert snapshot["cycles"] == 7 and not snapshot["done"]
        assert len(snapshot["contexts"]) == 3
        assert library.mlx_tagged_cycles(first) == 7  # polling did not execute work
        for handle in (first, second):
            for _ in range(1000):
                status = library.mlx_tagged_tick(handle)
                assert status >= 0, library.mlx_tagged_error(handle)
                if status == 1:
                    break
            else:
                pytest.fail("native tick loop failed to complete")
        one = json.loads(library.mlx_tagged_result(first))
        two = json.loads(library.mlx_tagged_result(second))
        assert one == two
        cycles = library.mlx_tagged_cycles(first)
        assert library.mlx_tagged_tick(first) == 1
        assert library.mlx_tagged_cycles(first) == cycles
        batch = run_native(program, tmp_path, trace=False)
        assert one["cycles"] == batch["cycles"]
        assert one["outputs"] == batch["outputs"]
        assert one["counters"] == batch["counters"]
    finally:
        library.mlx_tagged_destroy(first)
        library.mlx_tagged_destroy(second)
    small_error = ctypes.create_string_buffer(1)
    assert not library.mlx_tagged_create(b'{"abi_version":1}', None, small_error, 1)
    assert small_error.raw == b"\0"
    assert library.mlx_tagged_tick(None) == -1
    library.mlx_tagged_destroy(None)


@pytest.mark.parametrize("name", ["bsmm", "fft_cmp", "swa", "transformer_block"])
def test_typed_mlir_frontend_preserves_source_math_and_layer_regions(tmp_path, name):
    graph, golden = workload(name, lanes=4)
    hardware = Hardware(lanes=4)
    source = tmp_path / "source.mlir"
    source.write_text(graph_to_mlir(graph, hardware))
    program = compile_mlir(
        source, tmp_path / "mlir", build_native().with_name("mlx-mlir-front"), hardware
    )
    result = checked_native(program, tmp_path / "native")
    assert program.lineage["mlir"]["llvm_major"] == 14
    aliases = program.lineage["source_output_aliases"]
    assert all(
        tuple(result["outputs"][str(program.lineage["spm_values"][aliases[value]])]) == bits
        for value, bits in golden.items()
    )
    assert [b.layer for b in program.blocks] == [
        b.layer for b in compile_graph(graph, hardware).blocks
    ]
    assert (tmp_path / "mlir" / "optimized.mlir").is_file()


@pytest.mark.parametrize("same_region", [True, False])
def test_mlir_cse_respects_context_boundaries_and_output_aliases(tmp_path, same_region):
    graph = {
        "schema_version": 2,
        "name": "公共表达式",
        "inputs": {"a": [2.0] * 4},
        "operations": [
            {"id": "x", "op": "mul", "inputs": ["a", "a"], "region": "计算块", "layer": 0},
            {
                "id": "y",
                "op": "mul",
                "inputs": ["a", "a"],
                "region": "计算块" if same_region else "另一个块",
                "layer": 0,
            },
        ],
        "outputs": ["x", "y"],
    }
    hardware = Hardware(lanes=4)
    source = tmp_path / "source.mlir"
    source.write_text(graph_to_mlir(graph, hardware))
    program = compile_mlir(
        source, tmp_path / "mlir", build_native().with_name("mlx-mlir-front"), hardware
    )
    assert program.lineage["mlir"]["input_operations"] == 2
    assert program.lineage["mlir"]["output_operations"] == (1 if same_region else 2)
    aliases = program.lineage["source_output_aliases"]
    assert (aliases["x"] == aliases["y"]) == same_region
    result = checked_native(program, tmp_path / "native")
    assert all(vector == list(bits(4)) for vector in result["outputs"].values())


def test_mlir_rejects_operand_type_mismatch_before_lowering(tmp_path):
    graph, _ = workload("bsmm", lanes=4)
    text = graph_to_mlir(graph, Hardware(lanes=4))
    text = text.replace("-> vector<4xf16> loc", "-> vector<8xf16> loc", 1)
    source = tmp_path / "wrong_type.mlir"
    source.write_text(text)
    with pytest.raises(ProgramError, match="MLIR frontend rejected"):
        compile_mlir(source, tmp_path / "mlir", build_native().with_name("mlx-mlir-front"))


@pytest.mark.parametrize("lanes", [8, 16])
def test_native_fused_regions_spill_between_resource_limited_waves(tmp_path, lanes):
    graph = {
        "schema_version": 2,
        "name": "folded",
        "iterations": 3,
        "inputs": {"a": [2.0] * lanes, "b": [3.0] * lanes, "c": [4.0] * lanes},
        "operations": [
            {"id": "x", "op": "mul", "inputs": ["a", "b"], "region": "layer0", "layer": 0},
            {"id": "y", "op": "add", "inputs": ["x", "c"], "region": "layer0", "layer": 0},
            {"id": "z", "op": "mul", "inputs": ["y", "y"], "region": "layer1", "layer": 1},
            {"id": "out", "op": "add", "inputs": ["z", "a"], "region": "layer1", "layer": 1},
        ],
        "outputs": ["out"],
    }
    hw = Hardware(rows=1, columns=1, lanes=lanes, contexts=1)
    program = compile_graph(graph, hw)
    assert [b.wave for b in program.blocks] == [0, 1]
    assert all(sum(i.pipeline == "compute" for i in b.instructions) == 2 for b in program.blocks)
    result = checked_native(program, tmp_path)
    assert list(result["outputs"].values()) == [list(bits(102, lanes))]
    assert result["counters"]["waves_admitted"] == 2


def test_native_folded_bsmm_overlaps_real_layers_with_identical_resources(tmp_path):
    graph, golden = workload("bsmm", iterations=4)
    program = compile_graph(graph, Hardware(rows=1, columns=2))
    assert [(b.layer, b.pe) for b in program.blocks] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert all(b.registers == 7 for b in program.blocks)
    parallel = checked_native(program, tmp_path / "parallel")
    serial = checked_native(program, tmp_path / "serial", overlap=False)
    assert parallel["cycles"] < serial["cycles"]
    assert parallel["counters"]["overlap_pe_cycles"] > 0
    for kind in ("issue", "issue_compute", "issue_load", "issue_store", "xfer_sent"):
        assert parallel["counters"][kind] == serial["counters"][kind]
    assert all(
        tuple(parallel["outputs"][str(program.lineage["spm_values"][value])]) == bits
        for value, bits in golden.items()
    )
