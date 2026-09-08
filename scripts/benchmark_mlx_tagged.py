#!/usr/bin/env python3
"""Measure native simulator throughput separately from simulated MLX speedup."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from mlxsim.tagged_compiler import compile_graph
from mlxsim.tagged_simulator import TaggedSimulator, Timing
from mlxsim.tagged_workloads import workload
from scripts.run_mlx_tagged import run_native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-repeat", type=int, default=1000)
    parser.add_argument("--reference-repeat", type=int, default=10)
    args = parser.parse_args()
    if args.native_repeat < 1 or args.reference_repeat < 1:
        parser.error("repeat counts must be positive")
    graph, _ = workload("transformer_block")
    program = compile_graph(graph)
    args.output.mkdir(parents=True, exist_ok=True)
    native = run_native(program, args.output / "native", trace=False, repeat=args.native_repeat)
    elapsed = 0
    reference = None
    for _ in range(args.reference_repeat):
        machine = TaggedSimulator(program, Timing(trace=False))
        start = time.perf_counter_ns()
        reference = machine.run()
        elapsed += time.perf_counter_ns() - start
    assert reference is not None
    if reference.cycles != native["cycles"]:
        raise RuntimeError("benchmark backends used different simulated execution")
    for key in native["counters"].keys() | reference.counters.keys():
        if native["counters"].get(key, 0) != reference.counters.get(key, 0):
            raise RuntimeError(f"benchmark resource mismatch: {key}")
    if reference.outputs != {int(k): tuple(v) for k, v in native["outputs"].items()}:
        raise RuntimeError("benchmark numerical mismatch")
    cpp_ns = native["native_elapsed_ns"] / args.native_repeat
    py_ns = elapsed / args.reference_repeat
    result = {
        "classification": "simulator_host_execution_throughput_not_hardware_speedup",
        "program_sha256": program.digest(),
        "native_binary_sha256": native["binary_sha256"],
        "native_sources": native["sources"],
        "tracing": False,
        "simulated_cycles_per_run": native["cycles"],
        "native_repeat": args.native_repeat,
        "reference_repeat": args.reference_repeat,
        "native_ns_per_run": cpp_ns,
        "reference_ns_per_run": py_ns,
        "native_simulated_cycles_per_second": native["cycles"] * 1e9 / cpp_ns,
        "reference_simulated_cycles_per_second": native["cycles"] * 1e9 / py_ns,
        "reference_over_native_wall_time": py_ns / cpp_ns,
        "timing_scope": "native includes program validation, owned model construction and result assembly; Python times run only and excludes validation/construction; both exclude process startup, file IO and tracing",
    }
    (args.output / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
