#!/usr/bin/env python3
"""Compile inputs and check the native C++ MLX simulator; no RTL generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import yaml

from mlxsim.tagged_compiler import compile_graph
from mlxsim.tagged_mlir import compile_mlir, graph_to_mlir
from mlxsim.tagged_program import Program
from mlxsim.tagged_simulator import execute_reference
from mlxsim.tagged_workloads import workload

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "simulator_ext/tagged"
BUILD = ROOT / "build/mlx-tagged-cpp"


def build_native() -> Path:
    BUILD.mkdir(parents=True, exist_ok=True)
    commands = []
    if not (BUILD / "CMakeCache.txt").exists():
        commands.append(
            ["cmake", "-S", str(NATIVE), "-B", str(BUILD), "-DCMAKE_BUILD_TYPE=Release"]
        )
    commands.append(["cmake", "--build", str(BUILD), "-j4"])
    logs = []
    for command in commands:
        result = subprocess.run(
            command, cwd=ROOT, text=True, capture_output=True, timeout=120, check=False
        )
        logs.append(result.stdout + result.stderr)
        if result.returncode:
            (BUILD / "build.log").write_text("\n".join(logs))
            raise RuntimeError(f"native build failed: {BUILD / 'build.log'}")
    (BUILD / "build.log").write_text("\n".join(logs))
    return BUILD / "mlx-tagged-sim"


def run_native(
    program: Program,
    output: Path,
    *,
    overlap=True,
    compute_ii=1,
    memory_delay=3,
    memory_period=1,
    link_period=1,
    receive_period=1,
    trace=True,
    repeat=1,
    max_cycles=100000,
    program_override=None,
    expected_error=None,
):
    program.validate()
    binary = build_native()
    output.mkdir(parents=True, exist_ok=True)
    program_path = output / "program.json"
    result_path = output / "native.json"
    program_path.write_text(
        json.dumps(program.to_dict() if program_override is None else program_override, indent=2)
        + "\n"
    )
    command = [
        str(binary),
        "--program",
        str(program_path),
        "--output",
        str(result_path),
        "--compute-ii",
        str(compute_ii),
        "--load-latency",
        str(memory_delay),
        "--memory-period",
        str(memory_period),
        "--link-period",
        str(link_period),
        "--receive-period",
        str(receive_period),
        "--repeat",
        str(repeat),
        "--max-cycles",
        str(max_cycles),
    ]
    if not overlap:
        command.append("--serial")
    if not trace:
        command.append("--no-trace")
    result = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, timeout=120, check=False
    )
    (output / "run.log").write_text(result.stdout + result.stderr)
    if expected_error is not None:
        if result.returncode != 2 or expected_error not in result.stderr:
            raise RuntimeError(f"expected native rejection {expected_error}: {result.stderr}")
        return {"rejected": True, "error": result.stderr, "returncode": result.returncode}
    if result.returncode or "MLX_TAGGED_CPP_PASS" not in result.stdout:
        raise RuntimeError(f"native execution failed: {result.stderr}")
    data = json.loads(result_path.read_text())
    expected = execute_reference(program)
    actual = {int(address): tuple(vector) for address, vector in data["outputs"].items()}
    if actual != expected:
        raise RuntimeError("native C++ output differs from untimed reference")
    data["program_sha256"] = program.digest()
    data["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
    data["sources"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(NATIVE.iterdir())
        if path.is_file()
    }
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--program", type=Path)
    source.add_argument("--graph", type=Path)
    source.add_argument("--mlir", type=Path)
    source.add_argument("--workload", choices=("bsmm", "fft_cmp", "swa", "transformer_block"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-trace", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--frontend", choices=("mlir", "python"), default="mlir")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_golden = None
    if args.program:
        program = Program.from_dict(yaml.safe_load(args.program.read_text()))
    elif args.mlir:
        program = compile_mlir(
            args.mlir, output / "mlir", build_native().with_name("mlx-mlir-front")
        )
    else:
        if args.workload:
            data, source_golden = workload(args.workload)
        else:
            data = yaml.safe_load(args.graph.read_text())
        if args.frontend == "mlir":
            source_path = output / "source.mlir"
            source_path.write_text(graph_to_mlir(data))
            program = compile_mlir(
                source_path, output / "mlir", build_native().with_name("mlx-mlir-front")
            )
        else:
            program = compile_graph(data)
    results = {
        "program_sha256": program.digest(),
        "abi_version": program.abi_version,
        "classification": "native_cpp_component_cycle_simulation",
    }
    results["native"] = run_native(
        program, output / "native", trace=not args.no_trace, repeat=args.repeat
    )
    results["serial"] = run_native(
        program, output / "serial", overlap=False, trace=not args.no_trace, repeat=args.repeat
    )
    if source_golden is not None:
        aliases = program.lineage.get("source_output_aliases", {})
        for backend in ("native", "serial"):
            actual = {
                name: tuple(
                    results[backend]["outputs"][
                        str(program.lineage["spm_values"][aliases.get(name, name)])
                    ]
                )
                for name in source_golden
            }
            if actual != source_golden:
                raise RuntimeError(f"{backend}: source-operator golden mismatch")
        results["source_operator_golden_passed"] = True
        (output / "source_graph.json").write_text(json.dumps(data, indent=2) + "\n")
    (output / "result.json").write_text(json.dumps(results, indent=2) + "\n")
    print(
        json.dumps(
            {
                "program": program.name,
                "backend": "tagged_cpp_v2",
                "result": str(output / "result.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
