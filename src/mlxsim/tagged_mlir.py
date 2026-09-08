"""Bridge source graphs through the native typed MLIR frontend and spatial lowering."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np

from mlxsim.tagged_compiler import compile_graph
from mlxsim.tagged_program import ARITY, NUMERICS, Hardware, ProgramError


def quote(value: str) -> str:
    # MLIR uses byte escapes, not JSON's Unicode escapes.
    return (
        '"'
        + "".join(
            chr(byte) if 32 <= byte < 127 and byte not in (34, 92) else f"\\{byte:02X}"
            for byte in value.encode()
        )
        + '"'
    )


def graph_to_mlir(graph, hardware: Hardware | None = None) -> str:
    hw = hardware or Hardware()
    hw.validate()
    if graph.get("schema_version") not in (1, 2):
        raise ProgramError("unsupported vector graph schema")
    iterations = graph.get("iterations", 1)
    if type(iterations) is not int or not 1 <= iterations <= 65535:
        raise ProgramError("invalid iteration count")
    vector = f"vector<{hw.lanes}xf16>"
    header = (
        f"module attributes {{mlx.name = {quote(graph.get('name', 'kernel'))}, "
        f"mlx.iterations = {iterations} : i64, mlx.numerics = {quote(NUMERICS)}}} {{"
    )
    lines = [header]
    values = {}
    for index, (name, data) in enumerate(graph["inputs"].items()):
        bits = np.asarray(data, dtype=np.float16).view(np.uint16)
        if bits.shape != (hw.lanes,):
            raise ProgramError("input vector width mismatch")
        symbol = f"%in{index}"
        values[name] = symbol
        elements = ", ".join(f"{int(bit)} : i32" for bit in bits)
        lines.append(
            f'  {symbol} = "mlx.input"() {{name = {quote(name)}, bits = [{elements}]}} '
            f": () -> {vector}"
        )
    for index, op in enumerate(graph["operations"]):
        if op["id"] in values or op["op"] not in ARITY or op["op"] in ("load", "store", "xfer"):
            raise ProgramError("invalid or duplicate arithmetic value")
        if len(op["inputs"]) != ARITY[op["op"]] or not set(op["inputs"]) <= values.keys():
            raise ProgramError("invalid arithmetic operands")
        symbol = f"%v{index}"
        operands = ", ".join(values[value] for value in op["inputs"])
        types = ", ".join(vector for _ in op["inputs"])
        layer = op.get("layer", index)
        if type(layer) is not int or not 0 <= layer < 65536:
            raise ProgramError("invalid logical layer")
        region = op.get("region", f"ssa:{op['id']}")
        lines.append(
            f'  {symbol} = "mlx.compute"({operands}) {{kind = {quote(op["op"])}, '
            f"region = {quote(region)}, layer = {layer} : i64}} : ({types}) -> {vector} "
            f"loc({quote(op['id'])})"
        )
        values[op["id"]] = symbol
    if not graph["outputs"] or not set(graph["outputs"]) <= values.keys():
        raise ProgramError("invalid outputs")
    for name in graph["outputs"]:
        lines.append(f'  "mlx.output"({values[name]}) {{name = {quote(name)}}} : ({vector}) -> ()')
    return "\n".join([*lines, "}", ""])


def compile_mlir(source: Path, output: Path, binary: Path, hardware: Hardware | None = None):
    if not binary.is_file():
        raise RuntimeError("MLIR frontend missing; build with LLVM/MLIR 14 development packages")
    output.mkdir(parents=True, exist_ok=True)
    graph_path = output / "lowered_graph.json"
    optimized = output / "optimized.mlir"
    command = [str(binary), str(source), str(graph_path), str(optimized)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
    (output / "frontend.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise ProgramError(f"MLIR frontend rejected input: {result.stderr}")
    lowered = json.loads(graph_path.read_text())
    hw = hardware or Hardware(lanes=lowered["lanes"])
    if hw.lanes != lowered["lanes"]:
        raise ProgramError("MLIR SIMD width does not match target hardware")
    program = compile_graph(lowered["graph"], hw)
    return replace(
        program,
        lineage={
            **program.lineage,
            "source_output_aliases": lowered["output_aliases"],
            "mlir": {
                **lowered["optimizer"],
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "optimized_sha256": hashlib.sha256(optimized.read_bytes()).hexdigest(),
                "frontend_binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            },
        },
    )
