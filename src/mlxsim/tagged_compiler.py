"""Compile a restricted vector SSA graph into the shared MLX v2 program.

The input contains vector operations and data dependencies, not PE IDs,
register numbers or routes. Mapping packs a bounded topological wave; values
crossing a wave boundary are spilled to liveness-allocated SPM locations.
Within a wave producers send directly to each consumer's incoming register.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mlxsim.tagged_program import ARITY, Hardware, Program, ProgramError
from mlxsim.tagged_regions import lower_regions


def compile_graph(graph: dict[str, Any], hardware: Hardware | None = None) -> Program:
    hw = hardware or Hardware()
    hw.validate()
    if type(graph.get("schema_version")) is not int or graph["schema_version"] not in (1, 2):
        raise ProgramError("unsupported vector graph schema")
    if set(graph) - {"schema_version", "name", "inputs", "operations", "outputs", "iterations"}:
        raise ProgramError("unknown graph fields; spatial assignments are compiler-owned")
    inputs = graph.get("inputs", {})
    operations = graph.get("operations", [])
    outputs = graph.get("outputs", [])
    iterations = graph.get("iterations", 1)
    if type(iterations) is not int or not 1 <= iterations <= 65535:
        raise ProgramError("invalid graph iteration count")
    if not isinstance(inputs, dict) or not isinstance(operations, list) or not operations:
        raise ProgramError("graph requires named inputs and operations")
    values = set(inputs)
    by_id = {}
    for op in operations:
        required = {"id", "op", "inputs"} | (
            {"region", "layer"} if graph["schema_version"] == 2 else set()
        )
        if not isinstance(op, dict) or set(op) != required:
            raise ProgramError("operation fields do not match vector graph schema")
        name = op["id"]
        if not isinstance(name, str) or not name or name in values:
            raise ProgramError("duplicate or invalid SSA value name")
        if op["op"] not in ARITY or op["op"] in ("load", "store", "xfer"):
            raise ProgramError("graph nodes must be supported vector arithmetic")
        if not isinstance(op["inputs"], list) or len(op["inputs"]) != ARITY[op["op"]]:
            raise ProgramError("invalid vector operation arity")
        if not set(op["inputs"]) <= values:
            raise ProgramError("graph must be in topological order with defined operands")
        values.add(name)
        by_id[name] = op
    if not outputs or len(set(outputs)) != len(outputs) or not set(outputs) <= values:
        raise ProgramError("invalid graph outputs")

    # Reverse liveness removes dead computation before it consumes hardware.
    live = set(outputs)
    for op in reversed(operations):
        if op["id"] in live:
            live.update(op["inputs"])
    operations = [op for op in operations if op["id"] in live]
    if not operations:
        raise ProgramError("at least one output must require computation")
    inputs = {name: vector for name, vector in inputs.items() if name in live}
    input_bits = {}
    for name, vector in inputs.items():
        if not isinstance(name, str) or not isinstance(vector, list) or len(vector) != hw.lanes:
            raise ProgramError("each input must be a named SIMD-width vector")
        try:
            array = np.asarray(vector, dtype=np.float16)
        except (TypeError, ValueError) as error:
            raise ProgramError("invalid input vector") from error
        if array.shape != (hw.lanes,):
            raise ProgramError("input vector must be one-dimensional")
        input_bits[name] = tuple(int(v) for v in array.view(np.uint16))

    return lower_regions(graph, operations, input_bits, outputs, hw, iterations)
