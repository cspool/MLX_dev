"""Lower source regions into ordered multi-instruction MLX tagged blocks.

Incoming values have pinned mailboxes. Internal temporaries use static lifetime
allocation, including safe last-use source/destination register reuse. A region
is the scheduling context; its individual arithmetic operations are not contexts.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field

from mlxsim.tagged_program import Block, Hardware, Instruction, Program, ProgramError


@dataclass
class Region:
    name: str
    layer: int
    operations: list[dict] = field(default_factory=list)
    external: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)
    registers: dict[str, int] = field(default_factory=dict)
    arithmetic: list[tuple[str, Instruction]] = field(default_factory=list)
    demand: int = 0
    block_id: int = 0
    pe: int = 0
    wave: int = 0


def allocate_registers(region: Region, limit: int) -> None:
    last_use = {}
    for index, op in enumerate(region.operations):
        for value in op["inputs"]:
            last_use[value] = index
    for value in region.exports:
        last_use[value] = len(region.operations)
    live = {value: index for index, value in enumerate(region.external)}
    pinned = set(region.external)
    region.registers = dict(live)
    region.demand = len(live)
    if len(live) >= limit:
        raise ProgramError(f"region {region.name}: incoming values leave no arithmetic register")
    for index, op in enumerate(region.operations):
        src = tuple(live[value] for value in op["inputs"])
        # Inputs are captured before writeback. A dead internal source can share
        # the destination of this instruction without a dynamic hazard table.
        live = {
            value: reg
            for value, reg in live.items()
            if value in pinned or last_use.get(value, -1) > index
        }
        available = next((reg for reg in range(limit) if reg not in live.values()), None)
        if available is None:
            raise ProgramError(f"region {region.name}: register capacity exceeded")
        live[op["id"]] = available
        region.registers[op["id"]] = available
        region.demand = max(region.demand, available + 1)
        region.arithmetic.append((op["id"], Instruction(op["op"], dst=available, src=src)))


def lower_regions(graph, operations, input_bits, outputs, hw: Hardware, iterations) -> Program:
    groups: dict[str, Region] = {}
    producer: dict[str, Region] = {}
    for index, op in enumerate(operations):
        name = op.get("region", f"ssa:{op['id']}")
        layer = op.get("layer", index)
        if (
            not isinstance(name, str)
            or not name
            or type(layer) is not int
            or not 0 <= layer < 65536
        ):
            raise ProgramError("invalid source region/layer identity")
        if name not in groups:
            groups[name] = Region(name, layer)
        region = groups[name]
        if region.layer != layer:
            raise ProgramError("a region cannot span different logical layers")
        region.operations.append(op)
        producer[op["id"]] = region

    consumers: dict[str, list[Region]] = {name: [] for name in (*input_bits, *producer)}
    predecessors = {name: set() for name in groups}
    for region in groups.values():
        for op in region.operations:
            for value in dict.fromkeys(op["inputs"]):
                if value in producer and producer[value] is region:
                    continue
                if value not in region.external:
                    region.external.append(value)
                    consumers[value].append(region)
                if value in producer:
                    parent = producer[value]
                    if parent.layer > region.layer:
                        raise ProgramError("logical layer order contradicts the dataflow")
                    predecessors[region.name].add(parent.name)
    ordered = []
    pending = dict(groups)
    while pending:
        ready = next((name for name in pending if not predecessors[name] & pending.keys()), None)
        if ready is None:
            raise ProgramError("source region contraction creates a cyclic block graph")
        ordered.append(pending.pop(ready))
    for region in ordered:
        region.exports = [
            op["id"] for op in region.operations if op["id"] in outputs or consumers[op["id"]]
        ]
        allocate_registers(region, hw.registers)

    wave = 0
    contexts = [0] * hw.pes
    registers = [0] * hw.pes
    mapped = set()
    for block_id, region in enumerate(ordered):
        candidates = [
            pe
            for pe in range(hw.pes)
            if contexts[pe] < hw.contexts and registers[pe] + region.demand <= hw.registers
        ]
        if not candidates:
            wave += 1
            contexts = [0] * hw.pes
            registers = [0] * hw.pes
            candidates = list(range(hw.pes))
        parents = [
            producer[value].pe
            for value in region.external
            if value in producer and producer[value].name in mapped and producer[value].wave == wave
        ]

        def cost(pe, parents=parents, contexts=contexts):
            distance = sum(
                abs(pe % hw.columns - parent % hw.columns)
                + abs(pe // hw.columns - parent // hw.columns)
                for parent in parents
            )
            return contexts[pe], distance, pe

        region.block_id = block_id
        region.pe = min(candidates, key=cost)
        region.wave = wave
        mapped.add(region.name)
        contexts[region.pe] += 1
        registers[region.pe] += region.demand

    last_use = {name: max((r.wave for r in users), default=-1) for name, users in consumers.items()}
    for value in outputs:
        last_use[value] = wave + 1
    addresses: dict[str, int] = {}
    occupied = {}

    def allocate(value):
        address = next((a for a in range(hw.spm_vectors) if a not in occupied), None)
        if address is None:
            raise ProgramError("SPM capacity exceeded by live boundary values")
        occupied[address] = value
        addresses[value] = address

    for value in input_bits:
        allocate(value)
    initial = {addresses[value]: bits for value, bits in input_bits.items()}
    for current in range(wave + 1):
        for address, value in list(occupied.items()):
            if last_use[value] < current:
                del occupied[address]
        for region in ordered:
            if region.wave != current:
                continue
            for value in region.exports:
                if value in outputs or any(user.wave > current for user in consumers[value]):
                    allocate(value)

    blocks = []
    lineage = {}
    for region in ordered:
        instructions, origins = [], []

        def emit(inst, value, kind, instructions=instructions, origins=origins):
            origins.append({"pc": len(instructions), "value": value, "kind": kind})
            instructions.append(inst)

        for value in region.external:
            if value in input_bits or producer[value].wave < region.wave:
                emit(
                    Instruction("load", dst=region.registers[value], spm=addresses[value]),
                    value,
                    "boundary_load",
                )
        for value, instruction in region.arithmetic:
            emit(instruction, value, "arithmetic")
        for value in region.exports:
            for target in consumers[value]:
                if target.wave == region.wave:
                    emit(
                        Instruction(
                            "xfer",
                            dst=target.registers[value],
                            src=(region.registers[value],),
                            target=target.block_id,
                        ),
                        value,
                        "intra_wave_dataflow",
                    )
            if value in addresses:
                emit(
                    Instruction("store", src=(region.registers[value],), spm=addresses[value]),
                    value,
                    "boundary_store",
                )
        blocks.append(
            Block(
                region.block_id,
                region.layer,
                region.pe,
                region.wave,
                region.demand,
                tuple(instructions),
                iterations,
            )
        )
        lineage[str(region.block_id)] = {
            "region": region.name,
            "layer": region.layer,
            "value": region.operations[-1]["id"],
            "values": [op["id"] for op in region.operations],
            "register_assignment": region.registers,
            "incoming": region.external,
            "exports": region.exports,
            "instructions": origins,
        }
    program = Program(
        str(graph.get("name", "vector_graph")),
        tuple(blocks),
        initial,
        tuple(addresses[value] for value in outputs),
        hw,
        lineage={
            "frontend": f"vector_ssa_v{graph['schema_version']}",
            "source_sha256": hashlib.sha256(json.dumps(graph, sort_keys=True).encode()).hexdigest(),
            "blocks": lineage,
            "spm_values": addresses,
            "mapping": "bounded_wave_least_occupied_then_manhattan",
            "block_policy": "source_regions_static_local_liveness",
            "operation_counts": dict(Counter(op["op"] for op in operations)),
        },
    )
    program.validate()
    return program
