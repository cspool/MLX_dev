"""Bounded MLX multi-context execution with registered cycle boundaries.

All decisions observe the state at the start of a cycle. Completions, packet
moves and issues commit at its end. A newly visible value can issue next cycle.
One instruction issues per PE/cycle, one RF write commits per PE/cycle, one
memory request is outstanding globally, and each router has one vector buffer.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from mlxsim.tagged_program import Block, Instruction, Program


def arithmetic(op: str, operands: list[tuple[int, ...]]) -> tuple[int, ...]:
    values = [np.asarray(v, dtype=np.uint16).view(np.float16) for v in operands]
    with np.errstate(all="ignore"):
        if op == "fma":
            result = (values[0] * values[1]).astype(np.float16)
            result = (result + values[2]).astype(np.float16)
        elif op == "mul":
            result = (values[0] * values[1]).astype(np.float16)
        elif op == "add":
            result = (values[0] + values[1]).astype(np.float16)
        elif op == "max":
            result = np.maximum(values[0], values[1])
        elif op == "exp":
            result = np.exp(values[0].astype(np.float32)).astype(np.float16)
            result[len(result) // 4 :] = values[0][len(result) // 4 :]
        elif op == "div":
            result = (values[0] / values[1]).astype(np.float16)
            result[len(result) // 4 :] = values[0][len(result) // 4 :]
        elif op == "shuffle":
            result = values[0][np.arange(len(values[0])) ^ 1]
        else:
            raise ValueError(f"unsupported arithmetic operation {op}")
    return tuple(int(v) for v in np.asarray(result, dtype=np.float16).view(np.uint16))


def execute_reference(program: Program) -> dict[int, tuple[int, ...]]:
    """Untimed DAG interpreter: intentionally has no scheduler/resource model."""
    program.validate()
    memory = dict(program.inputs)
    for wave in sorted({b.wave for b in program.blocks}):
        blocks = {b.block_id: b for b in program.blocks if b.wave == wave}
        incoming: dict[tuple[int, int, int], tuple[int, ...]] = {}
        predecessors = {bid: set() for bid in blocks}
        for block in blocks.values():
            for inst in block.instructions:
                if inst.op == "xfer":
                    predecessors[inst.target].add(block.block_id)
        done = set()
        while len(done) != len(blocks):
            block = next(
                b for bid, b in blocks.items() if bid not in done and predecessors[bid] <= done
            )
            for iteration in range(block.trip_count):
                registers = {
                    reg: value
                    for (bid, it, reg), value in incoming.items()
                    if bid == block.block_id and it == iteration
                }
                for inst in block.instructions:
                    if inst.op == "load":
                        registers[inst.dst] = memory[inst.spm + iteration * inst.stride]
                    elif inst.op == "store":
                        memory[inst.spm + iteration * inst.stride] = registers[inst.src[0]]
                    elif inst.op == "xfer":
                        incoming[inst.target, iteration, inst.dst] = registers[inst.src[0]]
                    else:
                        registers[inst.dst] = arithmetic(inst.op, [registers[r] for r in inst.src])
            done.add(block.block_id)
    return {address: memory[address] for address in program.outputs}


@dataclass(frozen=True)
class Timing:
    latencies: dict[str, int] = field(
        default_factory=lambda: {
            "load": 3,
            "store": 1,
            "fma": 4,
            "add": 3,
            "max": 1,
            "exp": 8,
            "div": 12,
            "shuffle": 1,
            "mul": 3,
        }
    )
    compute_ii: int = 1
    overlap: bool = True
    trace: bool = True
    max_cycles: int = 100_000

    def validate(self) -> None:
        required = {"load", "store", "fma", "add", "max", "exp", "div", "shuffle", "mul"}
        if set(self.latencies) != required or any(
            type(v) is not int or v < 1 for v in self.latencies.values()
        ):
            raise ValueError("all supported operations require a positive integer latency")
        if type(self.compute_ii) is not int or self.compute_ii < 1 or self.max_cycles < 1:
            raise ValueError("invalid compute II or cycle budget")


@dataclass
class Context:
    block: Block
    slot: int
    base: int
    epoch: int
    pc: int = 0
    iteration: int = 0
    valid: set[int] = field(default_factory=set)
    inflight: bool = False
    retired: bool = False

    @property
    def inst(self) -> Instruction:
        return self.block.instructions[self.pc]

    @property
    def uid(self) -> str:
        return f"{self.epoch}:{self.block.block_id}:{self.iteration}:{self.pc}"


@dataclass
class Operation:
    owner: Context
    uid: str
    iteration: int
    pc: int
    inst: Instruction
    data: tuple[int, ...]
    due: int
    address: int = 0


@dataclass
class Packet:
    operation: Operation
    target: int
    register: int
    target_pe: int


@dataclass
class Result:
    outputs: dict[int, tuple[int, ...]]
    cycles: int
    events: list[dict]
    counters: dict[str, int]
    program_digest: str


class TaggedSimulator:
    def __init__(
        self,
        program: Program,
        timing: Timing | None = None,
        *,
        memory_ready: Callable[[int], bool] | None = None,
        link_ready: Callable[[int, int, int], bool] | None = None,
        receive_ready: Callable[[int, int], bool] | None = None,
    ):
        program.validate()
        self.program = program
        self.hw = program.hardware
        self.timing = timing or Timing()
        self.timing.validate()
        self.memory_ready = memory_ready or (lambda cycle: True)
        self.link_ready = link_ready or (lambda cycle, source, target: True)
        self.receive_ready = receive_ready or (lambda cycle, target: True)
        self.memory = dict(program.inputs)
        self.rf = [
            [(0,) * self.hw.lanes for _ in range(self.hw.registers)] for _ in range(self.hw.pes)
        ]
        self.contexts: dict[int, Context] = {}
        self.compute: dict[int, Operation] = {}
        self.next_compute = [0] * self.hw.pes
        self.memory_job: Operation | None = None
        self.routers: dict[int, Packet] = {}
        self.events: list[dict] = []
        self.counters: Counter = Counter()
        self.cycle = 0
        self.wave = -1
        self.last_wait: dict[int, str] = {}
        self.finished = False

    def emit(
        self,
        event: str,
        context: Context,
        *,
        inst: Instruction | None = None,
        uid: str | None = None,
        **extra,
    ) -> None:
        if not self.timing.trace:
            return
        instruction = inst or context.inst
        self.events.append(
            {
                "backend": "tagged_cycle_v2",
                "cycle": self.cycle,
                "phase": "edge_commit",
                "event": event,
                "pe": context.block.pe,
                "logical_layer_id": context.block.layer,
                "block_id": context.block.block_id,
                "context_slot": context.slot,
                "epoch": context.epoch,
                "iteration": context.iteration,
                "pc": context.pc,
                "op": instruction.op,
                "pipeline": instruction.pipeline,
                "uid": uid or context.uid,
                **extra,
            }
        )

    def admit(self) -> None:
        self.wave += 1
        self.contexts = {}
        self.last_wait = {}
        for pe in range(self.hw.pes):
            blocks = sorted(
                (b for b in self.program.blocks if b.pe == pe and b.wave == self.wave),
                key=lambda b: (b.layer, b.block_id),
            )
            base = 0
            for slot, block in enumerate(blocks):
                ctx = Context(block, slot, base, self.wave)
                self.contexts[block.block_id] = ctx
                self.emit("admit", ctx, rf_base=base, registers=block.registers)
                self.counters["admitted"] += 1
                self.counters["registers_reserved"] += block.registers
                base += block.registers
        self.counters["waves_admitted"] += 1

    def complete(self, operation: Operation) -> None:
        ctx = operation.owner
        if not ctx.inflight or ctx.uid != operation.uid or ctx.retired:
            raise RuntimeError(f"stale or duplicate completion: {operation.uid}")
        self.emit("complete", ctx, uid=operation.uid)
        self.counters["complete"] += 1
        self.counters[f"complete_{operation.inst.pipeline}"] += 1
        ctx.inflight = False
        if ctx.pc + 1 < len(ctx.block.instructions):
            ctx.pc += 1
        elif ctx.iteration + 1 < ctx.block.trip_count:
            self.emit("iteration_complete", ctx)
            ctx.iteration += 1
            ctx.pc = 0
            ctx.valid.clear()
        else:
            self.emit("retire", ctx)
            ctx.retired = True
            self.counters["retired"] += 1
            self.counters["registers_released"] += ctx.block.registers

    def next_hop(self, source: int, target: int) -> int:
        sx, sy = source % self.hw.columns, source // self.hw.columns
        tx, ty = target % self.hw.columns, target // self.hw.columns
        if sx != tx:
            return source + (1 if tx > sx else -1) * min(2, abs(tx - sx))
        return source + (1 if ty > sy else -1) * min(2, abs(ty - sy)) * self.hw.columns

    def step(self) -> None:
        if self.finished:
            raise RuntimeError("program has already finished")
        if not self.contexts:
            self.admit()
            self.cycle += 1
            return

        # Plan all edge actions from old state, including RF port reservations.
        writers: set[int] = set()
        completions: list[Operation] = []
        deliveries: list[tuple[int, Packet, Context]] = []
        if self.memory_job is not None and self.memory_job.due <= self.cycle:
            operation = self.memory_job
            completions.append(operation)
            if operation.inst.op == "load":
                writers.add(operation.owner.block.pe)
        for pe, operation in sorted(self.compute.items()):
            if operation.due <= self.cycle and pe not in writers:
                writers.add(pe)
                completions.append(operation)
            elif operation.due <= self.cycle:
                self.counters["writeback_stall"] += 1
        moves: list[tuple[int, int, Packet]] = []
        reserved_router: set[int] = set()
        proposals: dict[int, int] = {}
        for pe, packet in sorted(self.routers.items()):
            if pe == packet.target_pe:
                target = self.contexts[packet.target]
                if target.retired or packet.operation.iteration < target.iteration:
                    raise RuntimeError("late packet for retired context/iteration")
                if (
                    target.iteration == packet.operation.iteration
                    and packet.register not in target.valid
                    and pe not in writers
                    and self.receive_ready(self.cycle, pe)
                ):
                    writers.add(pe)
                    deliveries.append((pe, packet, target))
                else:
                    self.counters["network_delivery_stall"] += 1
            else:
                destination = self.next_hop(pe, packet.target_pe)
                if destination not in reserved_router and self.link_ready(
                    self.cycle, pe, destination
                ):
                    proposals[pe] = destination
                    reserved_router.add(destination)
                else:
                    self.counters["network_route_stall"] += 1

        # A registered buffer may dequeue and enqueue at the same edge. Keep
        # cycles of simultaneous exchanges; remove paths ending at a blocked
        # buffer. This uses one buffer per router, not an unbounded escape queue.
        delivering = {pe for pe, _, _ in deliveries}
        while True:
            blocked = {
                source
                for source, target in proposals.items()
                if target in self.routers and target not in delivering and target not in proposals
            }
            if not blocked:
                break
            for source in blocked:
                del proposals[source]
                self.counters["network_route_stall"] += 1
        reserved_router = set(proposals.values())
        moves = [(source, target, self.routers[source]) for source, target in proposals.items()]

        issues: list[Operation] = []
        memory_available = self.memory_job is None and self.memory_ready(self.cycle)
        for pe in range(self.hw.pes):
            contexts = sorted(
                (c for c in self.contexts.values() if c.block.pe == pe and not c.retired),
                key=lambda c: (c.block.layer, c.block.block_id),
            )
            active = [c for c in contexts if c.inflight]
            self.counters["max_resident_contexts_per_pe"] = max(
                self.counters["max_resident_contexts_per_pe"], len(contexts)
            )
            ready_count = sum(not c.inflight and set(c.inst.src) <= c.valid for c in contexts)
            self.counters["ready_context_cycles"] += ready_count
            for pipeline in ("load", "store", "compute", "xfer"):
                self.counters[f"busy_{pipeline}_pe_cycles"] += any(
                    c.inst.pipeline == pipeline for c in active
                )
            self.counters["resident_context_cycles"] += len(contexts)
            self.counters["inflight_context_cycles"] += len(active)
            if len(active) >= 2:
                self.counters["overlap_pe_cycles"] += 1
            issued = False
            for ctx in contexts:
                if ctx.inflight:
                    continue
                inst = ctx.inst
                reason = ""
                if not set(inst.src) <= ctx.valid:
                    reason = "dependency"
                elif not self.timing.overlap and active:
                    reason = "serial_policy"
                elif inst.pipeline == "compute" and (
                    pe in self.compute or self.cycle < self.next_compute[pe]
                ):
                    reason = "compute_capacity_or_ii"
                elif inst.op in ("load", "store") and not memory_available:
                    reason = "spm"
                elif inst.op == "xfer" and (pe in self.routers or pe in reserved_router):
                    reason = "network"
                elif inst.op == "xfer" and (
                    self.contexts[inst.target].retired
                    or self.contexts[inst.target].iteration != ctx.iteration
                    or inst.dst in self.contexts[inst.target].valid
                ):
                    reason = "destination_credit"
                elif issued:
                    reason = "issue_bandwidth"
                if reason:
                    self.counters[f"stall_{reason}"] += 1
                    if self.last_wait.get(ctx.block.block_id) != reason:
                        self.emit("wait_event", ctx, reason=reason)
                    self.last_wait[ctx.block.block_id] = reason
                    continue
                if ctx.block.block_id in self.last_wait:
                    self.emit("wake", ctx, reason=self.last_wait.pop(ctx.block.block_id))
                operands = [self.rf[pe][ctx.base + r] for r in inst.src]
                address = inst.spm + ctx.iteration * inst.stride
                data = ()
                if inst.pipeline == "compute":
                    data = arithmetic(inst.op, operands)
                elif inst.op == "load":
                    data = self.memory[address]
                else:
                    data = operands[0]
                operation = Operation(
                    ctx,
                    ctx.uid,
                    ctx.iteration,
                    ctx.pc,
                    inst,
                    data,
                    self.cycle + self.timing.latencies.get(inst.op, 1),
                    address,
                )
                issues.append(operation)
                if inst.op in ("load", "store"):
                    memory_available = False
                issued = True

        # Commit. Responses and payload data belong to their captured contexts.
        for operation in completions:
            ctx, inst = operation.owner, operation.inst
            if inst.op == "store":
                self.memory[operation.address] = operation.data
            else:
                self.rf[ctx.block.pe][ctx.base + inst.dst] = operation.data
                ctx.valid.add(inst.dst)
                self.emit("rf_write", ctx, register=inst.dst, physical_register=ctx.base + inst.dst)
            if inst.op in ("load", "store"):
                self.emit("memory_response", ctx, address=operation.address)
                self.memory_job = None
            else:
                del self.compute[ctx.block.pe]
            self.complete(operation)
        for pe, packet, target in deliveries:
            self.rf[pe][target.base + packet.register] = packet.operation.data
            target.valid.add(packet.register)
            self.emit(
                "rf_write",
                target,
                register=packet.register,
                physical_register=target.base + packet.register,
                producer_uid=packet.operation.uid,
            )
            self.emit(
                "xfer_receive",
                target,
                producer_uid=packet.operation.uid,
                register=packet.register,
                event_id=f"{self.wave}:{packet.target}:"
                f"{packet.operation.iteration}:{packet.register}",
            )
            self.complete(packet.operation)
            del self.routers[pe]
            self.counters["xfer_delivered"] += 1
        for source, target, packet in moves:
            del self.routers[source]
        for source, target, packet in moves:
            self.routers[target] = packet
            self.emit("route", packet.operation.owner, source_pe=source, target_pe=target)
            self.counters["routed_links"] += 1
        for operation in issues:
            ctx, inst = operation.owner, operation.inst
            if ctx.inflight or ctx.uid != operation.uid:
                raise RuntimeError("issue state changed across edge planning")
            ctx.inflight = True
            self.emit("issue", ctx)
            self.counters["issue"] += 1
            self.counters[f"issue_{inst.pipeline}"] += 1
            if inst.pipeline == "compute":
                self.compute[ctx.block.pe] = operation
                self.next_compute[ctx.block.pe] = self.cycle + self.timing.compute_ii
            elif inst.op in ("load", "store"):
                self.memory_job = operation
                self.emit("memory_request", ctx, address=operation.address)
            else:
                target = self.contexts[inst.target]
                self.routers[ctx.block.pe] = Packet(
                    operation, inst.target, inst.dst, target.block.pe
                )
                self.emit("xfer_send", ctx, target_block=inst.target, register=inst.dst)
                self.counters["xfer_sent"] += 1

        if all(c.retired for c in self.contexts.values()):
            if self.compute or self.memory_job is not None or self.routers:
                raise RuntimeError("retired wave still has outstanding work")
            self.contexts = {}
            self.finished = self.wave == max(b.wave for b in self.program.blocks)
        self.cycle += 1

    def run(self) -> Result:
        while not self.finished and self.cycle < self.timing.max_cycles:
            self.step()
        if not self.finished:
            pending = [
                (c.block.block_id, c.pc, c.iteration, c.inflight)
                for c in self.contexts.values()
                if not c.retired
            ]
            raise RuntimeError(f"cycle budget exhausted; pending contexts: {pending}")
        for left, right in (
            ("issue", "complete"),
            ("admitted", "retired"),
            ("registers_reserved", "registers_released"),
            ("xfer_sent", "xfer_delivered"),
        ):
            if self.counters[left] != self.counters[right]:
                raise RuntimeError(f"conservation failure: {left} != {right}")
        return Result(
            {address: self.memory[address] for address in self.program.outputs},
            self.cycle,
            self.events,
            dict(self.counters),
            self.program.digest(),
        )
