"""Versioned program contract for MLX's bounded, multi-context execution backend.

Logical blocks are mapped to PEs by the compiler. A wave reserves all of its
contexts together; wave boundaries drain messages before physical slots are
reused. Within a wave, incoming vectors (not whole-layer barriers) enable work.
The old per-PE instruction encoding remains available in the v1 toolchain.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

ABI_VERSION = 2
ABI_MAGIC = 0x4D4C5802
IMAGE_MAGIC = 0x4D4C580200000001
DESCRIPTOR_BASE = 0x100
CODE_BASE = 0x1000
MAX_BLOCKS = 256
OPCODES = {
    "load": 0,
    "store": 1,
    "fma": 2,
    "add": 3,
    "max": 4,
    "exp": 5,
    "div": 6,
    "shuffle": 7,
    "xfer": 8,
    "mul": 9,
}
ARITY = {
    "load": 0,
    "store": 1,
    "fma": 3,
    "add": 2,
    "max": 2,
    "exp": 1,
    "div": 2,
    "shuffle": 1,
    "xfer": 1,
    "mul": 2,
}
NUMERICS = "fp16_step_rounding_nonfused_fma_transcendental_quarter_lanes_v1"


class ProgramError(ValueError):
    """A program cannot be executed under its declared contract."""


@dataclass(frozen=True)
class Hardware:
    rows: int = 4
    columns: int = 4
    lanes: int = 32
    contexts: int = 4
    registers: int = 16
    instruction_words: int = 32
    spm_vectors: int = 128

    @property
    def pes(self) -> int:
        return self.rows * self.columns

    def validate(self) -> None:
        bounds = {
            "rows": (1, 4),
            "columns": (1, 4),
            "lanes": (4, 32),
            "contexts": (1, 4),
            "registers": (1, 16),
            "instruction_words": (1, 32),
            "spm_vectors": (1, 128),
        }
        for name, (low, high) in bounds.items():
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ProgramError(f"hardware {name} outside [{low}, {high}]")
        if self.lanes & (self.lanes - 1):
            raise ProgramError("SIMD lanes must be a power of two")


@dataclass(frozen=True)
class Instruction:
    op: str
    dst: int = 0
    src: tuple[int, ...] = ()
    spm: int = 0
    stride: int = 0
    target: int | None = None

    @property
    def pipeline(self) -> str:
        return self.op if self.op in ("load", "store", "xfer") else "compute"

    def encode(
        self, block: Block, descriptors: dict[int, int], blocks: dict[int, Block], hw: Hardware
    ) -> int:
        source = (*self.src, 0, 0, 0)
        pipeline = {"load": 0, "store": 1, "compute": 2, "xfer": 3}[self.pipeline]
        word = (
            OPCODES[self.op] << 60
            | pipeline << 54
            | self.dst << 50
            | source[0] << 46
            | source[1] << 42
            | source[2] << 38
        )
        if self.op == "xfer":
            target = blocks[self.target]
            dx = target.pe % hw.columns - block.pe % hw.columns
            dy = target.pe // hw.columns - block.pe // hw.columns
            word |= (dx & 31) << 33 | (dy & 31) << 28 | descriptors[self.target] << 4
        elif self.op in ("load", "store"):
            word |= self.spm << 20 | (self.stride & 255) << 12
        return word


@dataclass(frozen=True)
class Block:
    block_id: int
    layer: int
    pe: int
    wave: int
    registers: int
    instructions: tuple[Instruction, ...]
    trip_count: int = 1


@dataclass(frozen=True)
class Program:
    name: str
    blocks: tuple[Block, ...]
    inputs: dict[int, tuple[int, ...]]
    outputs: tuple[int, ...]
    hardware: Hardware = field(default_factory=Hardware)
    abi_version: int = ABI_VERSION
    numerics: str = NUMERICS
    lineage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Program:
        try:
            data = dict(value)
            data["hardware"] = Hardware(**data["hardware"])
            data["blocks"] = tuple(
                Block(
                    **{
                        **block,
                        "instructions": tuple(
                            Instruction(**{**inst, "src": tuple(inst.get("src", ()))})
                            for inst in block["instructions"]
                        ),
                    }
                )
                for block in data["blocks"]
            )
            data["inputs"] = {int(k): tuple(v) for k, v in data["inputs"].items()}
            data["outputs"] = tuple(data["outputs"])
            result = cls(**data)
        except (KeyError, TypeError, ValueError) as error:
            raise ProgramError(f"invalid program schema: {error}") from error
        result.validate()
        return result

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def validate(self) -> None:
        hw = self.hardware
        hw.validate()
        if self.abi_version != ABI_VERSION or self.numerics != NUMERICS:
            raise ProgramError("unsupported program ABI or numerical contract")
        if not isinstance(self.name, str) or not self.name:
            raise ProgramError("program name is required")
        if not 1 <= len(self.blocks) <= MAX_BLOCKS:
            raise ProgramError(f"program must contain 1..{MAX_BLOCKS} blocks")
        ids = [block.block_id for block in self.blocks]
        if len(set(ids)) != len(ids):
            raise ProgramError("duplicate logical block ID")
        by_id = {block.block_id: block for block in self.blocks}
        waves = sorted({block.wave for block in self.blocks})
        if waves != list(range(len(waves))):
            raise ProgramError("waves must be contiguous from zero")
        for address, vector in self.inputs.items():
            if type(address) is not int or not 0 <= address < hw.spm_vectors:
                raise ProgramError("input SPM address outside capacity")
            if len(vector) != hw.lanes or any(
                type(v) is not int or not 0 <= v <= 65535 for v in vector
            ):
                raise ProgramError("input vector must contain SIMD-width FP16 bit patterns")
        if not self.outputs or len(set(self.outputs)) != len(self.outputs):
            raise ProgramError("outputs must be nonempty and unique")
        if any(type(a) is not int or not 0 <= a < hw.spm_vectors for a in self.outputs):
            raise ProgramError("output SPM address outside capacity")

        incoming: dict[int, set[int]] = {bid: set() for bid in ids}
        successors: dict[int, set[int]] = {bid: set() for bid in ids}
        for block in self.blocks:
            for name, limit in (
                ("block_id", 65536),
                ("layer", 65536),
                ("pe", hw.pes),
                ("wave", MAX_BLOCKS),
            ):
                value = getattr(block, name)
                if type(value) is not int or not 0 <= value < limit:
                    raise ProgramError(f"block {block.block_id}: invalid {name}")
            if (
                type(block.registers) is not int
                or not 1 <= block.registers <= hw.registers
                or type(block.trip_count) is not int
                or not 1 <= block.trip_count <= 65535
            ):
                raise ProgramError(f"block {block.block_id}: invalid registers/trip count")
            if not 1 <= len(block.instructions) <= hw.instruction_words:
                raise ProgramError(f"block {block.block_id}: program capacity exceeded")
            for inst in block.instructions:
                if inst.op not in OPCODES or len(inst.src) != ARITY[inst.op]:
                    raise ProgramError(f"block {block.block_id}: unsupported op/arity {inst.op}")
                if any(type(r) is not int or not 0 <= r < block.registers for r in inst.src):
                    raise ProgramError(f"block {block.block_id}: invalid source register")
                if type(inst.dst) is not int or not 0 <= inst.dst < 16:
                    raise ProgramError("invalid destination register")
                if inst.op not in ("store", "xfer") and inst.dst >= block.registers:
                    raise ProgramError("destination exceeds block register allocation")
                if inst.op == "xfer":
                    target = by_id.get(inst.target)
                    if target is None or target.wave != block.wave:
                        raise ProgramError("xfer must target a block in the same admitted wave")
                    if target.trip_count != block.trip_count or inst.dst >= target.registers:
                        raise ProgramError(
                            "xfer iteration count or destination allocation mismatch"
                        )
                    if inst.dst in incoming[target.block_id]:
                        raise ProgramError("multiple producers for an incoming register")
                    incoming[target.block_id].add(inst.dst)
                    successors[block.block_id].add(target.block_id)
                elif inst.target is not None:
                    raise ProgramError("only xfer instructions have a target block")
                if inst.op in ("load", "store"):
                    if type(inst.stride) is not int or not -128 <= inst.stride <= 127:
                        raise ProgramError("SPM iteration stride outside signed eight-bit range")
                    last = inst.spm + inst.stride * (block.trip_count - 1)
                    if (
                        type(inst.spm) is not int
                        or not 0 <= min(inst.spm, last) <= max(inst.spm, last) < hw.spm_vectors
                    ):
                        raise ProgramError("SPM access exceeds capacity over loop iterations")
                elif inst.spm != 0 or inst.stride != 0:
                    raise ProgramError("SPM fields are valid only on load/store")

        # Static single-writer input mailboxes retain their data until the next
        # iteration. Senders cannot overwrite a still-valid incoming register.
        for block in self.blocks:
            defined = set(incoming[block.block_id])
            consumed: set[int] = set()
            for inst in block.instructions:
                if not set(inst.src) <= defined:
                    raise ProgramError(f"block {block.block_id}: read before definition")
                consumed.update(inst.src)
                if inst.op not in ("store", "xfer"):
                    if inst.dst in incoming[block.block_id]:
                        raise ProgramError("local writes cannot overwrite an incoming mailbox")
                    defined.add(inst.dst)
            if not incoming[block.block_id] <= consumed:
                raise ProgramError("incoming mailbox is never consumed")
        remaining = set(ids)
        while remaining:
            ready = {
                bid
                for bid in remaining
                if not any(bid in successors[parent] for parent in remaining)
            }
            if not ready:
                raise ProgramError("cyclic inter-block dataflow is outside the v2 contract")
            remaining -= ready

        valid_spm = set(self.inputs)
        for wave in waves:
            for pe in range(hw.pes):
                group = [b for b in self.blocks if b.wave == wave and b.pe == pe]
                if len(group) > hw.contexts or sum(b.registers for b in group) > hw.registers:
                    raise ProgramError(
                        f"wave {wave}, PE {pe}: context/RF admission capacity exceeded"
                    )
            accesses: dict[int, list[tuple[int, str]]] = {}
            for block in (b for b in self.blocks if b.wave == wave):
                local_valid = set(valid_spm)
                for iteration in range(block.trip_count):
                    for inst in block.instructions:
                        if inst.op not in ("load", "store"):
                            continue
                        address = inst.spm + inst.stride * iteration
                        accesses.setdefault(address, []).append((block.block_id, inst.op))
                        if inst.op == "load" and address not in local_valid:
                            raise ProgramError("SPM input is not available before its wave/load")
                        if inst.op == "store":
                            local_valid.add(address)
            for address, users in accesses.items():
                if any(op == "store" for _, op in users):
                    if len({bid for bid, _ in users}) > 1:
                        raise ProgramError(
                            "cross-block SPM read/write race; use xfer or split waves"
                        )
                    valid_spm.add(address)
        if not set(self.outputs) <= valid_spm:
            raise ProgramError("output is never initialized")
        self.layout()

    def layout(self) -> tuple[dict[int, list[int]], dict[int, int]]:
        """Deduplicate immutable templates; slots never own separate ROM copies."""
        by_id = {b.block_id: b for b in self.blocks}
        indices = {b.block_id: i for i, b in enumerate(self.blocks)}
        code: dict[int, list[int]] = {pe: [] for pe in range(self.hardware.pes)}
        templates: dict[tuple[int, tuple[int, ...]], int] = {}
        starts = {}
        for block in self.blocks:
            words = tuple(
                i.encode(block, indices, by_id, self.hardware) for i in block.instructions
            )
            key = (block.pe, words)
            if key not in templates:
                templates[key] = len(code[block.pe])
                code[block.pe].extend(words)
            starts[block.block_id] = templates[key]
        if any(len(words) > self.hardware.instruction_words for words in code.values()):
            raise ProgramError("deduplicated per-PE instruction ROM capacity exceeded")
        return code, starts

    def image(self) -> dict[int, int]:
        """Addressed 64-bit config words consumed by the host loader and RTL.

        Header 0: magic; 1: block count; 2: hardware geometry. Descriptors use
        four words: identity, template/resources, loop count, reserved zero.
        Code uses fixed 32-word PE strides; no implicit unlimited instruction RAM.
        """
        self.validate()
        hw = self.hardware
        code, starts = self.layout()
        image = {
            0: IMAGE_MAGIC,
            1: len(self.blocks),
            2: hw.rows
            | hw.columns << 8
            | hw.lanes << 16
            | hw.contexts << 24
            | hw.registers << 32
            | hw.instruction_words << 40
            | hw.spm_vectors << 48,
        }
        for index, block in enumerate(self.blocks):
            base = DESCRIPTOR_BASE + index * 4
            image[base] = block.block_id | block.layer << 16 | block.pe << 32 | block.wave << 40
            image[base + 1] = (
                starts[block.block_id] | len(block.instructions) << 8 | block.registers << 16
            )
            image[base + 2] = block.trip_count
            image[base + 3] = 0
        for pe, words in code.items():
            image.update({CODE_BASE + pe * 32 + offset: word for offset, word in enumerate(words)})
        return image

    @classmethod
    def from_image(
        cls,
        image: dict[int, int],
        *,
        name: str,
        inputs: dict[int, tuple[int, ...]],
        outputs: tuple[int, ...],
    ) -> Program:
        """Decode the actual config image, rejecting missing/reserved ABI fields."""
        if any(
            type(a) is not int or a < 0 or type(w) is not int or not 0 <= w < 1 << 64
            for a, w in image.items()
        ):
            raise ProgramError("invalid config address/64-bit word")
        try:
            if image[0] != IMAGE_MAGIC or not 1 <= image[1] <= MAX_BLOCKS:
                raise ProgramError("incompatible config image magic/block count")
            geometry = image[2]
            if geometry >> 56:
                raise ProgramError("reserved hardware header bits are nonzero")
            hw = Hardware(
                **dict(
                    zip(
                        (
                            "rows",
                            "columns",
                            "lanes",
                            "contexts",
                            "registers",
                            "instruction_words",
                            "spm_vectors",
                        ),
                        ((geometry >> offset) & 255 for offset in range(0, 56, 8)),
                        strict=True,
                    )
                )
            )
            hw.validate()
            descriptors = []
            for index in range(image[1]):
                address = DESCRIPTOR_BASE + 4 * index
                identity, template, loops, reserved = (image[address + i] for i in range(4))
                if identity >> 48 or template >> 24 or loops >> 16 or reserved:
                    raise ProgramError("reserved descriptor bits are nonzero")
                descriptors.append(
                    {
                        "block_id": identity & 65535,
                        "layer": (identity >> 16) & 65535,
                        "pe": (identity >> 32) & 255,
                        "wave": (identity >> 40) & 255,
                        "registers": (template >> 16) & 255,
                        "trip_count": loops,
                        "start": template & 255,
                        "length": (template >> 8) & 255,
                    }
                )
            blocks = []
            for desc in descriptors:
                instructions = []
                if desc["start"] + desc["length"] > hw.instruction_words:
                    raise ProgramError("descriptor template exceeds instruction ROM")
                for pc in range(desc["start"], desc["start"] + desc["length"]):
                    word = image[CODE_BASE + desc["pe"] * 32 + pc]
                    op = next((op for op, code in OPCODES.items() if code == word >> 60), None)
                    if op is None:
                        raise ProgramError("unsupported encoded opcode")
                    src = tuple((word >> bit) & 15 for bit in (46, 42, 38))[: ARITY[op]]
                    fields: dict[str, Any] = {"op": op, "dst": (word >> 50) & 15, "src": src}
                    if op == "xfer":
                        target_index = (word >> 4) & 65535
                        if target_index >= len(descriptors):
                            raise ProgramError("xfer descriptor index outside image")
                        fields["target"] = descriptors[target_index]["block_id"]
                    elif op in ("load", "store"):
                        stride = (word >> 12) & 255
                        fields.update(
                            spm=(word >> 20) & 255, stride=stride if stride < 128 else stride - 256
                        )
                    instructions.append(Instruction(**fields))
                blocks.append(
                    Block(
                        **{k: v for k, v in desc.items() if k not in ("start", "length")},
                        instructions=tuple(instructions),
                    )
                )
            program = cls(name, tuple(blocks), inputs, outputs, hw)
            program.validate()
            # Canonical re-encoding also validates routing, pipeline, reserved
            # bits and template boundaries instead of ignoring inconsistent data.
            if image != program.image():
                raise ProgramError("noncanonical config image or inconsistent instruction fields")
            return program
        except KeyError as error:
            raise ProgramError(f"missing config word {error}") from error
