#!/usr/bin/env python3
"""Compile MLX system-workload YAML to spatial words, C headers, and goldens."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/system/mlx_riscv_system_v1.yaml"

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
PIPELINES = {
    "load": 0,
    "store": 1,
    "fma": 2,
    "add": 2,
    "max": 2,
    "exp": 2,
    "div": 2,
    "shuffle": 2,
    "mul": 2,
    "xfer": 3,
}
SOURCE_REGISTERS = {
    "load": (),
    "store": ("src0",),
    "fma": ("src0", "src1", "src2"),
    "add": ("src0", "src1"),
    "max": ("src0", "src1"),
    "exp": ("src0",),
    "div": ("src0", "src1"),
    "shuffle": ("src0",),
    "xfer": ("src0",),
    "mul": ("src0", "src1"),
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fp16_bits(values: list[float] | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float16).view(np.uint16)


def bits_as_fp16(values: np.ndarray) -> np.ndarray:
    return values.view(np.float16)


def expand_vector(spec: dict[str, Any], lanes: int) -> np.ndarray:
    if "fill" in spec:
        values = [float(spec["fill"])] * lanes
    elif "pattern" in spec:
        pattern = [float(value) for value in spec["pattern"]]
        if not pattern or lanes % len(pattern):
            raise ValueError(f"pattern must divide {lanes} lanes: {spec}")
        values = (pattern * (lanes // len(pattern)))[:lanes]
    elif "lanes" in spec:
        values = [float(value) for value in spec["lanes"]]
        if len(values) != lanes:
            raise ValueError(f"expected {lanes} lanes, got {len(values)}")
    else:
        raise ValueError(f"vector requires fill, pattern, or lanes: {spec}")
    return fp16_bits(values)


def signed_five(value: int) -> int:
    if value < -16 or value > 15:
        raise ValueError(f"route displacement outside signed five-bit range: {value}")
    return value & 0x1F


def normalize_instruction(
    raw: dict[str, Any], pe: int, slots: dict[str, int]
) -> dict[str, int | str]:
    instruction: dict[str, int | str] = {"op": str(raw["op"])}
    operation = str(instruction["op"])
    if operation not in OPCODES:
        raise ValueError(f"unknown operation {operation}")
    for field in ("dst", "src0", "src1", "src2"):
        instruction[field] = int(raw.get(field, 0))
    spm = raw.get("spm", raw.get("immediate", 0))
    if isinstance(spm, str):
        if spm not in slots:
            raise ValueError(f"unknown SPM vector {spm}")
        spm = slots[spm]
    instruction["immediate"] = int(spm)
    instruction["tag"] = int(raw.get("tag", pe))
    if operation == "xfer":
        target = int(raw["target_pe"])
        if target < 0 or target >= 16:
            raise ValueError(f"invalid target PE {target}")
        instruction["target_pe"] = target
        instruction["dx"] = target % 4 - pe % 4
        instruction["dy"] = target // 4 - pe // 4
    else:
        instruction["dx"] = int(raw.get("dx", 0))
        instruction["dy"] = int(raw.get("dy", 0))
    return instruction


def encode(instruction: dict[str, int | str]) -> int:
    operation = str(instruction["op"])
    fields = {
        "opcode": OPCODES[operation],
        "tag": int(instruction["tag"]),
        "pipeline": PIPELINES[operation],
        "dst": int(instruction["dst"]),
        "src0": int(instruction["src0"]),
        "src1": int(instruction["src1"]),
        "src2": int(instruction["src2"]),
        "dx": signed_five(int(instruction["dx"])),
        "dy": signed_five(int(instruction["dy"])),
        "immediate": int(instruction["immediate"]),
    }
    limits = {
        "opcode": 0xF,
        "tag": 0xF,
        "pipeline": 0x3,
        "dst": 0xF,
        "src0": 0xF,
        "src1": 0xF,
        "src2": 0xF,
        "dx": 0x1F,
        "dy": 0x1F,
        "immediate": 0xFF,
    }
    if any(value < 0 or value > limits[name] for name, value in fields.items()):
        raise ValueError(f"instruction field outside encoding: {instruction}")
    return (
        fields["opcode"] << 60
        | fields["tag"] << 56
        | fields["pipeline"] << 54
        | fields["dst"] << 50
        | fields["src0"] << 46
        | fields["src1"] << 42
        | fields["src2"] << 38
        | fields["dx"] << 33
        | fields["dy"] << 28
        | fields["immediate"] << 20
    )


def execute_arithmetic(
    operation: str, operands: list[np.ndarray], transcendental_lanes: int
) -> np.ndarray:
    floats = [bits_as_fp16(value.copy()) for value in operands]
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        if operation == "fma":
            result = (floats[0] * floats[1]).astype(np.float16)
            result = (result + floats[2]).astype(np.float16)
        elif operation == "add":
            result = (floats[0] + floats[1]).astype(np.float16)
        elif operation == "max":
            result = np.maximum(floats[0], floats[1]).astype(np.float16)
        elif operation == "exp":
            result = np.exp(floats[0].astype(np.float32)).astype(np.float16)
            result[transcendental_lanes:] = floats[0][transcendental_lanes:]
        elif operation == "div":
            result = (floats[0] / floats[1]).astype(np.float16)
            result[transcendental_lanes:] = floats[0][transcendental_lanes:]
        elif operation == "shuffle":
            result = floats[0][np.arange(floats[0].size) ^ 1]
        elif operation == "mul":
            result = (floats[0] * floats[1]).astype(np.float16)
        else:
            raise ValueError(f"not an arithmetic operation: {operation}")
    return np.asarray(result, dtype=np.float16).view(np.uint16)


def reference_execute(
    program: dict[int, list[dict[str, int | str]]],
    initial_spm: dict[int, np.ndarray],
    lanes: int,
    spm_vectors: int,
    transcendental_lanes: int,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    spm = [np.zeros(lanes, dtype=np.uint16) for _ in range(spm_vectors)]
    spm_valid = [False] * spm_vectors
    for index, value in initial_spm.items():
        spm[index] = value.copy()
        spm_valid[index] = True
    registers = [
        [np.zeros(lanes, dtype=np.uint16) for _ in range(16)] for _ in range(16)
    ]
    register_valid = [[False] * 16 for _ in range(16)]
    pcs = [0] * 16
    dynamic_trace: list[dict[str, Any]] = []
    round_index = 0
    while any(pcs[pe] < len(program.get(pe, [])) for pe in range(16)):
        progress = False
        for pe in range(16):
            instructions = program.get(pe, [])
            if pcs[pe] >= len(instructions):
                continue
            instruction = instructions[pcs[pe]]
            operation = str(instruction["op"])
            sources = SOURCE_REGISTERS[operation]
            if not all(register_valid[pe][int(instruction[field])] for field in sources):
                continue
            spm_index = int(instruction["immediate"])
            if operation == "load" and not spm_valid[spm_index]:
                continue
            if operation == "load":
                dst = int(instruction["dst"])
                registers[pe][dst] = spm[spm_index].copy()
                register_valid[pe][dst] = True
            elif operation == "store":
                spm[spm_index] = registers[pe][int(instruction["src0"])].copy()
                spm_valid[spm_index] = True
            elif operation == "xfer":
                target = int(instruction["target_pe"])
                dst = int(instruction["dst"])
                registers[target][dst] = registers[pe][int(instruction["src0"])].copy()
                register_valid[target][dst] = True
            else:
                dst = int(instruction["dst"])
                operands = [registers[pe][int(instruction[field])] for field in sources]
                registers[pe][dst] = execute_arithmetic(
                    operation, operands, transcendental_lanes
                )
                register_valid[pe][dst] = True
            dynamic_trace.append(
                {
                    "round": round_index,
                    "pe": pe,
                    "pc": pcs[pe],
                    "operation": operation,
                }
            )
            pcs[pe] += 1
            progress = True
        if not progress:
            blocked = {
                str(pe): program[pe][pcs[pe]]
                for pe in program
                if pcs[pe] < len(program[pe])
            }
            raise RuntimeError(f"reference program deadlocked: {blocked}")
        round_index += 1
        if round_index > 4096:
            raise RuntimeError("reference program exceeded 4096 scheduling rounds")
    return spm, {"rounds": round_index, "trace": dynamic_trace}


def pack_vector(vector: np.ndarray) -> list[int]:
    if vector.size % 4:
        raise ValueError("FP16 vector must contain a multiple of four lanes")
    beats = []
    for offset in range(0, vector.size, 4):
        value = 0
        for lane in range(4):
            value |= int(vector[offset + lane]) << (16 * lane)
        beats.append(value)
    return beats


def vector_hex(vector: np.ndarray) -> str:
    value = 0
    for lane, bits in enumerate(vector):
        value |= int(bits) << (16 * lane)
    return f"{value:0128x}"


def emit_header(
    path: Path,
    name: str,
    entries: list[dict[str, int]],
    instruction_counts: list[int],
    inputs: list[np.ndarray],
    outputs: list[np.ndarray],
    output_base: int,
) -> None:
    lines = [
        "/* Generated by scripts/build_mlx_system_workloads.py. */",
        "#pragma once",
        f'#define MLX_WORKLOAD_NAME "{name}"',
        f"#define MLX_PROGRAM_ENTRIES {len(entries)}u",
        f"#define MLX_INPUT_VECTORS {len(inputs)}u",
        f"#define MLX_OUTPUT_VECTORS {len(outputs)}u",
        f"#define MLX_OUTPUT_SPM_BASE {output_base}u",
        "",
        "static const mlx_program_entry_t mlx_program[MLX_PROGRAM_ENTRIES] = {",
    ]
    lines.extend(
        f"  {{{entry['pe']}u, {entry['index']}u, UINT64_C(0x{entry['word']:016x})}},"
        for entry in entries
    )
    lines.extend(["};", "", "static const uint8_t mlx_instruction_counts[16] = {"])
    lines.append("  " + ", ".join(f"{count}u" for count in instruction_counts))
    lines.extend(
        [
            "};",
            "",
            "static const uint64_t mlx_input[MLX_INPUT_VECTORS][8] ",
            "    __attribute__((aligned(64))) = {",
        ]
    )
    for vector in inputs:
        beats = pack_vector(vector)
        lines.append("  {" + ", ".join(f"UINT64_C(0x{beat:016x})" for beat in beats) + "},")
    lines.extend(
        [
            "};",
            "",
            "static const uint64_t mlx_golden[MLX_OUTPUT_VECTORS][8] ",
            "    __attribute__((aligned(64))) = {",
        ]
    )
    for vector in outputs:
        beats = pack_vector(vector)
        lines.append("  {" + ", ".join(f"UINT64_C(0x{beat:016x})" for beat in beats) + "},")
    lines.extend(["};", ""])
    path.write_text("\n".join(lines))


def compile_workload(source_path: Path, config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    source = yaml.safe_load(source_path.read_text())
    lanes = int(config["vector_lanes"])
    spm_vectors = int(config["array"]["spm_vectors"])
    inputs: list[np.ndarray] = []
    slots: dict[str, int] = {}
    initial_spm: dict[int, np.ndarray] = {}
    for index, (name, spec) in enumerate(source["inputs"].items()):
        vector = expand_vector(spec, lanes)
        slots[name] = index
        inputs.append(vector)
        initial_spm[index] = vector
    program: dict[int, list[dict[str, int | str]]] = {}
    entries: list[dict[str, int]] = []
    lineage: list[dict[str, Any]] = []
    op_coverage: set[str] = set()
    for pe_text, raw_instructions in source["program"].items():
        pe = int(pe_text)
        if pe < 0 or pe >= 16:
            raise ValueError(f"invalid PE {pe}")
        normalized: list[dict[str, int | str]] = []
        if len(raw_instructions) > int(config["array"]["instructions_per_pe"]):
            raise ValueError(f"PE {pe} program exceeds instruction capacity")
        for index, raw in enumerate(raw_instructions):
            instruction = normalize_instruction(raw, pe, slots)
            word = encode(instruction)
            normalized.append(instruction)
            entries.append({"pe": pe, "index": index, "word": word})
            lineage.append(
                {
                    "pe": pe,
                    "index": index,
                    "word": f"{word:016x}",
                    "source": raw,
                    "normalized": instruction,
                }
            )
            op_coverage.add(str(instruction["op"]))
        program[pe] = normalized
    spm, reference = reference_execute(
        program,
        initial_spm,
        lanes,
        spm_vectors,
        int(config["array"]["transcendental_lanes"]),
    )
    output_slots = [int(value) for value in source["output_slots"]]
    if output_slots != list(range(output_slots[0], output_slots[0] + len(output_slots))):
        raise ValueError("output slots must be contiguous for DMA")
    outputs = [spm[index].copy() for index in output_slots]
    instruction_counts = [len(program.get(pe, [])) for pe in range(16)]

    output_root.mkdir(parents=True, exist_ok=True)
    name = str(source["name"])
    header_path = output_root / f"{name}.h"
    program_path = output_root / f"{name}.program"
    input_path = output_root / f"{name}.input.hex"
    golden_path = output_root / f"{name}.golden.hex"
    reference_path = output_root / f"{name}.reference.json"
    emit_header(
        header_path,
        name,
        entries,
        instruction_counts,
        inputs,
        outputs,
        output_slots[0],
    )
    program_path.write_text(
        "".join(f"{entry['pe']:02x} {entry['index']:02x} {entry['word']:016x}\n" for entry in entries)
    )
    input_path.write_text("".join(f"{vector_hex(vector)}\n" for vector in inputs))
    golden_path.write_text("".join(f"{vector_hex(vector)}\n" for vector in outputs))
    reference_payload = {
        "name": name,
        "operator": source["operator"],
        "input_vectors": [[f"{value:04x}" for value in vector] for vector in inputs],
        "output_slots": output_slots,
        "golden_vectors": [[f"{value:04x}" for value in vector] for vector in outputs],
        "reference_schedule": reference,
    }
    reference_path.write_text(json.dumps(reference_payload, indent=2, sort_keys=True) + "\n")
    route_hops = sum(
        abs(int(item["normalized"]["dx"])) + abs(int(item["normalized"]["dy"]))
        for item in lineage
        if item["normalized"]["op"] == "xfer"
    )
    skip_hops = sum(
        abs(int(item["normalized"]["dx"])) // 2
        + abs(int(item["normalized"]["dy"])) // 2
        for item in lineage
        if item["normalized"]["op"] == "xfer"
    )
    return {
        "name": name,
        "operator": source["operator"],
        "source": str(source_path.relative_to(PROJECT_ROOT)),
        "source_sha256": sha256(source_path.read_bytes()),
        "header": str(header_path.relative_to(PROJECT_ROOT)),
        "header_sha256": sha256(header_path.read_bytes()),
        "program": str(program_path.relative_to(PROJECT_ROOT)),
        "program_sha256": sha256(program_path.read_bytes()),
        "input_hex": str(input_path.relative_to(PROJECT_ROOT)),
        "input_hex_sha256": sha256(input_path.read_bytes()),
        "golden_hex": str(golden_path.relative_to(PROJECT_ROOT)),
        "golden_hex_sha256": sha256(golden_path.read_bytes()),
        "reference": str(reference_path.relative_to(PROJECT_ROOT)),
        "reference_sha256": sha256(reference_path.read_bytes()),
        "input_vectors": len(inputs),
        "output_vectors": len(outputs),
        "output_spm_base": output_slots[0],
        "instruction_count": len(entries),
        "instruction_counts_per_pe": instruction_counts,
        "active_pes": sum(count > 0 for count in instruction_counts),
        "operation_coverage": sorted(op_coverage),
        "route_manhattan_hops": route_hops,
        "route_skip_opportunities": skip_hops,
        "lineage": lineage,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    output_root = PROJECT_ROOT / config["output_root"]
    workloads = [
        compile_workload(PROJECT_ROOT / relative, config, output_root)
        for relative in config["workloads"]
    ]
    all_operations = sorted({op for workload in workloads for op in workload["operation_coverage"]})
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "source_config": str(config_path.relative_to(PROJECT_ROOT)),
        "source_config_sha256": sha256(config_path.read_bytes()),
        "paper_performance_targets_consumed": False,
        "chipyard": config["chipyard"],
        "interface": config["interface"],
        "array": config["array"],
        "encoding": {
            "word_bits": 64,
            "opcode": [63, 60],
            "tag": [59, 56],
            "pipeline": [55, 54],
            "destination": [53, 50],
            "sources": [[49, 46], [45, 42], [41, 38]],
            "dx": [37, 33],
            "dy": [32, 28],
            "spm_vector": [27, 20],
        },
        "opcodes": OPCODES,
        "pipelines": PIPELINES,
        "workloads": workloads,
        "checks": {
            "four_workloads": len(workloads) == 4,
            "required_names": [item["name"] for item in workloads]
            == ["bsmm", "fft_cmp", "swa", "transformer_block"],
            "all_instructions_lineaged": all(
                len(item["lineage"]) == item["instruction_count"] for item in workloads
            ),
            "real_mesh_activity": all(item["active_pes"] >= 6 for item in workloads),
            "all_key_operations": all_operations == sorted(OPCODES),
            "routes_present": all(item["route_manhattan_hops"] > 0 for item in workloads),
            "skip_hops_present": all(item["route_skip_opportunities"] > 0 for item in workloads),
            "capacity": all(max(item["instruction_counts_per_pe"]) <= 32 for item in workloads),
        },
    }
    manifest_path = PROJECT_ROOT / config["manifest"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    summary = {
        "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "workloads": {
            item["name"]: {
                "instructions": item["instruction_count"],
                "active_pes": item["active_pes"],
                "operations": item["operation_coverage"],
            }
            for item in workloads
        },
        "checks": manifest["checks"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if all(manifest["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
