#!/usr/bin/env python3
"""Assemble paper-derived MLX RTL workload YAML into 64-bit words."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/rtl/mlx_critical_rtl_v1.yaml"

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


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def encode(instruction: dict[str, Any]) -> int:
    operation = str(instruction["op"])
    fields = {
        "opcode": OPCODES[operation],
        "tag": int(instruction.get("tag", 0)),
        "pipeline": PIPELINES[operation],
        "dst": int(instruction.get("dst", 0)),
        "src0": int(instruction.get("src0", 0)),
        "src1": int(instruction.get("src1", 0)),
        "src2": int(instruction.get("src2", 0)),
        "dx": int(instruction.get("dx", 0)) & 0x1F,
        "dy": int(instruction.get("dy", 0)) & 0x1F,
        "immediate": int(instruction.get("immediate", 0)),
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output = PROJECT_ROOT / config["output_root"] / "programs"
    output.mkdir(parents=True, exist_ok=True)
    programs = []
    for relative in config["workloads"]:
        source_path = PROJECT_ROOT / relative
        source = yaml.safe_load(source_path.read_text())
        lineage = []
        words = []
        for index, instruction in enumerate(source["instructions"]):
            word = encode(instruction)
            words.append(word)
            lineage.append(
                {
                    "index": index,
                    "source_operation": instruction,
                    "encoded_word": f"{word:016x}",
                    "opcode": OPCODES[instruction["op"]],
                    "pipeline": PIPELINES[instruction["op"]],
                }
            )
        payload = "".join(f"{word:016x}\n" for word in words).encode()
        hex_path = output / f"{source['name']}.hex"
        hex_path.write_bytes(payload)
        programs.append(
            {
                "name": source["name"],
                "paper_operator": source["paper_operator"],
                "reduced_eligible": bool(source["reduced_eligible"]),
                "source_path": str(source_path.relative_to(PROJECT_ROOT)),
                "source_sha256": digest_bytes(source_path.read_bytes()),
                "hex_path": str(hex_path.relative_to(PROJECT_ROOT)),
                "hex_bytes": len(payload),
                "hex_sha256": digest_bytes(payload),
                "instruction_count": len(words),
                "lineage": lineage,
            }
        )
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "paper_performance_targets_consumed": False,
        "encoding": {
            "word_bits": 64,
            "opcode": [63, 60],
            "tag": [59, 56],
            "pipeline": [55, 54],
            "destination": [53, 50],
            "sources": [[49, 46], [45, 42], [41, 38]],
            "dx": [37, 33],
            "dy": [32, 28],
            "immediate": [27, 20],
        },
        "opcodes": OPCODES,
        "pipelines": PIPELINES,
        "programs": programs,
        "checks": {
            "programs": len(programs) == 3,
            "nonempty": all(item["instruction_count"] > 0 for item in programs),
            "lineage": all(
                len(item["lineage"]) == item["instruction_count"] for item in programs
            ),
            "unique_names": len({item["name"] for item in programs}) == len(programs),
            "reduced": [item["name"] for item in programs if item["reduced_eligible"]]
            == ["bsmm"],
        },
    }
    manifest_path = PROJECT_ROOT / config["program_manifest"]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"programs": len(programs), "checks": manifest["checks"]}, indent=2))
    return 0 if all(manifest["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
