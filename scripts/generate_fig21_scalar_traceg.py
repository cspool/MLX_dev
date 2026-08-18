#!/usr/bin/env python3
"""Generate deterministic H147 SP/SFU/ALU scalar-service traceg files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig21_xavier_scalar_traceg_v1.yaml"


def digest(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def render_trace(*, repeats: int, ctas: int, service: str, spec: dict[str, Any]) -> str:
    opcode = spec["opcode"]
    destination = spec["destination"]
    sources = spec["sources"]
    source_text = " ".join(sources)
    lines = [
        f"-kernel name = mlx_source_derived_{service}",
        "-kernel id = 1",
        f"-grid dim = ({ctas},1,1)",
        "-block dim = (32,1,1)",
        "-shmem = 0",
        "-nregs = 4",
        "-binary version = 70",
        "-cuda stream id = 0",
        "-shmem base_addr = 0x0000000100000000",
        "-local mem base_addr = 0x0000000200000000",
        "-nvbit version = source-derived",
        "-accelsim tracer version = 3",
        "",
        "#traces format = threadblock_x threadblock_y threadblock_z warpid_tb PC mask dest_num [reg_dests] opcode src_num [reg_srcs] mem_width [addresses]",
        "",
    ]
    for cta in range(ctas):
        lines.extend(
            [
                "#BEGIN_TB",
                "",
                f"thread block = {cta},0,0",
                "",
                "warp = 0",
                f"insts = {repeats + 2}",
                "0000 ffffffff 1 R1 MOV 0 0",
            ]
        )
        for repeat in range(repeats):
            pc = (repeat + 1) * 16
            lines.append(
                f"{pc:04x} ffffffff 1 {destination} {opcode} {len(sources)} {source_text} 0"
            )
        exit_pc = (repeats + 1) * 16
        lines.extend(
            [
                f"{exit_pc:04x} ffffffff 0 EXIT 0 0",
                "",
                "#END_TB",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output_root = PROJECT_ROOT / config["output_root"]
    ctas = int(config["trace_contract"]["grid"][0])
    scalar_lanes = int(config["trace_contract"]["scalar_operations_per_warp_instruction"])
    repeats_values = [
        *config["trace_contract"]["fit_repeats"],
        *config["trace_contract"]["holdout_repeats"],
    ]
    outputs = {}
    for service, spec in config["trace_contract"]["service_classes"].items():
        for repeats_value in repeats_values:
            repeats = int(repeats_value)
            text = render_trace(repeats=repeats, ctas=ctas, service=service, spec=spec)
            replay_text = render_trace(repeats=repeats, ctas=ctas, service=service, spec=spec)
            key = f"{service}-r{repeats}"
            primary_dir = output_root / "traces" / key
            replay_dir = output_root / "replay" / key
            primary_dir.mkdir(parents=True, exist_ok=True)
            replay_dir.mkdir(parents=True, exist_ok=True)
            primary_trace = primary_dir / "kernel-1.traceg"
            replay_trace = replay_dir / "kernel-1.traceg"
            primary_list = primary_dir / "kernelslist.g"
            replay_list = replay_dir / "kernelslist.g"
            primary_trace.write_text(text)
            replay_trace.write_text(replay_text)
            primary_list.write_text("kernel-1.traceg\n")
            replay_list.write_text("kernel-1.traceg\n")
            primary = digest(primary_trace)
            replay = digest(replay_trace)
            checks = {
                "begin_tb": text.count("#BEGIN_TB") == ctas,
                "end_tb": text.count("#END_TB") == ctas,
                "service_opcode": text.count(f" {spec['opcode']} ") == ctas * repeats,
                "mov": text.count(" MOV ") == ctas,
                "exit": text.count(" EXIT ") == ctas,
                "no_memory": all(token not in text for token in (" LD", " ST", "ATOM", "RED")),
                "deterministic": primary["sha256"] == replay["sha256"],
            }
            outputs[key] = {
                "service": service,
                "opcode": spec["opcode"],
                "repeats": repeats,
                "ctas": ctas,
                "warp_instructions": ctas * repeats,
                "scalar_operations": ctas * repeats * scalar_lanes,
                "primary_trace": primary,
                "primary_list": digest(primary_list),
                "replay_trace": replay,
                "replay_list": digest(replay_list),
                "checks": checks,
                "pass": all(checks.values()),
            }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "paper_performance_targets_consumed": False,
        "trace_identity": config["trace_contract"]["identity"],
        "outputs": outputs,
    }
    path = output_root / "scalar-traceg-compile-manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"trace_count": len(outputs), "outputs": outputs}, indent=2))
    return 0 if len(outputs) == 12 and all(item["pass"] for item in outputs.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
