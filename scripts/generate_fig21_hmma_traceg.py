#!/usr/bin/env python3
"""Generate deterministic H146 compute-only HMMA traceg microtraces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig21_xavier_hmma_traceg_v1.yaml"


def digest(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def render_trace(*, repeats: int, ctas: int) -> str:
    if repeats <= 0 or ctas <= 0:
        raise ValueError("repeats and CTAs must be positive")
    lines = [
        "-kernel name = mlx_source_derived_hmma",
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
            lines.append(f"{pc:04x} ffffffff 1 R3 HMMA 3 R1 R2 R3 0")
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


def trace_checks(text: str, *, repeats: int, ctas: int) -> dict[str, bool]:
    return {
        "begin_tb": text.count("#BEGIN_TB") == ctas,
        "end_tb": text.count("#END_TB") == ctas,
        "warps": text.count("warp = 0") == ctas,
        "hmma": text.count(" HMMA ") == ctas * repeats,
        "mov": text.count(" MOV ") == ctas,
        "exit": text.count(" EXIT ") == ctas,
        "no_memory": all(token not in text for token in (" LD", " ST", "ATOM", "RED")),
        "binary70": "-binary version = 70" in text,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output_root = PROJECT_ROOT / config["output_root"]
    ctas = int(config["trace_contract"]["grid"][0])
    repeats_values = [
        *config["trace_contract"]["fit_repeats"],
        *config["trace_contract"]["holdout_repeats"],
    ]
    outputs = {}
    for repeats_value in repeats_values:
        repeats = int(repeats_value)
        text = render_trace(repeats=repeats, ctas=ctas)
        replay_text = render_trace(repeats=repeats, ctas=ctas)
        primary_dir = output_root / "traces" / f"r{repeats}"
        replay_dir = output_root / "replay" / f"r{repeats}"
        primary_dir.mkdir(parents=True, exist_ok=True)
        replay_dir.mkdir(parents=True, exist_ok=True)
        trace_name = "kernel-1.traceg"
        primary_trace = primary_dir / trace_name
        replay_trace = replay_dir / trace_name
        primary_list = primary_dir / "kernelslist.g"
        replay_list = replay_dir / "kernelslist.g"
        primary_trace.write_text(text)
        replay_trace.write_text(replay_text)
        primary_list.write_text(trace_name + "\n")
        replay_list.write_text(trace_name + "\n")
        primary = digest(primary_trace)
        replay = digest(replay_trace)
        checks = trace_checks(text, repeats=repeats, ctas=ctas)
        checks["deterministic"] = primary["sha256"] == replay["sha256"]
        outputs[f"r{repeats}"] = {
            "repeats": repeats,
            "ctas": ctas,
            "hmma_instructions": ctas * repeats,
            "fma_equivalents": ctas
            * repeats
            * int(config["trace_contract"]["fma_equivalents_per_hmma"]),
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
    path = output_root / "hmma-traceg-compile-manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"outputs": outputs}, indent=2))
    return 0 if len(outputs) == 4 and all(item["pass"] for item in outputs.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
