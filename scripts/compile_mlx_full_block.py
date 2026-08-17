#!/usr/bin/env python3
"""Compile H48's fixed and real-DMA full MLX block proxy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from mlxsim.dsagen_dma import read_elf_symbols
from mlxsim.dsagen_full_block import compile_full_block
from mlxsim.dsagen_overlay import canonical_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ELF = (
    PROJECT_ROOT / "third_party/dsa-framework/dsa-apps/sdk/compiled/ss-mlx-dma.out"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/environment/h48"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elf", type=Path, default=DEFAULT_ELF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--replay-check", type=Path)
    return parser.parse_args()


def qualify(path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "metadata": metadata,
    }


def main() -> int:
    args = parse_args()
    elf = args.elf.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    symbols = read_elf_symbols(elf)
    outputs: dict[str, Any] = {}
    for name, backend in (("fixed", "fixed"), ("dma", "dsagen_dma")):
        document, metadata = compile_full_block(symbols, memory_backend=backend)
        path = output_dir / f"mlx-full-block-{name}.json"
        path.write_text(canonical_json(document), encoding="utf-8")
        outputs[name] = qualify(path, metadata)
    elf_data = elf.read_bytes()
    manifest = {
        "schema_version": 1,
        "experiment_id": "H48",
        "paper_performance_targets_consumed": False,
        "guest_elf": {
            "path": str(elf),
            "bytes": len(elf_data),
            "sha256": hashlib.sha256(elf_data).hexdigest(),
            "symbols": {
                name: {"address": symbol.address, "size": symbol.size}
                for name, symbol in sorted(symbols.items())
            },
        },
        "outputs": outputs,
    }
    manifest_path = output_dir / "mlx-full-block-compile-manifest.json"
    manifest_path.write_text(canonical_json(manifest), encoding="utf-8")
    if args.reference_dir is not None:
        reference_dir = args.reference_dir.resolve()
        comparisons: dict[str, Any] = {}
        for name in ("fixed", "dma"):
            filename = f"mlx-full-block-{name}.json"
            reference = reference_dir / filename
            replay = output_dir / filename
            comparisons[name] = {
                "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
                "replay_sha256": hashlib.sha256(replay.read_bytes()).hexdigest(),
                "reference_bytes": reference.stat().st_size,
                "replay_bytes": replay.stat().st_size,
            }
            comparisons[name]["identical"] = (
                comparisons[name]["reference_sha256"]
                == comparisons[name]["replay_sha256"]
                and comparisons[name]["reference_bytes"] == comparisons[name]["replay_bytes"]
            )
        replay_report = {
            "schema_version": 1,
            "experiment_id": "H48",
            "comparisons": comparisons,
            "all_identical": all(item["identical"] for item in comparisons.values()),
        }
        replay_path = (
            args.replay_check.resolve()
            if args.replay_check is not None
            else output_dir / "compiler-replay-check.json"
        )
        replay_path.parent.mkdir(parents=True, exist_ok=True)
        replay_path.write_text(canonical_json(replay_report), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
