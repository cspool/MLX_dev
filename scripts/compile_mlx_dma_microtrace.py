#!/usr/bin/env python3
"""Compile the frozen H47 fixed/DMA pair from guest ELF symbols."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from mlxsim.dsagen_dma import compile_dma_microtrace, read_elf_symbols
from mlxsim.dsagen_overlay import canonical_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ELF = (
    PROJECT_ROOT / "third_party/dsa-framework/dsa-apps/sdk/compiled/ss-mlx-dma.out"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/environment/h47"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elf", type=Path, default=DEFAULT_ELF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
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
        document, metadata = compile_dma_microtrace(symbols, memory_backend=backend)
        path = output_dir / ("mlx-dma-fixed.json" if name == "fixed" else "mlx-dma-real.json")
        path.write_text(canonical_json(document), encoding="utf-8")
        outputs[name] = qualify(path, metadata)

    elf_data = elf.read_bytes()
    manifest = {
        "schema_version": 1,
        "experiment_id": "H47",
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
    manifest_path = output_dir / "mlx-dma-compile-manifest.json"
    manifest_path.write_text(canonical_json(manifest), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
