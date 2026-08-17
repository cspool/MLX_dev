#!/usr/bin/env python3
"""Compile H49's 24 target-independent real-DMA operator proxies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from mlxsim.dsagen_dma import read_elf_symbols
from mlxsim.dsagen_operator_sweep import compile_operator_proxy
from mlxsim.dsagen_overlay import canonical_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/dsagen_mlx_fig25_transfer_v1.yaml"
DEFAULT_ELF = (
    PROJECT_ROOT / "third_party/dsa-framework/dsa-apps/sdk/compiled/ss-mlx-dma.out"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/environment/h49"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--elf", type=Path, default=DEFAULT_ELF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--replay-check", type=Path)
    return parser.parse_args()


def digest(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    symbols = read_elf_symbols(args.elf.resolve())
    outputs: list[dict[str, Any]] = []
    for operator in config["operators"]:
        for case in config["cases"]:
            document, metadata = compile_operator_proxy(operator, case, symbols)
            filename = f"{operator['name']}--{case['name']}.json"
            path = output_dir / filename
            path.write_text(canonical_json(document), encoding="utf-8")
            outputs.append({**digest(path), "operator": operator["name"], "case": case["name"], "metadata": metadata})
    manifest = {
        "schema_version": 1,
        "experiment_id": "H49",
        "paper_target_values_consumed": False,
        "output_count": len(outputs),
        "outputs": outputs,
    }
    (output_dir / "fig25-transfer-compile-manifest.json").write_text(
        canonical_json(manifest), encoding="utf-8"
    )
    if args.reference_dir is not None:
        comparisons = []
        reference_dir = args.reference_dir.resolve()
        for item in outputs:
            replay = Path(item["path"])
            reference = reference_dir / replay.name
            comparisons.append(
                {
                    "name": replay.name,
                    "reference": digest(reference),
                    "replay": digest(replay),
                    "identical": reference.read_bytes() == replay.read_bytes(),
                }
            )
        report = {
            "schema_version": 1,
            "experiment_id": "H49",
            "comparisons": comparisons,
            "all_identical": all(item["identical"] for item in comparisons),
        }
        replay_path = args.replay_check.resolve() if args.replay_check else output_dir / "replay-check.json"
        replay_path.parent.mkdir(parents=True, exist_ok=True)
        replay_path.write_text(canonical_json(report), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
