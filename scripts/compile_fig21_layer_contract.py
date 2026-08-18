#!/usr/bin/env python3
"""Materialize H91 Figure 21 u=1 Attention graphs and layer contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from mlxsim.dsagen_overlay import canonical_json
from mlxsim.fig21_layer_contract import build_shape_contract

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig21_layer_contract_v1.yaml"


def digest(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_root = (
        args.output_dir.resolve()
        if args.output_dir
        else (PROJECT_ROOT / config["output_root"]).resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)
    identity = json.loads(
        (
            PROJECT_ROOT / config["frozen_inputs"]["identity"]["path"]
        ).read_text(encoding="utf-8")
    )
    shape = config["shape"]
    outputs = {}
    contracts = {}
    for n_value in shape["sequence_lengths"]:
        n = int(n_value)
        document, contract = build_shape_contract(
            sequence_length=n,
            batch=int(shape["batch"]),
            hidden_dimension=int(shape["hidden_dimension"]),
            ffn_dimension=int(shape["ffn_dimension"]),
            simd_width=int(shape["simd_width"]),
            vector_bytes=int(shape["vector_bytes"]),
            active_window=int(shape["active_window"]),
            logical_profile=identity["logical_profiles"][f"N{n}"],
        )
        path = output_root / f"fig21-N{n}-u1.json"
        path.write_text(canonical_json(document), encoding="utf-8")
        outputs[f"N{n}"] = digest(path)
        contracts[f"N{n}"] = contract
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "paper_performance_targets_consumed": False,
        "outputs": outputs,
        "contracts": contracts,
    }
    path = output_root / "fig21-layer-contract-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if len(outputs) == 5 and all(
        all(contract["checks"].values()) for contract in contracts.values()
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
