#!/usr/bin/env python3
"""Compile H82 grouped compressed-attention CDC fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from mlxsim.dsagen_grouped_attention import compile_grouped_attention
from mlxsim.dsagen_overlay import canonical_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/grouped_attention_cdc_v1.yaml"


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
    config_root = output_root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    scales = [*config["fit_scales"], *config["holdout_scales"]]
    outputs = {}
    for shape_name, shape in config["shapes"].items():
        for scale_value in scales:
            scale = int(scale_value)
            key = f"{shape_name}-q{scale}"
            document, metadata = compile_grouped_attention(
                name=key,
                retained_length=int(shape["retained_length"]),
                hidden_dimension=int(config["hardware"]["hidden_dimension"]),
                scale=scale,
            )
            path = config_root / f"{key}.json"
            path.write_text(canonical_json(document), encoding="utf-8")
            outputs[key] = {"artifact": digest(path), "metadata": metadata}
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "paper_performance_targets_consumed": False,
        "outputs": outputs,
    }
    path = output_root / "grouped-attention-compile-manifest.json"
    path.write_text(canonical_json(manifest), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
