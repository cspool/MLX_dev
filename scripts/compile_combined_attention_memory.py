#!/usr/bin/env python3
"""Compile H83 combined SIMD32 Attention memory fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from mlxsim.dsagen_combined_attention import compile_combined_attention
from mlxsim.dsagen_overlay import canonical_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/combined_attention_memory_v1.yaml"


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
            key = f"{shape_name}-u{scale}"
            document, metadata = compile_combined_attention(
                name=key,
                sequence_length=int(shape["sequence_length"]),
                retained_length=int(shape["retained_length"]),
                hidden_dimension=int(config["hardware"]["hidden_dimension"]),
                forward_stages=int(shape["fft_forward_stages"]),
                inverse_stages=int(shape["fft_inverse_stages"]),
                fft_scale=scale * int(shape["fft_scale_per_u"]),
                attention_scale=scale * int(shape["attention_scale_per_u"]),
                vector_bytes=int(config["hardware"]["vector_bytes"]),
                active_window=int(config["hardware"]["active_window"]),
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
    path = output_root / "combined-attention-compile-manifest.json"
    path.write_text(canonical_json(manifest), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
