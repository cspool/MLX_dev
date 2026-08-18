#!/usr/bin/env python3
"""Compile H93 scaled batch-8 Attention timing configs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from mlxsim.dsagen_combined_attention import compile_combined_attention
from mlxsim.dsagen_overlay import canonical_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig21_attention_timing_v1.yaml"


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
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    parent = json.loads(
        (
            PROJECT_ROOT / config["frozen_inputs"]["layer_contract"]["path"]
        ).read_text(encoding="utf-8")
    )
    output_root = PROJECT_ROOT / config["output_root"]
    config_root = output_root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    scales = [*config["fit_scales"], *config["holdout_scales"]]
    outputs = {}
    for shape_name, contract in parent["contracts"].items():
        n = int(contract["sequence_length"])
        retained = int(contract["retained_length"])
        for scale_value in scales:
            scale = int(scale_value)
            key = f"{shape_name}-u{scale}"
            document, metadata = compile_combined_attention(
                name=key,
                sequence_length=n,
                retained_length=retained,
                hidden_dimension=4096,
                forward_stages=int(math.log2(n)),
                inverse_stages=int(math.log2(retained)),
                fft_scale=int(contract["fft_scale_per_u"]) * scale,
                attention_scale=scale,
                vector_bytes=64,
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
    path = output_root / "fig21-attention-compile-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_count": len(outputs)}, indent=2))
    return 0 if len(outputs) == 20 else 1


if __name__ == "__main__":
    raise SystemExit(main())
