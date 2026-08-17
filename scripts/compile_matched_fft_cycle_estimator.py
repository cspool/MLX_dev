#!/usr/bin/env python3
"""Compile H80 variable-depth FFT-CMP timing fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from mlxsim.dsagen_dma import read_elf_symbols
from mlxsim.dsagen_matched_fft import compile_matched_fft
from mlxsim.dsagen_overlay import canonical_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/matched_fft_cycle_estimator_v1.yaml"
DEFAULT_ELF = PROJECT_ROOT / "third_party/dsa-framework/dsa-apps/sdk/compiled/ss-mlx-dma.out"


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
    parser.add_argument("--elf", type=Path, default=DEFAULT_ELF)
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
    symbols = read_elf_symbols(args.elf.resolve())
    scales = [*config["fit_scales"], *config["holdout_scales"]]
    outputs = {}
    for shape_name, shape in config["shapes"].items():
        for scale_value in scales:
            scale = int(scale_value)
            key = f"{shape_name}-q{scale}"
            document, metadata = compile_matched_fft(
                name=key,
                forward_stages=int(shape["forward_stages"]),
                inverse_stages=int(shape["inverse_stages"]),
                scale=scale,
                symbols=symbols,
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
    path = output_root / "matched-fft-compile-manifest.json"
    path.write_text(canonical_json(manifest), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
