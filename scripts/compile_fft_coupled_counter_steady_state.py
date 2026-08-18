#!/usr/bin/env python3
"""Compile H117 FFT q64/q128 coupled counter configurations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from mlxsim.coupled_full_mesh_paths import compile_coupled_path
from mlxsim.dsagen_overlay import canonical_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/simulators/fft_coupled_counter_steady_state_v1.yaml"
)


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
    h114 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h114"]["path"]).read_text()
    )
    h107 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h107"]["path"]).read_text()
    )
    h110 = json.loads(
        (PROJECT_ROOT / h114["frozen_inputs"]["h110"]["path"]).read_text()
    )
    h110_compile = json.loads(
        (PROJECT_ROOT / h110["compile_manifest"]["path"]).read_text()
    )
    base_config = yaml.safe_load(
        (PROJECT_ROOT / config["frozen_inputs"]["h114_config"]["path"])
        .read_text()
    )
    output_root = PROJECT_ROOT / config["output_root"]
    overlay_root = output_root / "configs/overlay"
    memory_root = output_root / "configs/memory"
    overlay_root.mkdir(parents=True, exist_ok=True)
    memory_root.mkdir(parents=True, exist_ok=True)
    outputs = {}
    fft_paths = [
        key
        for key, path in h107["path_results"].items()
        if path["family"] == "fft"
    ]
    for path_key in sorted(fft_paths):
        for scale in config["execution"]["new_scales"]:
            run_key = f"{path_key}-q{int(scale)}"
            overlay, memory, metadata, _ = compile_coupled_path(
                run_key=run_key,
                contract=h110_compile["path_contracts"][path_key],
                path=h107["path_results"][path_key],
                scale=int(scale),
                config=base_config,
            )
            overlay_path = overlay_root / f"{run_key}.json"
            memory_path = memory_root / f"{run_key}.json"
            overlay_path.write_text(canonical_json(overlay), encoding="utf-8")
            memory_path.write_text(canonical_json(memory), encoding="utf-8")
            outputs[run_key] = {
                "overlay": digest(overlay_path),
                "memory": digest(memory_path),
                "metadata": metadata,
            }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "paper_performance_targets_consumed": False,
        "outputs": outputs,
    }
    path = output_root / "fft-coupled-counter-compile-manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    passed = len(outputs) == int(config["execution"]["required_configs"]) and all(
        all(item["metadata"]["checks"].values()) for item in outputs.values()
    )
    print(json.dumps({"configs": len(outputs), "pass": passed}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
