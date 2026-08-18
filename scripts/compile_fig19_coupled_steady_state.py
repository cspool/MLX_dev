#!/usr/bin/env python3
"""Compile H129 q64/q128 Figure 19 coupled extensions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from mlxsim.dsagen_overlay import canonical_json
from mlxsim.fig19_coupled_paths import compile_fig19_coupled_path
from mlxsim.fig19_source_paths import compile_fft2d_path
from mlxsim.fig21_timed_paths import compile_timed_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig19_coupled_steady_state_v1.yaml"


def digest(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def source_config(
    path_key: str, scale: int, h98_manifest: dict[str, Any], h98_config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_key = f"{path_key}-q{scale}"
    if path_key.endswith("fft2d"):
        sequence_length = int(path_key.split("-", 1)[0][1:])
        return compile_fft2d_path(
            name=run_key,
            sequence_length=sequence_length,
            scale=scale,
            vector_bytes=int(h98_config["hardware"]["vector_bytes"]),
            active_window=int(h98_config["hardware"]["active_window"]),
        )
    normalized = h98_manifest["path_contracts"][path_key]
    return compile_timed_path(
        name=run_key,
        normalized=normalized,
        scale=scale,
        active_window=int(h98_config["hardware"]["active_window"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    h128_config = yaml.safe_load(
        (PROJECT_ROOT / config["frozen_inputs"]["h128_config"]["path"]).read_text()
    )
    h98_manifest = json.loads(
        (PROJECT_ROOT / h128_config["frozen_inputs"]["h98_manifest"]["path"]).read_text()
    )
    h98_config = yaml.safe_load(
        (PROJECT_ROOT / h128_config["frozen_inputs"]["h98_config"]["path"]).read_text()
    )
    output_root = PROJECT_ROOT / config["output_root"]
    overlay_root = output_root / "configs/overlay"
    memory_root = output_root / "configs/memory"
    overlay_root.mkdir(parents=True, exist_ok=True)
    memory_root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Any] = {}
    for path_key in config["paths"]:
        for scale_value in config["scales"]["holdout"]:
            scale = int(scale_value)
            run_key = f"{path_key}-q{scale}"
            source, source_metadata = source_config(
                path_key, scale, h98_manifest, h98_config
            )
            overlay, memory, metadata = compile_fig19_coupled_path(
                run_key=run_key,
                source=source,
                source_metadata=source_metadata,
                config=h128_config,
            )
            overlay_path = overlay_root / f"{run_key}.json"
            memory_path = memory_root / f"{run_key}.json"
            overlay_path.write_text(canonical_json(overlay))
            memory_path.write_text(canonical_json(memory))
            outputs[run_key] = {
                "overlay": digest(overlay_path),
                "memory": digest(memory_path),
                "metadata": metadata,
            }
    checks = {
        "outputs": len(outputs) == int(config["execution"]["required_configs"]),
        "contracts": all(
            all(item["metadata"]["checks"].values()) for item in outputs.values()
        ),
        "power_of_two_tiles": all(
            item["metadata"]["tile_count"]
            & (item["metadata"]["tile_count"] - 1)
            == 0
            for item in outputs.values()
        ),
        "target_free": all(
            item["metadata"]["paper_performance_targets_consumed"] is False
            for item in outputs.values()
        ),
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "paper_performance_targets_consumed": False,
        "outputs": outputs,
        "checks": checks,
    }
    path = output_root / "fig19-coupled-steady-state-compile-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"outputs": len(outputs), "checks": checks}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
