#!/usr/bin/env python3
"""Compile all target-free H64 Figure 23 scalability configurations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from mlxsim.dsagen_overlay import canonical_json
from mlxsim.fig10_scaling import compile_scalability_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig10_scalability_v1.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def digest(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path.resolve().relative_to(PROJECT_ROOT)),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = PROJECT_ROOT / config["output_root"]
    outputs: dict[str, Any] = {}
    work: dict[str, Any] = {}
    all_identical = True
    for sequence in config["workload"]["sequence_lengths"]:
        work[str(sequence)] = {}
        for name, hardware in config["configurations"].items():
            mesh = tuple(int(value) for value in hardware["mesh"])
            document, metadata = compile_scalability_config(
                sequence_length=int(sequence),
                batch=int(config["workload"]["batch"]),
                hidden_width=int(config["workload"]["hidden_width"]),
                hardware_name=name,
                simd_width=int(hardware["simd_width"]),
                mesh=mesh,  # type: ignore[arg-type]
                active_window=int(config["workload"]["active_window"]),
            )
            replay, replay_metadata = compile_scalability_config(
                sequence_length=int(sequence),
                batch=int(config["workload"]["batch"]),
                hidden_width=int(config["workload"]["hidden_width"]),
                hardware_name=name,
                simd_width=int(hardware["simd_width"]),
                mesh=mesh,  # type: ignore[arg-type]
                active_window=int(config["workload"]["active_window"]),
            )
            key = f"{sequence}-{name}"
            primary_path = root / "configs" / f"fig10-scale-{key}.json"
            replay_path = root / "replay" / f"fig10-scale-{key}.json"
            primary_path.parent.mkdir(parents=True, exist_ok=True)
            replay_path.parent.mkdir(parents=True, exist_ok=True)
            primary_path.write_text(canonical_json(document), encoding="utf-8")
            replay_path.write_text(canonical_json(replay), encoding="utf-8")
            primary = digest(primary_path)
            replay_file = digest(replay_path)
            identical = primary["sha256"] == replay_file["sha256"]
            all_identical &= identical and metadata == replay_metadata
            outputs[key] = {
                "primary": primary,
                "replay": replay_file,
                "identical": identical,
                "metadata": metadata,
            }
            work[str(sequence)][name] = metadata["lane_normalized_work"]
    conservation = {
        sequence: {
            name: values == per_config["baseline"]
            for name, values in per_config.items()
        }
        for sequence, per_config in work.items()
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "paper_performance_targets_consumed": False,
        "output_count": len(outputs),
        "all_identical": all_identical,
        "work": work,
        "conservation": conservation,
        "outputs": outputs,
    }
    path = root / "fig10-scalability-compile-manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(manifest), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if len(outputs) == 20 and all_identical and all(
        all(values.values()) for values in conservation.values()
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
