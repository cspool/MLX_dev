#!/usr/bin/env python3
"""Compile H141 complete-block scaling robustness configurations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from mlxsim.dsagen_overlay import canonical_json
from mlxsim.fig23_complete_block import compile_complete_block_scaling

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig23_complete_block_robustness_v1.yaml"


def qualify(path: Path) -> dict[str, Any]:
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
    config = yaml.safe_load(args.config.read_text())
    base = json.loads((PROJECT_ROOT / config["frozen_inputs"]["h48_fixed"]["path"]).read_text())
    output_root = PROJECT_ROOT / config["output_root"]
    outputs: dict[str, Any] = {}
    conservation: dict[str, Any] = {}
    all_identical = True
    for window in config["robustness_grid"]["active_windows"]:
        for sequence in config["paper_disclosed_workload"]["sequence_lengths"]:
            group = f"N{sequence}-w{window}"
            conservation[group] = {}
            reference_work = None
            for name, hardware in config["robustness_grid"]["configurations"].items():
                kwargs = {
                    "sequence_length": int(sequence),
                    "hidden_dimension": int(config["paper_disclosed_workload"]["hidden_dimension"]),
                    "batch": int(config["paper_disclosed_workload"]["batch"]),
                    "active_window": int(window),
                    "baseline_repeat": int(
                        config["robustness_grid"]["baseline_repeat_by_sequence"][int(sequence)]
                    ),
                    "hardware_name": name,
                    "simd_width": int(hardware["simd_width"]),
                    "mesh": tuple(int(value) for value in hardware["mesh"]),
                }
                document, metadata = compile_complete_block_scaling(base, **kwargs)
                replay, replay_metadata = compile_complete_block_scaling(base, **kwargs)
                key = f"{group}-{name}"
                primary_path = output_root / "configs" / f"{key}.json"
                replay_path = output_root / "replay" / f"{key}.json"
                primary_path.parent.mkdir(parents=True, exist_ok=True)
                replay_path.parent.mkdir(parents=True, exist_ok=True)
                primary_path.write_text(canonical_json(document))
                replay_path.write_text(canonical_json(replay))
                primary = qualify(primary_path)
                replay_file = qualify(replay_path)
                identical = (
                    primary["sha256"] == replay_file["sha256"] and metadata == replay_metadata
                )
                all_identical &= identical
                outputs[key] = {
                    "primary": primary,
                    "replay": replay_file,
                    "identical": identical,
                    "metadata": metadata,
                }
                scalarized = {
                    key: value
                    for key, value in metadata["work"].items()
                    if key.startswith("scalarized_")
                }
                if reference_work is None:
                    reference_work = scalarized
                conservation[group][name] = scalarized == reference_work
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "paper_performance_targets_consumed": False,
        "output_count": len(outputs),
        "all_identical": all_identical,
        "conservation": conservation,
        "outputs": outputs,
    }
    path = output_root / "complete-block-compile-manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output_count": len(outputs),
                "all_identical": all_identical,
                "all_conserved": all(all(group.values()) for group in conservation.values()),
                "manifest": str(path.relative_to(PROJECT_ROOT)),
            },
            indent=2,
        )
    )
    return (
        0
        if len(outputs) == int(config["acceptance"]["expected_configs"])
        and all_identical
        and all(all(group.values()) for group in conservation.values())
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
