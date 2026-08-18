#!/usr/bin/env python3
"""Compile H152 exact Figure 21 paths across all 16 physical PEs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from compile_fig21_timed_paths import family_spec

from mlxsim.dsagen_overlay import canonical_json
from mlxsim.fig21_full_array_paths import compile_full_array_path
from mlxsim.fig21_timed_paths import normalize_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig21_full_array_paths_v1.yaml"


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
    config = yaml.safe_load(args.config.read_text())
    output_root = PROJECT_ROOT / config["output_root"]
    config_root = output_root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    contracts = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["layer_contract"]["path"]).read_text()
    )["contracts"]
    scales = [*config["fit_scales"], *config["holdout_scales"]]
    outputs = {}
    path_contracts = {}
    checks = {}
    for shape_name, contract in contracts.items():
        sequence = int(contract["sequence_length"])
        for family in config["families"]:
            fu_counts, load_bytes, store_bytes, stage_count = family_spec(
                family, sequence, contract
            )
            normalized = normalize_path(
                fu_counts=fu_counts,
                load_bytes=load_bytes,
                store_bytes=store_bytes,
                stage_count=stage_count,
                simd_width=int(config["hardware"]["simd_width"]),
                vector_bytes=int(config["hardware"]["vector_bytes"]),
                lanes=int(config["hardware"]["lanes"]),
            )
            key = f"{shape_name}-{family}"
            path_contracts[key] = normalized
            derived_fu: dict[str, int] = {}
            for step in normalized["unit_compute_steps"]:
                operation = step["operation"]
                derived_fu[operation] = derived_fu.get(operation, 0) + (
                    int(step["trip_per_lane"])
                    * int(normalized["full_scale"])
                    * int(normalized["lanes"])
                    * int(normalized["simd_width"])
                )
            checks[key] = {
                "fu": derived_fu == fu_counts,
                "load": int(normalized["unit_load_trip_per_lane"])
                * int(normalized["full_scale"])
                * int(normalized["lanes"])
                * int(normalized["vector_bytes"])
                == load_bytes,
                "store": int(normalized["unit_store_trip_per_lane"])
                * int(normalized["full_scale"])
                * int(normalized["lanes"])
                * int(normalized["vector_bytes"])
                == store_bytes,
                "lanes": normalized["lanes"] == 16,
            }
            for scale_value in scales:
                scale = int(scale_value)
                run_key = f"{key}-q{scale}"
                document, metadata = compile_full_array_path(
                    name=run_key,
                    normalized=normalized,
                    scale=scale,
                    active_window=int(config["hardware"]["active_window"]),
                )
                replay, replay_metadata = compile_full_array_path(
                    name=run_key,
                    normalized=normalized,
                    scale=scale,
                    active_window=int(config["hardware"]["active_window"]),
                )
                path = config_root / f"{run_key}.json"
                path.write_text(canonical_json(document))
                outputs[run_key] = {
                    "artifact": digest(path),
                    "metadata": metadata,
                    "deterministic": document == replay and metadata == replay_metadata,
                }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "paper_performance_targets_consumed": False,
        "outputs": outputs,
        "path_contracts": path_contracts,
        "checks": checks,
    }
    path = output_root / "fig21-timed-paths-compile-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output_count": len(outputs),
                "path_count": len(path_contracts),
                "all_checks": all(all(item.values()) for item in checks.values()),
                "all_deterministic": all(item["deterministic"] for item in outputs.values()),
            },
            indent=2,
        )
    )
    return (
        0
        if len(outputs) == 180
        and all(all(item.values()) for item in checks.values())
        and all(item["deterministic"] for item in outputs.values())
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
