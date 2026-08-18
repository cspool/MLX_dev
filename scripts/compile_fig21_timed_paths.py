#!/usr/bin/env python3
"""Compile H92 exact-unit timed component paths for all Figure 21 shapes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from mlxsim.dsagen_overlay import canonical_json
from mlxsim.experiments import _llama_kernel_workloads
from mlxsim.fig21_timed_paths import compile_timed_path, normalize_path
from mlxsim.workloads import compile_workload

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig21_timed_paths_v1.yaml"


def digest(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def component_io(n: int, *, sparse: bool, component: str) -> tuple[int, int]:
    workloads = _llama_kernel_workloads(n, sparse=sparse, batch=8)[component]
    profiles = [compile_workload(workload) for workload in workloads]
    load = sum(stage.load_bytes for profile in profiles for stage in profile.stages)
    store = sum(stage.store_bytes for profile in profiles for stage in profile.stages)
    return int(load), int(store)


def elementwise_io(contract: dict[str, Any]) -> tuple[int, int]:
    tokens = int(contract["elementwise"]["tokens"])
    hidden = 4096
    ffn = 11008
    load_elements = tokens * (8 * hidden + 2 * ffn)
    store_elements = tokens * (6 * hidden + ffn)
    return 2 * load_elements, 2 * store_elements


def family_spec(
    family: str, n: int, contract: dict[str, Any]
) -> tuple[dict[str, int], int, int, int]:
    if family == "elementwise":
        load, store = elementwise_io(contract)
        return (
            {
                name: int(value)
                for name, value in contract["elementwise"][
                    "fu_instruction_instances"
                ].items()
            },
            load,
            store,
            1,
        )
    mode, component = family.split("_", maxsplit=1)
    sparse = mode == "structured"
    profile = contract[f"{mode}_components"][component]
    load, store = component_io(n, sparse=sparse, component=component)
    return (
        {"fma": int(profile["fma_equivalents"])},
        load,
        store,
        5 if sparse else 1,
    )


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
    parent = json.loads(
        (
            PROJECT_ROOT / config["frozen_inputs"]["layer_contract"]["path"]
        ).read_text(encoding="utf-8")
    )
    scales = [*config["fit_scales"], *config["holdout_scales"]]
    outputs = {}
    path_contracts = {}
    checks = {}
    for shape_name, contract in parent["contracts"].items():
        n = int(contract["sequence_length"])
        for family in config["families"]:
            fu_counts, load_bytes, store_bytes, stage_count = family_spec(
                family, n, contract
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
            }
            for scale_value in scales:
                scale = int(scale_value)
                run_key = f"{key}-q{scale}"
                document, metadata = compile_timed_path(
                    name=run_key,
                    normalized=normalized,
                    scale=scale,
                    active_window=int(config["hardware"]["active_window"]),
                )
                path = config_root / f"{run_key}.json"
                path.write_text(canonical_json(document), encoding="utf-8")
                outputs[run_key] = {"artifact": digest(path), "metadata": metadata}
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
            },
            indent=2,
        )
    )
    return 0 if len(outputs) == 180 and all(
        all(item.values()) for item in checks.values()
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
