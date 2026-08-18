#!/usr/bin/env python3
"""Compile H102 exact batch-32 paths over the full 4x4 MLX mesh."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from mlxsim.dsagen_overlay import canonical_json
from mlxsim.fig21_timed_paths import normalize_path
from mlxsim.fig24_25_full_mesh_paths import (
    compile_full_mesh_fft_cmp_path,
    compile_full_mesh_swa_path,
    compile_full_mesh_timed_path,
    normalize_full_mesh_path,
)
from mlxsim.schema import Workload
from mlxsim.workloads import compile_workload

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig24_25_full_mesh_paths_v1.yaml"


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
    identity = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["identity"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    unique: dict[str, dict[str, Any]] = {}
    for item in identity["comparisons"]:
        key = item["key"]
        if key in unique:
            assert unique[key]["actual"] == item["actual"]
        else:
            unique[key] = item
    output_root = PROJECT_ROOT / config["output_root"]
    config_root = output_root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    scales = [*config["fit_scales"], *config["holdout_scales"]]
    outputs: dict[str, dict[str, Any]] = {}
    contracts: dict[str, dict[str, Any]] = {}
    checks: dict[str, dict[str, bool]] = {}
    for key, item in unique.items():
        case = item["case"]
        operator = item["operator"]
        family = operator["family"]
        if family == "qkv_bsmm":
            profile = compile_workload(
                Workload(
                    kernel="bsmm",
                    n=int(case["n"]),
                    d=int(case["d"]),
                    batch=int(case["batch"]),
                    projections=3,
                    block_size=int(operator["block_size"]),
                )
            )
            load = int(sum(stage.load_bytes for stage in profile.stages))
            store = int(sum(stage.store_bytes for stage in profile.stages))
            four_strip = normalize_path(
                fu_counts={"fma": int(item["actual"]["fu"]["fma"])},
                load_bytes=load,
                store_bytes=store,
                stage_count=int(item["actual"]["stage_count"]),
                simd_width=int(config["hardware"]["simd_width"]),
                vector_bytes=int(config["hardware"]["vector_bytes"]),
                lanes=4,
            )
            normalized = normalize_full_mesh_path(four_strip)
            contracts[key] = {
                "family": family,
                "normalized": normalized,
                "actual": item["actual"],
                "case": case,
                "operator": operator,
            }
            checks[key] = {
                "fma": normalized["full_fu_counts"]["fma"]
                == item["actual"]["fu"]["fma"],
                "stages": normalized["stage_count"]
                == item["actual"]["stage_count"],
                "lanes": normalized["lanes"] == 16,
            }
            builder = "qkv"
        elif family == "fft":
            full_scale = int(case["batch"]) * int(case["d"]) * int(case["n"]) // 512
            contracts[key] = {
                "family": family,
                "full_scale": full_scale,
                "actual": item["actual"],
                "case": case,
                "operator": operator,
            }
            checks[key] = {"stages": int(item["actual"]["stage_count"]) > 0}
            builder = "fft"
        elif family == "swa":
            full_scale = (
                int(case["batch"])
                * int(case["n"])
                * int(operator["window"])
                // 128
            )
            contracts[key] = {
                "family": family,
                "full_scale": full_scale,
                "actual": item["actual"],
                "case": case,
                "operator": operator,
            }
            checks[key] = {
                "stages": item["actual"]["stage_count"] == 4,
                "query_tile": item["actual"]["query_tile"]
                == operator["query_tile"],
            }
            builder = "swa"
        else:
            raise ValueError(family)
        for scale_value in scales:
            scale = int(scale_value)
            run_key = f"{key}-q{scale}"
            if builder == "qkv":
                document, metadata = compile_full_mesh_timed_path(
                    name=run_key,
                    normalized=normalized,
                    scale=scale,
                    active_window=int(config["hardware"]["active_window"]),
                )
            elif builder == "fft":
                document, metadata = compile_full_mesh_fft_cmp_path(
                    name=run_key,
                    sequence_length=int(case["n"]),
                    hidden_dimension=int(case["d"]),
                    batch=int(case["batch"]),
                    scale=scale,
                )
            else:
                document, metadata = compile_full_mesh_swa_path(
                    name=run_key,
                    sequence_length=int(case["n"]),
                    hidden_dimension=int(case["d"]),
                    batch=int(case["batch"]),
                    window=int(operator["window"]),
                    query_tile=int(operator["query_tile"]),
                    scale=scale,
                )
            path = config_root / f"{run_key}.json"
            path.write_text(canonical_json(document), encoding="utf-8")
            outputs[run_key] = {"artifact": digest(path), "metadata": metadata}
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "paper_performance_targets_consumed": False,
        "outputs": outputs,
        "path_contracts": contracts,
        "checks": checks,
    }
    path = output_root / "fig24-25-full-mesh-compile-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "path_count": len(contracts),
        "output_count": len(outputs),
        "all_checks": all(all(values.values()) for values in checks.values()),
    }
    print(json.dumps(summary, indent=2))
    return 0 if summary == {"path_count": 48, "output_count": 192, "all_checks": True} else 1


if __name__ == "__main__":
    raise SystemExit(main())
