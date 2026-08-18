#!/usr/bin/env python3
"""Compile H98 two-axis FFT and global-BSMM source paths."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from mlxsim.dsagen_overlay import canonical_json
from mlxsim.fig19_mlx_transfer import mapped_workloads
from mlxsim.fig19_source_paths import compile_fft2d_path
from mlxsim.fig21_timed_paths import compile_timed_path, normalize_path
from mlxsim.workloads import compile_workload

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig19_source_paths_v1.yaml"


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
    mapping_config = {"model": identity["model"], "mapping": identity["mapping"]}
    output_root = PROJECT_ROOT / config["output_root"]
    config_root = output_root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    scales = [*config["fit_scales"], *config["holdout_scales"]]
    outputs = {}
    path_contracts = {}
    checks = {}
    for n_value in config["sequence_lengths"]:
        n = int(n_value)
        shape = f"N{n}"
        profiles = identity["profiles"][shape]
        fft_key = f"{shape}-fft2d"
        path_contracts[fft_key] = {
            "full_scale": 4 * n,
            "stage_count": 10 + n.bit_length() - 1,
            "analytical_operations": sum(
                int(profile["operations"]) for profile in profiles["attention"]
            ),
            "isolated_offchip_bytes": sum(
                int(profile["offchip_bytes"]) for profile in profiles["attention"]
            ),
            "combined_offchip_bytes": 4 * n * 1024,
        }
        for scale_value in scales:
            scale = int(scale_value)
            run_key = f"{fft_key}-q{scale}"
            document, metadata = compile_fft2d_path(
                name=run_key,
                sequence_length=n,
                scale=scale,
                vector_bytes=int(config["hardware"]["vector_bytes"]),
                active_window=int(config["hardware"]["active_window"]),
            )
            path = config_root / f"{run_key}.json"
            path.write_text(canonical_json(document), encoding="utf-8")
            outputs[run_key] = {"artifact": digest(path), "metadata": metadata}
        checks[fft_key] = {
            "operations": path_contracts[fft_key]["analytical_operations"]
            == (512 * n * path_contracts[fft_key]["stage_count"] * 10),
            "roundtrip": path_contracts[fft_key]["isolated_offchip_bytes"]
            == 2 * path_contracts[fft_key]["combined_offchip_bytes"],
        }

        workloads = mapped_workloads(mapping_config, n)["ffn"]
        for index, (workload, profile_record) in enumerate(
            zip(workloads, profiles["ffn"], strict=True), start=1
        ):
            profile = compile_workload(workload)
            load_bytes = int(sum(stage.load_bytes for stage in profile.stages))
            store_bytes = int(sum(stage.store_bytes for stage in profile.stages))
            normalized = normalize_path(
                fu_counts={"fma": int(profile.operations / 2)},
                load_bytes=load_bytes,
                store_bytes=store_bytes,
                stage_count=len(profile.stages),
                simd_width=int(config["hardware"]["simd_width"]),
                vector_bytes=int(config["hardware"]["vector_bytes"]),
                lanes=int(config["hardware"]["lanes"]),
            )
            path_key = f"{shape}-global_ffn{index}"
            path_contracts[path_key] = normalized
            derived_fma = sum(
                int(step["trip_per_lane"])
                * int(normalized["full_scale"])
                * int(normalized["lanes"])
                * int(normalized["simd_width"])
                for step in normalized["unit_compute_steps"]
            )
            checks[path_key] = {
                "operations": derived_fma == int(profile_record["operations"] / 2),
                "stages": len(profile.stages) == profile_record["stage_count"],
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
                run_key = f"{path_key}-q{scale}"
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
    path = output_root / "fig19-source-paths-compile-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "path_count": len(path_contracts),
                "output_count": len(outputs),
                "all_checks": all(all(item.values()) for item in checks.values()),
            },
            indent=2,
        )
    )
    return 0 if len(path_contracts) == 12 and len(outputs) == 48 and all(
        all(item.values()) for item in checks.values()
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
