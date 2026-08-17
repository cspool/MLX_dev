#!/usr/bin/env python3
"""Compile the target-independent MLX configs and Orin jobs for H55."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from mlxsim.dsagen_dma import read_elf_symbols
from mlxsim.dsagen_operator_sweep import compile_operator_proxy
from mlxsim.dsagen_overlay import canonical_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig24_cross_simulator_v1.yaml"
DEFAULT_ELF = PROJECT_ROOT / "third_party/dsa-framework/dsa-apps/sdk/compiled/ss-mlx-dma.out"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/environment/h55"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--elf", type=Path, default=DEFAULT_ELF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def digest(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def compiler_operator(specification: dict[str, Any]) -> dict[str, Any]:
    if specification["family"] == "fft":
        return {"name": specification["name"], "family": "fft", "stages": 7}
    if specification["family"] == "qkv_bsmm":
        return {
            "name": specification["name"],
            "family": "qkv_bsmm",
            "block_size": specification["block_size"],
            "stages": specification["mlx_stages"],
        }
    return {
        "name": specification["name"],
        "family": "swa",
        "window": specification["window"],
        "query_tile": specification["query_tile"],
        "fma_repeats": 1,
        "score_fma_groups": specification["mlx_score_fma_groups"],
        "sv_fma_groups": specification["mlx_sv_fma_groups"],
        "kv_load_waves": specification["mlx_kv_load_waves"],
    }


def gpu_fma_count(operator: dict[str, Any], count: int) -> int:
    if operator["family"] in {"fft", "qkv_bsmm"}:
        return count * int(operator["gpu_parameter"]) * int(
            operator["gpu_fma_per_element_stage"]
        )
    return count * int(operator["gpu_parameter"]) * int(
        operator["gpu_fma_per_element_parameter"]
    )


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    symbols = read_elf_symbols(args.elf.resolve())
    output_root = args.output_dir.resolve()
    mlx_root = output_root / "mlx"
    mlx_root.mkdir(parents=True, exist_ok=True)
    outputs = []
    jobs = []
    for operator_spec in config["operators"]:
        operator = compiler_operator(operator_spec)
        for case in config["cases"]:
            compiler_case = {
                "name": case["name"],
                "sequence": case["n"],
                "trip_count": case["trip"],
            }
            document, metadata = compile_operator_proxy(
                operator,
                compiler_case,
                symbols,
                arithmetic_expanded=True,
            )
            document["pe_dependency_model"] = "paper_static"
            metadata["pe_dependency_model"] = "paper_static"
            metadata["scoreboard_is_paper_semantics"] = False
            filename = f"{operator['name']}--{case['name']}.json"
            path = mlx_root / filename
            path.write_text(canonical_json(document), encoding="utf-8")
            mlx_fmas = int(metadata["operation_counts"]["fma"]) * int(
                config["normalization"]["mlx_simd_lanes"]
            )
            outputs.append(
                {
                    "name": path.stem,
                    "operator": operator["name"],
                    "case": case["name"],
                    "artifact": digest(path),
                    "mlx_fma_equivalents": mlx_fmas,
                    "metadata": metadata,
                }
            )
            jobs.append(
                {
                    "name": path.stem,
                    "operator": operator["name"],
                    "case": case["name"],
                    "gpu_operation": operator_spec["gpu_operation"],
                    "gpu_count": int(case["gpu_count"]),
                    "gpu_parameter": int(operator_spec["gpu_parameter"]),
                    "gpu_fma_equivalents": gpu_fma_count(
                        operator_spec, int(case["gpu_count"])
                    ),
                }
            )
    manifest = {
        "schema_version": 1,
        "experiment_id": "H55",
        "paper_target_values_consumed": False,
        "output_count": len(outputs),
        "mlx_outputs": outputs,
        "orin_jobs": jobs,
        "normalization": config["normalization"],
    }
    manifest_path = output_root / "fig24-cross-simulator-compile-manifest.json"
    manifest_path.write_text(canonical_json(manifest), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
