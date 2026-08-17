#!/usr/bin/env python3
"""Compile H57's target-independent MLX configs and Xavier jobs."""

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
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig20_sparse_xavier_v1.yaml"
DEFAULT_ELF = PROJECT_ROOT / "third_party/dsa-framework/dsa-apps/sdk/compiled/ss-mlx-dma.out"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/environment/h57"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--elf", type=Path, default=DEFAULT_ELF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def digest(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    symbols = read_elf_symbols(args.elf.resolve())
    output_root = args.output_dir.resolve()
    mlx_root = output_root / "mlx"
    mlx_root.mkdir(parents=True, exist_ok=True)
    proxy_specs = {
        "bsmm": {"name": "bsmm", "family": "qkv_bsmm", "block_size": 32, "stages": 5},
        "fft": {"name": "fft", "family": "fft", "stages": 7},
    }
    outputs = []
    jobs = []
    for proxy_name, operator in proxy_specs.items():
        kernel = next(item for item in config["kernels"].values() if item["proxy"] == proxy_name)
        for case_name, case in config["cases"].items():
            compiler_case = {
                "name": case_name,
                "sequence": 256 if case_name == "short" else 8192,
                "trip_count": case["mlx_trip"],
            }
            document, metadata = compile_operator_proxy(
                operator, compiler_case, symbols, arithmetic_expanded=True
            )
            document["pe_dependency_model"] = "paper_static"
            metadata["pe_dependency_model"] = "paper_static"
            metadata["scoreboard_is_paper_semantics"] = False
            name = f"{proxy_name}--{case_name}"
            path = mlx_root / f"{name}.json"
            path.write_text(canonical_json(document), encoding="utf-8")
            mlx_fmas = int(metadata["operation_counts"]["fma"]) * int(
                config["normalization"]["mlx_simd_lanes"]
            )
            outputs.append(
                {
                    "name": name,
                    "artifact": digest(path),
                    "mlx_fma_equivalents": mlx_fmas,
                    "metadata": metadata,
                }
            )
            count = int(case["gpu_count"])
            parameter = int(kernel["gpu_parameter"])
            jobs.append(
                {
                    "name": name,
                    "gpu_operation": proxy_name,
                    "gpu_count": count,
                    "gpu_parameter": parameter,
                    "gpu_fma_equivalents": count
                    * parameter
                    * int(kernel["gpu_fma_per_element_stage"]),
                }
            )
    manifest = {
        "schema_version": 1,
        "experiment_id": "H57",
        "paper_target_values_consumed": False,
        "mlx_outputs": outputs,
        "gpu_jobs": jobs,
        "normalization": {
            "device_clock_hz": config["normalization"]["xavier_clock_hz"]
        },
    }
    (output_root / "fig20-sparse-xavier-compile-manifest.json").write_text(
        canonical_json(manifest), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
