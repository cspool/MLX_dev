#!/usr/bin/env python3
"""Compile H44's frozen no-fit DSAGEN Figure 22 workload matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from mlxsim.dsagen_overlay import OverlayFixture, canonical_json, compile_aggregate_radix2_cdc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/dsagen_mlx_fig22_v1.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/environment/h44"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "operator",
        "width",
        "stages",
        "pairs_per_stage",
        "total_pairs",
        "block_count",
        "max_trip_count",
        "max_active_instruction_footprint_per_pe",
        "instruction_count",
        "memory_requests",
        "transfers",
        "operation_counts",
        "paper_performance_targets_consumed",
    )
    return {key: metadata[key] for key in keys}


def flatten_parameters(value: Any, prefix: str = "") -> list[str]:
    if isinstance(value, dict):
        result: list[str] = []
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            result.extend(flatten_parameters(child, name))
        return result
    if isinstance(value, list):
        return [prefix] if prefix else [str(item) for item in value]
    return [prefix]


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    parameters = config["parameter_manifest"]
    classified = {
        name: flatten_parameters(parameters[name])
        for name in (
            "paper_disclosed",
            "dsagen_upstream",
            "gpgpu_sim_reference_only",
            "independently_inferred_and_frozen",
            "unavailable_and_not_fitted",
        )
    }
    all_fields = [field for fields in classified.values() for field in fields]
    parameter_checks = {
        "classes_nonempty": all(classified.values()),
        "fields_unique": len(all_fields) == len(set(all_fields)),
        "legacy_calibration_forbidden": config["frozen_inputs"]["legacy_calibration"]
        ["forbidden_as_input"]
        is True,
        "paper_peak_consistent": (
            parameters["paper_disclosed"]["peak_ops_per_second"]
            / (parameters["paper_disclosed"]["frequency_ghz"] * 1e9)
            == 2
            * parameters["paper_disclosed"]["mesh"][0]
            * parameters["paper_disclosed"]["mesh"][1]
            * parameters["paper_disclosed"]["simd_width"]
        ),
        "gpgpu_timing_not_imported": parameters["gpgpu_sim_reference_only"]
        ["imported_as_mlx_timing"]
        is False,
    }
    fixture = OverlayFixture(
        mesh_width=parameters["paper_disclosed"]["mesh"][0],
        mesh_height=parameters["paper_disclosed"]["mesh"][1],
        active_window=parameters["independently_inferred_and_frozen"]["active_window"],
        simd_width=parameters["paper_disclosed"]["simd_width"],
        scalar_bytes=parameters["dsagen_upstream"]["scratchpad_bank_width_bytes"],
        skip_steps=tuple(parameters["independently_inferred_and_frozen"]["route_steps"]),
        memory_backend=config["experiment"]["memory_backend"],
        trip_count=config["experiment"]["outer_trip_count"],
    )
    outputs: dict[str, Any] = {}
    for kernel in config["experiment"]["kernels"]:
        operator_kind = "fft" if kernel == "fft" else "bsmm"
        for size in config["experiment"]["sizes"]:
            document, metadata = compile_aggregate_radix2_cdc(
                operator_kind, int(size), fixture
            )
            name = f"{kernel}-{size}"
            path = output_dir / f"fig22-{name}.json"
            path.write_text(canonical_json(document), encoding="utf-8")
            outputs[name] = {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "metadata": compact_metadata(metadata),
            }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "paper_performance_targets_consumed_by_compiler": False,
        "parameter_classes": classified,
        "parameter_checks": parameter_checks,
        "outputs": outputs,
    }
    manifest_path = output_dir / "fig22-compile-manifest.json"
    manifest_path.write_text(canonical_json(manifest), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if all(parameter_checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
