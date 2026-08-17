#!/usr/bin/env python3
"""Compile the frozen H43 trip-count-folded MLX radix fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from mlxsim.dsagen_overlay import (
    OverlayFixture,
    canonical_json,
    compile_aggregate_radix2_cdc,
    compile_radix2_cdc,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/environment/h43"
FIXTURES = (
    ("bsmm", 8),
    ("bsmm", 16),
    ("bsmm", 32),
    ("bsmm", 64),
    ("fft", 8),
    ("fft", 256),
    ("fft", 8192),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def artifact(path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": path.name,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "metadata": compact_metadata(metadata),
    }


def compact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "schema_version",
        "compilation_mode",
        "operator",
        "width",
        "stages",
        "pairs_per_stage",
        "total_pairs",
        "active_slots_per_stage",
        "block_count",
        "max_trip_count",
        "static_instruction_count",
        "max_active_instruction_footprint_per_pe",
        "instruction_count",
        "memory_requests",
        "transfers",
        "operation_counts",
        "paper_performance_targets_consumed",
    )
    return {key: metadata[key] for key in keys if key in metadata}


def conservation(reference: dict[str, Any], aggregate: dict[str, Any]) -> dict[str, bool]:
    keys = (
        "stages",
        "pairs_per_stage",
        "total_pairs",
        "instruction_count",
        "memory_requests",
        "transfers",
        "operation_counts",
    )
    return {key: aggregate[key] == reference[key] for key in keys}


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture = OverlayFixture()
    outputs: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    for operator_kind, width in FIXTURES:
        aggregate_config, aggregate_metadata = compile_aggregate_radix2_cdc(
            operator_kind, width, fixture
        )
        _, reference_metadata = compile_radix2_cdc(operator_kind, width, fixture)
        name = f"{operator_kind}-{width}"
        path = output_dir / f"mlx-{operator_kind}-{width}-aggregate.json"
        path.write_text(canonical_json(aggregate_config), encoding="utf-8")
        outputs[name] = artifact(path, aggregate_metadata)
        checks = conservation(reference_metadata, aggregate_metadata)
        comparisons[name] = {
            "reference": compact_metadata(reference_metadata),
            "checks": checks,
            "pass": all(checks.values()),
        }

    fixed_fixture = replace(fixture, memory_backend="fixed")
    pairwise_fixed, pairwise_fixed_metadata = compile_radix2_cdc(
        "bsmm", 8, fixed_fixture
    )
    aggregate_fixed, aggregate_fixed_metadata = compile_aggregate_radix2_cdc(
        "bsmm", 8, fixed_fixture
    )
    for name, config, metadata in (
        ("bsmm-8-pairwise-fixed", pairwise_fixed, pairwise_fixed_metadata),
        ("bsmm-8-aggregate-fixed", aggregate_fixed, aggregate_fixed_metadata),
    ):
        path = output_dir / f"mlx-{name}.json"
        path.write_text(canonical_json(config), encoding="utf-8")
        outputs[name] = artifact(path, metadata)

    manifest = {
        "schema_version": 1,
        "experiment_id": "H43",
        "paper_performance_targets_consumed": False,
        "outputs": outputs,
        "comparisons": comparisons,
    }
    manifest_path = output_dir / "mlx-aggregate-manifest.json"
    manifest_path.write_text(canonical_json(manifest), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
