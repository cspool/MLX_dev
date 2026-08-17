#!/usr/bin/env python3
"""Compile the frozen H42 BSMM-8 and FFT-8 CDC fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from mlxsim.dsagen_overlay import OverlayFixture, canonical_json, write_compilation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/environment/h42"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trip-count", type=int, default=1)
    return parser.parse_args()


def qualify(path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": path.name,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "metadata": metadata,
    }


def main() -> int:
    args = parse_args()
    fixture = OverlayFixture(trip_count=args.trip_count)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bsmm_path = output_dir / "mlx-bsmm-b8.json"
    fft_path = output_dir / "mlx-fft-l8.json"
    stress_path = output_dir / "mlx-bsmm-b16-memory-stress.json"
    bsmm = write_compilation("bsmm", 8, bsmm_path, fixture)
    fft = write_compilation("fft", 8, fft_path, fixture)
    stress = write_compilation("bsmm", 16, stress_path, fixture)
    manifest = {
        "schema_version": 1,
        "experiment_id": "H42",
        "paper_performance_targets_consumed": False,
        "outputs": {
            "bsmm": qualify(bsmm_path, bsmm),
            "fft": qualify(fft_path, fft),
            "bsmm_b16_memory_stress": qualify(stress_path, stress),
        },
    }
    manifest_path = output_dir / "mlx-cdc-compiler-manifest.json"
    manifest_path.write_text(canonical_json(manifest), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
