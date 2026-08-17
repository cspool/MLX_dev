#!/usr/bin/env python3
"""Compile all H62 Figure 10 mappings and deterministic replay copies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from mlxsim.fig10_mapping import (
    canonical_json,
    compile_fig10_mapping,
    fixed_memory_control,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig10_mapping_v1.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def digest(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    resolved = path.resolve()
    try:
        display_path = resolved.relative_to(PROJECT_ROOT)
    except ValueError:
        display_path = resolved
    return {
        "path": str(display_path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def write(path: Path, document: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(document), encoding="utf-8")
    return digest(path)


def main() -> int:
    args = parse_args()
    specification = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = PROJECT_ROOT / specification["experiment"]["output_root"]
    primary_root = output / "configs"
    replay_root = output / "replay"
    outputs: dict[str, Any] = {}
    all_identical = True
    for operator in specification["experiment"]["operators"]:
        for size in specification["experiment"]["sizes"]:
            key = f"{operator}-{size}"
            document, metadata = compile_fig10_mapping(operator, int(size))
            primary = write(primary_root / f"fig10-{key}.json", document)
            replay_document, replay_metadata = compile_fig10_mapping(
                operator, int(size)
            )
            replay = write(replay_root / f"fig10-{key}.json", replay_document)
            identical = primary["sha256"] == replay["sha256"]
            all_identical &= identical and metadata == replay_metadata
            outputs[key] = {
                "primary": primary,
                "replay": replay,
                "identical": identical,
                "metadata": metadata,
            }
    controls: dict[str, Any] = {}
    for operator in specification["experiment"]["operators"]:
        document, _ = compile_fig10_mapping(operator, 64)
        control = fixed_memory_control(document)
        controls[operator] = write(
            primary_root / f"fig10-{operator}-64-fixed.json", control
        )
    manifest = {
        "schema_version": 1,
        "experiment_id": specification["experiment_id"],
        "paper_performance_targets_consumed": False,
        "output_count": len(outputs),
        "all_identical": all_identical,
        "outputs": outputs,
        "fixed_controls": controls,
    }
    manifest_path = output / "fig10-compile-manifest.json"
    manifest_path.write_text(canonical_json(manifest), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if len(outputs) == 16 and all_identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
