#!/usr/bin/env python3
"""Compile all 48 H102 paths into H106 full memory schedules."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from mlxsim.dsagen_overlay import canonical_json
from mlxsim.full_mesh_memory_residency import compile_residency_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/simulators/full_mesh_memory_residency_v1.yaml"
)


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
    snapshot = json.loads(
        (
            PROJECT_ROOT / config["frozen_inputs"]["contracts"]["path"]
        ).read_text(encoding="utf-8")
    )
    output_root = PROJECT_ROOT / config["output_root"]
    config_root = output_root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for key, contract in snapshot["paths"].items():
        memory, metadata = compile_residency_path(
            key=key, contract=contract, config=config
        )
        path = config_root / f"{key}.json"
        path.write_text(canonical_json(memory), encoding="utf-8")
        outputs[key] = {"artifact": digest(path), "metadata": metadata}
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "paper_performance_targets_consumed": False,
        "family_counts": snapshot["family_counts"],
        "outputs": outputs,
    }
    path = output_root / "full-mesh-memory-residency-compile-manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    all_checks = all(
        all(item["metadata"]["checks"].values()) for item in outputs.values()
    )
    summary = {
        "paths": len(outputs),
        "family_counts": snapshot["family_counts"],
        "all_checks": all_checks,
        "maximum_tiles": max(
            item["metadata"]["tile_count"] for item in outputs.values()
        ),
    }
    print(json.dumps(summary, indent=2))
    return 0 if (
        len(outputs) == int(config["execution"]["required_paths"])
        and snapshot["family_counts"]
        == config["execution"]["required_family_counts"]
        and all_checks
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())

