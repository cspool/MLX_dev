#!/usr/bin/env python3
"""Compile the six H113 live compute-memory coupling scenarios."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from mlxsim.coupled_pipelined_dpu_memory import scenarios
from mlxsim.dsagen_overlay import canonical_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/simulators/coupled_pipelined_dpu_memory_v1.yaml"
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
    output_root = PROJECT_ROOT / config["output_root"]
    overlay_root = output_root / "configs/overlay"
    memory_root = output_root / "configs/memory"
    overlay_root.mkdir(parents=True, exist_ok=True)
    memory_root.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, item in scenarios(config).items():
        overlay_path = overlay_root / f"{name}.json"
        memory_path = memory_root / f"{name}.json"
        overlay_path.write_text(canonical_json(item["overlay"]), encoding="utf-8")
        memory_path.write_text(canonical_json(item["memory"]), encoding="utf-8")
        outputs[name] = {
            "overlay": digest(overlay_path),
            "memory": digest(memory_path),
            "expected": item["expected"],
        }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "paper_performance_targets_consumed": False,
        "outputs": outputs,
    }
    path = output_root / "coupled-pipelined-dpu-memory-compile-manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    passed = len(outputs) == int(config["execution"]["required_scenarios"])
    print(json.dumps({"scenarios": len(outputs), "pass": passed}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
