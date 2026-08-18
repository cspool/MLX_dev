#!/usr/bin/env python3
"""Compile H118's 16 direct full-size coupled Figure 22 workloads."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from mlxsim.dsagen_overlay import canonical_json
from mlxsim.fig22_coupled_workloads import compile_fig22_coupled_workload

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig22_coupled_workloads_v1.yaml"


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
    config = yaml.safe_load(args.config.read_text())
    h62_manifest_path = PROJECT_ROOT / config["frozen_inputs"]["h62_compile"]["path"]
    h62_manifest = json.loads(h62_manifest_path.read_text())
    output_root = PROJECT_ROOT / config["output_root"]
    overlay_root = output_root / "configs/overlay"
    memory_root = output_root / "configs/memory"
    overlay_root.mkdir(parents=True, exist_ok=True)
    memory_root.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Any] = {}
    source_checks: dict[str, bool] = {}
    for operator in config["workloads"]["operators"]:
        for size in config["workloads"]["sizes"]:
            key = f"{operator}-{int(size)}"
            overlay, memory, metadata, source = compile_fig22_coupled_workload(
                operator, int(size), config
            )
            parent = h62_manifest["outputs"][key]["primary"]
            parent_path = PROJECT_ROOT / parent["path"]
            source_payload = canonical_json(source)
            source_checks[key] = (
                parent_path.read_text() == source_payload
                and digest(parent_path)["bytes"] == int(parent["bytes"])
                and digest(parent_path)["sha256"] == parent["sha256"]
            )
            overlay_path = overlay_root / f"{key}.json"
            memory_path = memory_root / f"{key}.json"
            overlay_path.write_text(canonical_json(overlay))
            memory_path.write_text(canonical_json(memory))
            outputs[key] = {
                "parent": digest(parent_path),
                "overlay": digest(overlay_path),
                "memory": digest(memory_path),
                "metadata": metadata,
            }

    checks = {
        "output_count": len(outputs) == int(config["workloads"]["required_paths"]),
        "source_replays": all(source_checks.values()),
        "compile_contracts": all(
            all(item["metadata"]["checks"].values()) for item in outputs.values()
        ),
        "target_free": all(
            item["metadata"]["paper_performance_targets_consumed"] is False
            for item in outputs.values()
        ),
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "paper_performance_targets_consumed": False,
        "outputs": outputs,
        "source_checks": source_checks,
        "checks": checks,
    }
    manifest_path = output_root / "fig22-coupled-compile-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"outputs": len(outputs), "checks": checks}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
