#!/usr/bin/env python3
"""Compile H128's 48 current-coupled Figure 19 configurations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from mlxsim.dsagen_overlay import canonical_json
from mlxsim.fig19_coupled_paths import compile_fig19_coupled_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig19_coupled_paths_v1.yaml"


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
    parent_manifest = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h98_manifest"]["path"]).read_text()
    )
    output_root = PROJECT_ROOT / config["output_root"]
    overlay_root = output_root / "configs/overlay"
    memory_root = output_root / "configs/memory"
    overlay_root.mkdir(parents=True, exist_ok=True)
    memory_root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Any] = {}
    for run_key, item in parent_manifest["outputs"].items():
        source_path = PROJECT_ROOT / item["artifact"]["path"]
        source = json.loads(source_path.read_text())
        overlay, memory, metadata = compile_fig19_coupled_path(
            run_key=run_key,
            source=source,
            source_metadata=item["metadata"],
            config=config,
        )
        overlay_path = overlay_root / f"{run_key}.json"
        memory_path = memory_root / f"{run_key}.json"
        overlay_path.write_text(canonical_json(overlay))
        memory_path.write_text(canonical_json(memory))
        outputs[run_key] = {
            "parent": digest(source_path),
            "overlay": digest(overlay_path),
            "memory": digest(memory_path),
            "metadata": metadata,
        }
    checks = {
        "outputs": len(outputs) == int(config["execution"]["required_configs"]),
        "contracts": all(
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
        "path_contracts": parent_manifest["path_contracts"],
        "checks": checks,
    }
    path = output_root / "fig19-coupled-compile-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"outputs": len(outputs), "checks": checks}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
