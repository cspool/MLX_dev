#!/usr/bin/env python3
"""Create H69 adapter configs with only the memory backend changed."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from mlxsim.dsagen_overlay import canonical_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/multiport_spad_v1.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def digest(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path.resolve().relative_to(PROJECT_ROOT)),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def transform(document: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(document)
    result["memory_backend"] = "dsagen_spad"
    return result


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    parent_root = PROJECT_ROOT / config["parent_configs"]
    output_root = PROJECT_ROOT / config["output_root"] / "configs"
    records: dict[str, Any] = {}
    for parent_path in sorted(parent_root.glob("fig10-scale-*.json")):
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        output_path = output_root / parent_path.name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output = transform(parent)
        output_path.write_text(canonical_json(output), encoding="utf-8")
        restored = {**output, "memory_backend": parent["memory_backend"]}
        key = parent_path.stem.removeprefix("fig10-scale-")
        records[key] = {
            "parent": digest(parent_path),
            "output": digest(output_path),
            "only_backend_changed": restored == parent,
            "ports": int(parent["routing"]["mesh_width"]),
            "axis": "x",
        }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "paper_performance_targets_consumed": False,
        "record_count": len(records),
        "all_only_backend_changed": all(
            record["only_backend_changed"] for record in records.values()
        ),
        "records": records,
    }
    path = PROJECT_ROOT / config["output_root"] / "multiport-compile-manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(manifest), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if len(records) == 20 and manifest["all_only_backend_changed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
