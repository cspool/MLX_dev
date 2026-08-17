#!/usr/bin/env python3
"""Create H63 fixed-memory controls by changing one root field only."""

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
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig10_fig22_transfer_v1.yaml"


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


def fixed_control(document: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(document)
    result["memory_backend"] = "fixed"
    return result


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source_root = PROJECT_ROOT / config["configs"]
    output_root = PROJECT_ROOT / config["output_root"] / "fixed"
    records: dict[str, Any] = {}
    for source in sorted(source_root.glob("fig10-*-*.json")):
        if source.name.endswith("-fixed.json"):
            continue
        parent = json.loads(source.read_text(encoding="utf-8"))
        output = output_root / source.name
        output.parent.mkdir(parents=True, exist_ok=True)
        control = fixed_control(parent)
        output.write_text(canonical_json(control), encoding="utf-8")
        restored = copy.deepcopy(control)
        restored["memory_backend"] = parent["memory_backend"]
        records[source.stem.removeprefix("fig10-")] = {
            "parent": digest(source),
            "control": digest(output),
            "only_backend_changed": restored == parent,
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
    path = PROJECT_ROOT / config["output_root"] / "fixed-control-manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(manifest), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if len(records) == 16 and manifest["all_only_backend_changed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
