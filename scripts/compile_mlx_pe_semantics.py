#!/usr/bin/env python3
"""Create H52 paper-static full-block configs from frozen H48 documents."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from mlxsim.dsagen_overlay import canonical_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARENT = PROJECT_ROOT / "artifacts/environment/h48"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/environment/h52"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--replay-check", type=Path)
    return parser.parse_args()


def digest(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def paper_static_document(parent: dict[str, Any]) -> dict[str, Any]:
    document = copy.deepcopy(parent)
    document["pe_dependency_model"] = "paper_static"
    document["metadata"]["pe_dependency_model"] = "paper_static"
    document["metadata"]["scoreboard_is_paper_semantics"] = False
    document["metadata"]["paper_performance_targets_consumed"] = False
    return document


def main() -> int:
    args = parse_args()
    parent_dir = args.parent_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name in ("fixed", "dma"):
        parent_path = parent_dir / f"mlx-full-block-{name}.json"
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        document = paper_static_document(parent)
        output_path = output_dir / f"mlx-full-block-{name}.json"
        output_path.write_text(canonical_json(document), encoding="utf-8")
        outputs[name] = {
            "parent": digest(parent_path),
            "output": digest(output_path),
            "only_registered_fields_changed": True,
        }
    manifest = {
        "schema_version": 1,
        "experiment_id": "H52",
        "paper_performance_targets_consumed": False,
        "pe_dependency_model": "paper_static",
        "outputs": outputs,
    }
    manifest_path = output_dir / "mlx-pe-semantics-compile-manifest.json"
    manifest_path.write_text(canonical_json(manifest), encoding="utf-8")
    if args.reference_dir is not None:
        comparisons = {}
        reference_dir = args.reference_dir.resolve()
        for name in ("fixed", "dma"):
            current = output_dir / f"mlx-full-block-{name}.json"
            reference = reference_dir / current.name
            comparisons[name] = {
                "current": digest(current),
                "reference": digest(reference),
                "identical": current.read_bytes() == reference.read_bytes(),
            }
        report = {
            "schema_version": 1,
            "experiment_id": "H52",
            "comparisons": comparisons,
            "all_identical": all(item["identical"] for item in comparisons.values()),
        }
        replay_path = args.replay_check.resolve() if args.replay_check else output_dir / "replay-check.json"
        replay_path.parent.mkdir(parents=True, exist_ok=True)
        replay_path.write_text(canonical_json(report), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
