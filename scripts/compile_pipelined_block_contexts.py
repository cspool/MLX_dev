#!/usr/bin/env python3
"""Compile H109 pipelined tagged-block context scenarios."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from mlxsim.dsagen_overlay import canonical_json
from mlxsim.pipelined_block_contexts import scenarios

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/simulators/pipelined_block_contexts_v1.yaml"
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
    config_root = output_root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    expected_failures = {
        "operand_context_overflow": "DPU operand-context capacity exceeded"
    }
    outputs = {}
    for name, document in sorted(scenarios().items()):
        path = config_root / f"{name}.json"
        path.write_text(canonical_json(document), encoding="utf-8")
        outputs[name] = {
            "artifact": digest(path),
            "expected_failure": expected_failures.get(name),
        }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "paper_performance_targets_consumed": False,
        "outputs": outputs,
    }
    path = output_root / "pipelined-block-contexts-compile-manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "scenarios": len(outputs),
        "expected_failures": sum(
            item["expected_failure"] is not None for item in outputs.values()
        ),
    }
    print(json.dumps(summary, indent=2))
    return 0 if (
        len(outputs) == int(config["execution"]["required_scenarios"])
        and summary["expected_failures"] == 1
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())

