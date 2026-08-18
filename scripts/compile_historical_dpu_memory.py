#!/usr/bin/env python3
"""Compile H106 historical DPU memory scenarios without paper targets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from mlxsim.dsagen_overlay import canonical_json
from mlxsim.historical_dpu_memory import invalid_relative_address_case, scenarios

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/historical_dpu_memory_v1.yaml"


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
    outputs = {}
    for name, item in sorted(scenarios(config).items()):
        overlay_path = config_root / f"{name}-overlay.json"
        memory_path = config_root / f"{name}-memory.json"
        overlay_path.write_text(canonical_json(item["overlay"]), encoding="utf-8")
        memory_path.write_text(canonical_json(item["memory"]), encoding="utf-8")
        outputs[name] = {
            "overlay": digest(overlay_path),
            "memory": digest(memory_path),
            "expected_failure": item["expected_failure"],
        }
    auxiliary_item = invalid_relative_address_case(config)
    auxiliary_overlay = config_root / "invalid_relative_address-overlay.json"
    auxiliary_memory = config_root / "invalid_relative_address-memory.json"
    auxiliary_overlay.write_text(
        canonical_json(auxiliary_item["overlay"]), encoding="utf-8"
    )
    auxiliary_memory.write_text(
        canonical_json(auxiliary_item["memory"]), encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "paper_performance_targets_consumed": False,
        "source_fixtures": config["fixtures"],
        "outputs": outputs,
        "auxiliary_outputs": {
            "invalid_relative_address": {
                "overlay": digest(auxiliary_overlay),
                "memory": digest(auxiliary_memory),
                "expected_failure": auxiliary_item["expected_failure"],
            }
        },
    }
    path = output_root / "historical-dpu-memory-compile-manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "scenario_count": len(outputs),
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
