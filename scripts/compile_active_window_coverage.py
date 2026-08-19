#!/usr/bin/env python3
"""Compile every H165 active-window/workload pair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from mlxsim.active_window_coverage import compile_active_window_path
from mlxsim.dsagen_overlay import canonical_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/simulators/active_window_coverage_sweep_v1.yaml"
)


def digest(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    h120_config = yaml.safe_load(
        (PROJECT_ROOT / config["frozen_inputs"]["h120_config"]["path"]).read_text()
    )
    h118_config = yaml.safe_load(
        (PROJECT_ROOT / config["frozen_inputs"]["h118_config"]["path"]).read_text()
    )
    output_root = PROJECT_ROOT / config["output_root"]
    outputs: dict[str, Any] = {}
    for window in config["window_sweep"]["compiled_windows"]:
        for operator in config["workloads"]["operators"]:
            for size in config["workloads"]["sizes"]:
                overlay, memory, metadata = compile_active_window_path(
                    operator,
                    int(size),
                    int(window),
                    config,
                    h120_config,
                    h118_config,
                )
                key = metadata["key"]
                overlay_path = output_root / f"configs/overlay/w{window}/{operator}-{size}.json"
                memory_path = output_root / f"configs/memory/w{window}/{operator}-{size}.json"
                overlay_path.parent.mkdir(parents=True, exist_ok=True)
                memory_path.parent.mkdir(parents=True, exist_ok=True)
                overlay_path.write_text(canonical_json(overlay))
                memory_path.write_text(canonical_json(memory))
                outputs[key] = {
                    "overlay": digest(overlay_path),
                    "memory": digest(memory_path),
                    "metadata": metadata,
                }
    maxima = {
        str(window): max(
            item["metadata"]["footprint"]
            for item in outputs.values()
            if int(item["metadata"]["window"]) == int(window)
        )
        for window in config["window_sweep"]["compiled_windows"]
    }
    global_feasibility = {
        str(window): all(
            item["metadata"]["path_capacity_feasible"]
            for item in outputs.values()
            if int(item["metadata"]["window"]) == int(window)
        )
        for window in config["window_sweep"]["compiled_windows"]
    }
    expected_maxima = {
        str(key): int(value)
        for key, value in config["window_sweep"][
            "expected_max_footprint_by_window"
        ].items()
    }
    expected_feasible = {
        str(window): int(window)
        in {int(value) for value in config["window_sweep"]["globally_feasible_windows"]}
        for window in config["window_sweep"]["compiled_windows"]
    }
    checks = {
        "output_count": len(outputs) == 128,
        "metadata": all(
            all(item["metadata"]["checks"].values()) for item in outputs.values()
        ),
        "footprint_maxima": maxima == expected_maxima,
        "global_feasibility": global_feasibility == expected_feasible,
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
        "maximum_footprint_by_window": maxima,
        "global_feasibility_by_window": global_feasibility,
        "checks": checks,
    }
    path = output_root / "active-window-compile-manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"outputs": len(outputs), "checks": checks}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
