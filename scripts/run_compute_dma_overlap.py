#!/usr/bin/env python3
"""Run the H108 two-resource overlap envelope twice."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from mlxsim.compute_dma_overlap import compose_point

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/compute_dma_overlap_v1.yaml"


def digest(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def build_points(config: dict[str, Any]) -> list[dict[str, Any]]:
    h102 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h102"]["path"]).read_text()
    )
    h107 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h107"]["path"]).read_text()
    )
    points = []
    for key, path in sorted(h107["path_results"].items()):
        cycles = h102["full_estimates"][key]["cycles"]
        if float(cycles) != int(cycles):
            raise ValueError(f"H102 full cycles are not integral: {key}")
        for bandwidth in config["hardware"]["bandwidth_sweep_bytes_per_cycle"]:
            points.append(
                compose_point(
                    key=key,
                    h102_cycles=int(cycles),
                    path=path,
                    bandwidth=int(bandwidth),
                    peak_effective_ops_per_cycle=float(
                        config["hardware"]["peak_effective_ops_per_cycle"]
                    ),
                )
            )
    return points


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_root = PROJECT_ROOT / config["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    replay_files = []
    for replay in range(1, int(config["execution"]["deterministic_replays"]) + 1):
        payload = {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "paper_performance_targets_consumed": False,
            "selected_mlx_bandwidth_bytes_per_cycle": None,
            "figure25_reproductions": None,
            "points": build_points(config),
        }
        path = output_root / f"compute-dma-overlap-r{replay}.json"
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        replay_files.append(digest(path))
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "paper_performance_targets_consumed": False,
        "replays": replay_files,
        "checks": {
            "replay_count": len(replay_files)
            == int(config["execution"]["deterministic_replays"]),
            "point_count": all(
                json.loads((PROJECT_ROOT / item["path"]).read_text())["points"]
                and len(
                    json.loads(
                        (PROJECT_ROOT / item["path"]).read_text()
                    )["points"]
                )
                == int(config["execution"]["required_paths"])
                * int(config["execution"]["required_bandwidth_points"])
                for item in replay_files
            ),
            "deterministic": len({item["sha256"] for item in replay_files}) == 1,
            "all_points": all(
                all(point["checks"].values())
                for item in replay_files
                for point in json.loads(
                    (PROJECT_ROOT / item["path"]).read_text()
                )["points"]
            ),
        },
    }
    path = output_root / "compute-dma-overlap-run-manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["checks"], indent=2))
    return 0 if all(manifest["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
