#!/usr/bin/env python3
"""Compile the target-free H84 Xavier job manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/xavier_matched_attention_v1.yaml"


def digest(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def work(family: str, count: int, shape: dict[str, Any]) -> dict[str, int]:
    if family == "fftcmp":
        pairs = count * int(shape["forward_stages"])
        pairs += count // 2 * int(shape["inverse_stages"])
        return {"fma": 4 * pairs, "add": 6 * pairs, "shuffle": count}
    if family == "qk":
        return {"fma": count * int(shape["hidden_dimension"])}
    if family == "softmax":
        elements = count * int(shape["retained_length"])
        return {"fmax": elements, "fexp": elements, "add": elements}
    if family == "sv":
        return {
            "fma": count * int(shape["retained_length"]),
            "fdiv": count,
        }
    raise ValueError(f"unsupported family: {family}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_root = PROJECT_ROOT / config["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    jobs = []
    for shape_name, shape in config["shapes"].items():
        for family, ranges in config["families"].items():
            counts = [*ranges["fit_counts"], *ranges["holdout_counts"]]
            for count_value in counts:
                count = int(count_value)
                parameter = (
                    int(shape["forward_stages"])
                    if family == "fftcmp"
                    else int(
                        shape[
                            "hidden_dimension"
                            if family == "qk"
                            else "retained_length"
                        ]
                    )
                )
                parameter2 = int(shape["inverse_stages"]) if family == "fftcmp" else 0
                jobs.append(
                    {
                        "name": f"{shape_name}-{family}-c{count}",
                        "shape": shape_name,
                        "family": family,
                        "count": count,
                        "parameter": parameter,
                        "parameter2": parameter2,
                        "work": work(family, count, shape),
                    }
                )
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "paper_performance_targets_consumed": False,
        "device_clock_hz": int(config["device_clock_hz"]),
        "cuda_source": digest(PROJECT_ROOT / config["source_layout"]["cuda_source"]),
        "jobs": jobs,
    }
    path = output_root / "xavier-attention-compile-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if len(jobs) == 32 else 1


if __name__ == "__main__":
    raise SystemExit(main())
