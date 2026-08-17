#!/usr/bin/env python3
"""Derive an RTX3090 GPGPU-Sim config from the tested SM86 RTX3070 model."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/gpgpusim_rtx3090_proxy_v1.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/environment/h51/config"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def digest(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def derive_config(text: str) -> tuple[str, dict[str, int]]:
    replacements = {
        "clusters": ("-gpgpu_n_clusters 46", "-gpgpu_n_clusters 82", 1),
        "memory_partitions": ("-gpgpu_n_mem 16", "-gpgpu_n_mem 24", 1),
        "clocks": (
            "-gpgpu_clock_domains 1132:1132:1132:3500.5",
            "-gpgpu_clock_domains 1695:1695:1695:5250",
            1,
        ),
    }
    counts: dict[str, int] = {}
    result = text
    for name, (source, target, expected_count) in replacements.items():
        count = result.count(source)
        if count != expected_count:
            raise ValueError(f"{name} source count is {count}, expected {expected_count}")
        result = result.replace(source, target)
        counts[name] = count
    return result, counts


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base = PROJECT_ROOT / config["source"]["base_config"]
    interconnect = PROJECT_ROOT / config["source"]["interconnect"]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    derived, counts = derive_config(base.read_text(encoding="utf-8"))
    output_config = output_dir / "gpgpusim.config"
    output_config.write_text(derived, encoding="utf-8")
    output_icnt = output_dir / interconnect.name
    shutil.copy2(interconnect, output_icnt)
    manifest = {
        "schema_version": 1,
        "experiment_id": "H51",
        "paper_performance_targets_consumed": False,
        "base": digest(base),
        "derived": digest(output_config),
        "interconnect_source": digest(interconnect),
        "interconnect_copy": digest(output_icnt),
        "replacement_counts": counts,
        "registered_substitutions": config["registered_substitutions"],
    }
    manifest_path = output_dir / "rtx3090-config-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
