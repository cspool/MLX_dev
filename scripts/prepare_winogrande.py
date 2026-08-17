#!/usr/bin/env python3
"""Download and qualify the pinned WinoGrande-xl validation parquet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import hf_hub_download

from mlxsim.winogrande import qualify_parquet_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/llama2_winogrande_v1.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> int:
    args = _parse_args()
    config = _load_yaml(args.config)
    dataset = config["dataset"]
    destination = PROJECT_ROOT / dataset["qualification_path"]

    if destination.exists():
        qualification = qualify_parquet_dataset(destination, dataset)
        if not qualification["pass"]:
            raise SystemExit(f"existing dataset file failed qualification: {qualification}")
    else:
        local_dir = destination.parents[1]
        downloaded = Path(
            hf_hub_download(
                repo_id="allenai/winogrande",
                repo_type="dataset",
                filename=dataset["official_relative_path"],
                revision=dataset["official_revision"],
                local_dir=local_dir,
                token=False,
            )
        )
        if downloaded.resolve() != destination.resolve():
            raise RuntimeError(f"unexpected download path: {downloaded}")
        qualification = qualify_parquet_dataset(destination, dataset)
        if not qualification["pass"]:
            raise RuntimeError(f"downloaded dataset failed qualification: {qualification}")

    print(json.dumps(qualification, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
