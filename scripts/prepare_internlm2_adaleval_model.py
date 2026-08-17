#!/usr/bin/env python3
"""Build H30's read-only historical model view from qualified sources."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlxsim.adaleval import prepare_historical_view

DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/internlm2_adaleval_v1.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    model = config["model"]
    report = prepare_historical_view(
        history_root=PROJECT_ROOT / model["source_history_root"],
        binary_root=PROJECT_ROOT / model["downloaded_weight_root"],
        view_root=PROJECT_ROOT / model["historical_view_root"],
        revision=model["historical_revision"],
        historical_files=model["historical_files"],
        binary_files=model["binary_files"],
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
