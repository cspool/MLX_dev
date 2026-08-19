#!/usr/bin/env python3
"""Expand all Figure19/20/23 shapes and multi-layer plans through one CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from mlxsim.workload_coverage import expand_coverage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/full_workload_coverage_v1.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    frozen = config["frozen_inputs"]
    spec = yaml.safe_load((PROJECT_ROOT / frozen["coverage_spec"]["path"]).read_text())
    manifest = expand_coverage(
        spec=spec,
        physical_compile=json.loads(
            (PROJECT_ROOT / frozen["physical_compile"]["path"]).read_text()
        ),
        source_manifest=json.loads(
            (PROJECT_ROOT / frozen["source_manifest"]["path"]).read_text()
        ),
        source_config=yaml.safe_load(
            (PROJECT_ROOT / frozen["source_config"]["path"]).read_text()
        ),
        coupled_config=yaml.safe_load(
            (PROJECT_ROOT / frozen["coupled_config"]["path"]).read_text()
        ),
        output_root=PROJECT_ROOT / config["output_root"],
        project_root=PROJECT_ROOT,
    )
    manifest.update(
        experiment_id=config["experiment_id"],
        run_id=config["run_id"],
        paper_performance_targets_consumed=True,
        single_entrypoint=True,
    )
    path = PROJECT_ROOT / config["coverage_manifest"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"units": len(manifest["units"]), "formats": manifest["format_counts"], "checks": manifest["checks"]}, indent=2))
    return 0 if all(manifest["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
