#!/usr/bin/env python3
"""Lower a unified MLX workload suite to native simulator execution formats."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from mlxsim.workload_lowering import lower_suite

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/unified_workload_lowering_v1.yaml"


def load_document(path: Path) -> Any:
    return yaml.safe_load(path.read_text()) if path.suffix in {".yaml", ".yml"} else json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    frozen = config["frozen_inputs"]
    spec = load_document(PROJECT_ROOT / frozen["workload_spec"]["path"])
    contexts = {
        reference: load_document(PROJECT_ROOT / frozen[input_name]["path"])
        for reference, input_name in config["context_refs"].items()
    }
    manifest = lower_suite(
        spec=spec,
        contexts=contexts,
        output_root=PROJECT_ROOT / config["output_root"],
        project_root=PROJECT_ROOT,
    )
    manifest.update(
        experiment_id=config["experiment_id"],
        run_id=config["run_id"],
        source_spec=frozen["workload_spec"],
        paper_performance_targets_consumed=True,
    )
    path = PROJECT_ROOT / config["lowering_manifest"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "graphs": len(manifest["topological_orders"]),
                "units": len(manifest["units"]),
                "formats": manifest["format_counts"],
                "checks": manifest["checks"],
            },
            indent=2,
        )
    )
    return 0 if all(manifest["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
