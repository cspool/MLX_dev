#!/usr/bin/env python3
"""Compile 192 corrected H102 paths into H109 pipelined mode."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from mlxsim.dsagen_overlay import canonical_json
from mlxsim.pipelined_full_mesh_paths import compile_pipelined_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/pipelined_full_mesh_paths_v1.yaml"


def digest(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def execution_semantics(document: dict[str, Any]) -> dict[str, Any]:
    return {
        key: document[key]
        for key in (
            "active_window",
            "memory_backend",
            "register_file",
            "pipelines",
            "functional_units",
            "routing",
            "blocks",
        )
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    snapshot = json.loads(
        (
            PROJECT_ROOT / config["frozen_inputs"]["contracts"]["path"]
        ).read_text(encoding="utf-8")
    )
    output_root = PROJECT_ROOT / config["output_root"]
    config_root = output_root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    scales = [*config["fit_scales"], *config["holdout_scales"]]
    outputs = {}
    checks = {}
    for path_key, contract in snapshot["contracts"].items():
        for scale_value in scales:
            scale = int(scale_value)
            run_key = f"{path_key}-q{scale}"
            document, metadata, original = compile_pipelined_path(
                run_key=run_key,
                contract=contract,
                scale=scale,
                active_window=int(config["hardware"]["active_window"]),
                contexts=int(
                    config["hardware"]["iteration_contexts_per_block"]
                ),
                operand_contexts_per_pe=int(
                    config["hardware"]["operand_contexts_per_pe"]
                ),
            )
            path = config_root / f"{run_key}.json"
            path.write_text(canonical_json(document), encoding="utf-8")
            outputs[run_key] = {"artifact": digest(path), "metadata": metadata}
            checks[run_key] = {
                "semantic_identity": execution_semantics(document)
                == execution_semantics(original),
                "mode": document["pe_dependency_model"] == "dpu_pipelined",
                "contexts": document["dpu"]["iteration_contexts_per_block"]
                == int(config["hardware"]["iteration_contexts_per_block"]),
                "operand_capacity": document["dpu"]["operand_contexts_per_pe"]
                == int(config["hardware"]["operand_contexts_per_pe"]),
            }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "paper_performance_targets_consumed": False,
        "path_contracts": snapshot["contracts"],
        "outputs": outputs,
        "checks": checks,
    }
    path = output_root / "pipelined-full-mesh-compile-manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "paths": len(snapshot["contracts"]),
        "configs": len(outputs),
        "all_checks": all(all(item.values()) for item in checks.values()),
    }
    print(json.dumps(summary, indent=2))
    return 0 if (
        summary["paths"] == int(config["required"]["paths"])
        and summary["configs"] == int(config["required"]["configs"])
        and summary["all_checks"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())

