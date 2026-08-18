#!/usr/bin/env python3
"""Compile H105 source-derived DPU fixtures and semantic scenarios."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from mlxsim.dsagen_overlay import canonical_json
from mlxsim.simict_dpu_contract import historical_fixtures, semantic_scenarios

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/simict_dpu_contract_v1.yaml"


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
    documents = semantic_scenarios()
    documents.update(historical_fixtures(config["fixtures"]))
    expected_failures = {
        "instruction_slot_overflow": "DPU instruction-slot capacity exceeded",
        "operand_context_overflow": "DPU operand-context capacity exceeded",
    }
    output_root = PROJECT_ROOT / config["output_root"]
    config_root = output_root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, document in sorted(documents.items()):
        path = config_root / f"{name}.json"
        path.write_text(canonical_json(document), encoding="utf-8")
        outputs[name] = {
            "artifact": digest(path),
            "expected_failure": expected_failures.get(name),
            "metadata": document["metadata"],
        }
    fixture_checks = {
        name: documents[name]["metadata"]["source_contract"] == specification
        for name, specification in config["fixtures"].items()
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "paper_performance_targets_consumed": False,
        "outputs": outputs,
        "fixture_checks": fixture_checks,
        "expected_failures": expected_failures,
    }
    path = output_root / "simict-dpu-compile-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "scenario_count": len(outputs),
        "fixture_count": len(fixture_checks),
        "all_fixtures_match": all(fixture_checks.values()),
    }
    print(json.dumps(summary, indent=2))
    return 0 if (
        len(outputs) == int(config["execution"]["required_scenarios"])
        and len(fixture_checks) == 3
        and all(fixture_checks.values())
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
