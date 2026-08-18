#!/usr/bin/env python3
"""Freeze H102 path contracts needed for the corrected H110 recompile."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
H102_MANIFEST = (
    PROJECT_ROOT
    / "artifacts/environment/h102/fig24-25-full-mesh-compile-manifest.json"
)
H102_RESULT = (
    PROJECT_ROOT / "artifacts/results/fig24-25-full-mesh-paths-run107.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts/source-snapshots/h110-pipelined-full-mesh-contracts-20260818.json"
)


def digest(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def build_snapshot() -> dict[str, Any]:
    manifest = json.loads(H102_MANIFEST.read_text(encoding="utf-8"))
    result = json.loads(H102_RESULT.read_text(encoding="utf-8"))
    contracts = manifest["path_contracts"]
    if set(contracts) != set(result["full_estimates"]):
        raise ValueError("H102 contract/result key mismatch")
    return {
        "schema_version": 1,
        "experiment_id": "H110",
        "classification": "target_free_h102_contract_snapshot",
        "paper_performance_targets_consumed": False,
        "sources": {
            "h102_manifest": digest(H102_MANIFEST),
            "h102_result": digest(H102_RESULT),
        },
        "path_count": len(contracts),
        "contracts": contracts,
        "old_full_estimates": result["full_estimates"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    canonical = json.dumps(build_snapshot(), indent=2, sort_keys=True) + "\n"
    if args.verify:
        matches = args.output.read_text(encoding="utf-8") == canonical
        print(json.dumps({"matches": matches}))
        return 0 if matches else 1
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical, encoding="utf-8")
    print(json.dumps({"output": str(args.output), "bytes": len(canonical)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

