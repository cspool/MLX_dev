#!/usr/bin/env python3
"""Extract the compact H102 full-work basis required by H107."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
H102_RESULT = (
    PROJECT_ROOT / "artifacts/results/fig24-25-full-mesh-paths-run107.json"
)
H102_MANIFEST = (
    PROJECT_ROOT
    / "artifacts/environment/h102/fig24-25-full-mesh-compile-manifest.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts/source-snapshots/h107-full-mesh-memory-contracts-20260818.json"
)


def digest(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def build_snapshot() -> dict[str, Any]:
    result = json.loads(H102_RESULT.read_text(encoding="utf-8"))
    manifest = json.loads(H102_MANIFEST.read_text(encoding="utf-8"))
    paths = {}
    for key, contract in sorted(manifest["path_contracts"].items()):
        work = result["scale_work"][f"{key}-q4"]
        estimate = result["full_estimates"][key]
        if not work["pass"] or not result["full_work_checks"][key]:
            raise ValueError(f"H102 full-work gate failed: {key}")
        paths[key] = {
            "family": contract["family"],
            "case": contract["case"],
            "operator": contract["operator"],
            "actual": contract["actual"],
            "full_scale": work["full_scale"],
            "full_fu_counts": work["expected_fu"],
            "full_load_bytes": work["expected_load_bytes"],
            "full_store_bytes": work["expected_store_bytes"],
            "h102_full_cycles": estimate["cycles"],
            "h102_physical_fma_pe_cycles": estimate[
                "physical_fma_pe_cycles"
            ],
            "h102_fma_utilization": estimate["fma_utilization"],
        }
    return {
        "schema_version": 1,
        "experiment_id": "H107",
        "classification": "target_free_compact_h102_memory_basis",
        "paper_performance_targets_consumed": False,
        "sources": {
            "h102_result": digest(H102_RESULT),
            "h102_compile_manifest": digest(H102_MANIFEST),
        },
        "path_count": len(paths),
        "family_counts": {
            family: sum(item["family"] == family for item in paths.values())
            for family in ("fft", "qkv_bsmm", "swa")
        },
        "paths": paths,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    snapshot = build_snapshot()
    canonical = json.dumps(snapshot, indent=2, sort_keys=True) + "\n"
    if args.verify:
        matches = args.output.read_text(encoding="utf-8") == canonical
        print(json.dumps({"matches": matches, "paths": snapshot["path_count"]}))
        return 0 if matches else 1
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.relative_to(PROJECT_ROOT)),
                "paths": snapshot["path_count"],
                "family_counts": snapshot["family_counts"],
            },
            indent=2,
        )
    )
    return 0 if snapshot["path_count"] == 48 else 1


if __name__ == "__main__":
    raise SystemExit(main())

