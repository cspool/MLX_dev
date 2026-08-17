#!/usr/bin/env python3
"""Transform frozen Figure 22/23 configs to paper-static PE semantics."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

from mlxsim.dsagen_overlay import canonical_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/environment/h59"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fig22-parent", type=Path, default=PROJECT_ROOT / "artifacts/environment/h44")
    parser.add_argument("--fig23-parent", type=Path, default=PROJECT_ROOT / "artifacts/environment/h46")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def transform(document: dict) -> dict:
    result = copy.deepcopy(document)
    result["pe_dependency_model"] = "paper_static"
    result.setdefault("metadata", {})["pe_dependency_model"] = "paper_static"
    result["metadata"]["scoreboard_is_paper_semantics"] = False
    return result


def digest(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    resolved = path.resolve()
    try:
        display_path = resolved.relative_to(PROJECT_ROOT)
    except ValueError:
        display_path = resolved
    return {
        "path": str(display_path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    records = {}
    for figure, parent, pattern in (
        ("fig22", args.fig22_parent.resolve(), "fig22-*-*.json"),
        ("fig23", args.fig23_parent.resolve(), "fig23-*.json"),
    ):
        target = output / figure
        target.mkdir(parents=True, exist_ok=True)
        items = []
        for source in sorted(parent.glob(pattern)):
            if "manifest" in source.name or "check" in source.name:
                continue
            destination = target / source.name
            document = json.loads(source.read_text(encoding="utf-8"))
            destination.write_text(canonical_json(transform(document)), encoding="utf-8")
            items.append({"parent": digest(source), "output": digest(destination)})
        records[figure] = items
    if len(records["fig22"]) != 16 or len(records["fig23"]) != 20:
        raise ValueError("unexpected Figure 22/23 config count")
    shutil.copy2(
        args.fig23_parent.resolve() / "fig23-compile-manifest.json",
        output / "fig23/fig23-compile-manifest.json",
    )
    manifest = {
        "schema_version": 1,
        "experiment_id": "H59",
        "paper_target_values_consumed": False,
        "pe_dependency_model": "paper_static",
        "records": records,
    }
    (output / "paper-static-fig22-23-compile-manifest.json").write_text(
        canonical_json(manifest), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
