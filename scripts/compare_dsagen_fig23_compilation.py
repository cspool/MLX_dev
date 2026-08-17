#!/usr/bin/env python3
"""Record deterministic equality of two H46 compiler directories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("primary", type=Path)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = parse_args()
    manifest_path = args.primary / "fig23-compile-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    names = ["fig23-compile-manifest.json"] + [
        item["path"] for item in manifest["outputs"].values()
    ]
    checks = {
        name: (args.replay / name).is_file()
        and digest(args.primary / name) == digest(args.replay / name)
        for name in names
    }
    report = {
        "schema_version": 1,
        "file_count": len(names),
        "checks": checks,
        "all_identical": all(checks.values()),
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["all_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
