#!/usr/bin/env python3
"""Record deterministic equality of two H44 compiler output directories."""

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
    primary_manifest = args.primary / "fig22-compile-manifest.json"
    replay_manifest = args.replay / "fig22-compile-manifest.json"
    manifest = json.loads(primary_manifest.read_text(encoding="utf-8"))
    files = ["fig22-compile-manifest.json"] + [
        item["path"] for item in (manifest.get("outputs") or {}).values()
    ]
    checks = {
        name: (args.replay / name).is_file()
        and digest(args.primary / name) == digest(args.replay / name)
        for name in files
    }
    report = {
        "schema_version": 1,
        "file_count": len(files),
        "checks": checks,
        "all_identical": all(checks.values()),
        "primary_manifest_sha256": digest(primary_manifest),
        "replay_manifest_sha256": digest(replay_manifest),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["all_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
