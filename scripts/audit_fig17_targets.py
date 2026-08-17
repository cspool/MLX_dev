#!/usr/bin/env python3
"""Derive and audit the frozen Fig. 17 H100 speedup targets."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from mlxsim.fig17_digitization import audit_fig17_digitization, load_pixel_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/fig17_target_audit_v1.yaml"


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={PROJECT_ROOT}", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = _load_config(args.config)
    output = args.output or PROJECT_ROOT / config["run"]["output"]
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    if output.exists():
        raise SystemExit(f"refusing to overwrite official result: {output}")

    manifest = load_pixel_manifest(PROJECT_ROOT / config["input"]["manifest"])
    audit = audit_fig17_digitization(manifest, verify_source=True)
    report = {
        "run_id": config["run"]["id"],
        "hypothesis": config["run"]["hypothesis"],
        "protocol": config["run"]["protocol"],
        "git_commit": _git_commit(),
        **audit,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0 if report["summary"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
