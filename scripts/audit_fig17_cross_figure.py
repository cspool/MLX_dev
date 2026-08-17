#!/usr/bin/env python3
"""Audit whether Fig. 3 H100 profiles reconstruct Fig. 17 prefill speedup."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from mlxsim.fig17_consistency import audit_fig17_cross_figure, load_yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/fig17_cross_figure_v1.yaml"


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
    config = load_yaml(args.config)
    output = args.output or PROJECT_ROOT / config["run"]["output"]
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    if output.exists():
        raise SystemExit(f"refusing to overwrite official result: {output}")
    audit = audit_fig17_cross_figure(config)
    report = {
        "run_id": config["run"]["id"],
        "hypothesis": config["run"]["hypothesis"],
        "protocol": config["run"]["protocol"],
        "git_commit": _git_commit(),
        "config": config,
        **audit,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0 if report["summary"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
