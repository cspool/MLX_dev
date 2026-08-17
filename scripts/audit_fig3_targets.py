#!/usr/bin/env python3
"""Run the pre-registered H26 full Figure 3 target audit."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from mlxsim.fig3_digitization import (
    CONFIG_PATH,
    PROJECT_ROOT,
    load_fig3_target_config,
    run_fig3_target_completion,
)


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
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_fig3_target_config(args.config)
    output = args.output or PROJECT_ROOT / config["run"]["output"]
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite formal result: {output}")

    report = {"project_git_revision": _git_commit(), **run_fig3_target_completion(config)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "new_qkv_attention_flops_pct": report["derived_targets"][
                    "qkv_attention_flops_pct"
                ],
                "summary": report["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["verdict"] == "supported" else 1


if __name__ == "__main__":
    raise SystemExit(main())
