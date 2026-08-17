#!/usr/bin/env python3
"""Run the pre-registered H24 fused two-axis FFT audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlxsim.fig19_fused_fft2d import (
    CONFIG_PATH,
    load_fusion_config,
    run_fused_fft2d_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_fusion_config(args.config)
    output = args.output or Path(config["run"]["output"])
    if output.exists():
        raise FileExistsError(f"refusing to overwrite formal result: {output}")

    report = run_fused_fft2d_audit(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "attention": report["attention_comparison"]["summary"],
                "totals": report["diagnostic_totals"]["comparison"]["summary"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
