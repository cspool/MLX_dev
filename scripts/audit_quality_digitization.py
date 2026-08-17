#!/usr/bin/env python3
"""Derive all Fig. 15/16 quality targets from the pre-registered pixel manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlxsim.quality_digitization import audit_quality_digitization, load_pixel_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/results/fig15-16-quality-targets-run015.json"),
    )
    args = parser.parse_args()

    report = {
        "run_id": "run_015",
        "hypothesis": "H12",
        "protocol": "experiments/h12-quality-target-digitization/protocol.md",
        **audit_quality_digitization(load_pixel_manifest(), verify_sources=True),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0 if report["summary"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
