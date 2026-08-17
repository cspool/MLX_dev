#!/usr/bin/env python3
"""Run the pre-registered H21 FGSCR-42 public-input audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlxsim.fgscr42_audit import DEFAULT_CONFIG, load_input_audit_config, run_input_audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = load_input_audit_config(args.config)
    output = args.output or Path(config["run"]["output"])
    if output.exists():
        raise FileExistsError(f"refusing to overwrite formal result: {output}")

    report = run_input_audit(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "run_id": report["run_id"],
                "verdict": report["decision"]["verdict"],
                "input_sufficient": report["decision"]["input_sufficient"],
                "missing_required_inputs_fraction": report["decision"][
                    "missing_required_inputs_fraction"
                ],
                "audit_integrity_pass": report["audit_integrity"]["pass"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
