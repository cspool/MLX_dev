#!/usr/bin/env python3
"""Run the pre-registered H22 Fig. 19 component holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlxsim.fig19_components import (
    CONFIG_PATH,
    load_component_config,
    run_fig19_component_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_component_config(args.config)
    output = args.output or Path(config["run"]["output"])
    if output.exists():
        raise FileExistsError(f"refusing to overwrite formal result: {output}")

    report = run_fig19_component_audit(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": report["verdict"], **report["comparison"]["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
