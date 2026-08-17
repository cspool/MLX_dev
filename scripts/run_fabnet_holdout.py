#!/usr/bin/env python3
"""Run pre-registered H13 against the pinned official FABNet simulator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlxsim.fabnet_audit import (
    DEFAULT_FABNET_ROOT,
    audit_fig19_digitization,
    compare_fabnet_results,
    inspect_fabnet_checkout,
    load_fig19_manifest,
    run_fabnet_simulator,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fabnet-root", type=Path, default=DEFAULT_FABNET_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/results/fig19-fabnet-run016.json"),
    )
    args = parser.parse_args()

    manifest = load_fig19_manifest()
    digitization = audit_fig19_digitization(manifest, verify_source=True)
    checkout = inspect_fabnet_checkout(args.fabnet_root)
    if not digitization["summary"]["pass"]:
        raise RuntimeError("Fig. 19 source or annotation cross-check failed")
    if not checkout["pass"]:
        raise RuntimeError(f"FABNet checkout failed pin/cleanliness check: {checkout}")

    simulator_results = run_fabnet_simulator(
        digitization["targets"]["sequence_lengths"], repo_root=args.fabnet_root
    )
    comparison = compare_fabnet_results(digitization["targets"], simulator_results)
    report = {
        "run_id": "run_016",
        "hypothesis": "H13",
        "protocol": "experiments/h13-fabnet-open-simulator/protocol.md",
        "classification": "external-open-simulator-holdout",
        "validation_eligible": True,
        "digitization": digitization,
        "upstream_checkout": checkout,
        "frozen_configuration": {
            "model_version": "large",
            "num_layers": 24,
            "hidden_dim": 1024,
            "ffn_inner_dim": 4096,
            "head_dim": 32,
            "frequency_mhz": 200,
            "implementation_efficiency": 0.85,
            "fpga_board": "zcu128",
            "offchip_memory": "hbm",
            "parallel_butterfly_units_per_engine": 4,
            "parallel_butterfly_engines": 40,
        },
        "comparison": comparison,
        "verdict": "supported" if comparison["summary"]["all_points_pass"] else "rejected",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": report["verdict"], **comparison["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

