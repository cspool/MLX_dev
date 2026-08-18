#!/usr/bin/env python3
"""Compose target-free Xavier and MLX Attention totals for H135."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/xavier_attention_composition_v1.yaml"


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    h133 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h133"]["path"]).read_text()
    )
    h134 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h134"]["path"]).read_text()
    )
    h83 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h83"]["path"]).read_text()
    )
    parent_checks = {
        name: parent["hypothesis_status"] == spec["required_status"]
        and parent["audit_integrity"] is spec["required_integrity"]
        for name, parent in (("h133", h133), ("h134", h134), ("h83", h83))
        for spec in [config["frozen_inputs"][name]]
    }
    xavier_clock = int(config["composition"]["xavier_clock_hz"])
    mlx_clock = int(config["composition"]["mlx_clock_hz"])
    compositions: dict[str, Any] = {}
    component_checks: dict[str, bool] = {}
    finite_checks: dict[str, bool] = {}
    for shape, mapping in config["shapes"].items():
        fft = h133["full_estimates"][mapping["xavier_fft"]]
        non_fft = h134["component_estimates"][shape]
        components = {
            "fftcmp": float(fft["cycles"]),
            "qk": float(non_fft["qk"]["cycles"]),
            "softmax": float(non_fft["softmax"]["cycles"]),
            "sv": float(non_fft["sv"]["cycles"]),
        }
        xavier_cycles = sum(components.values())
        mlx_cycles = float(
            h83["models"][mapping["mlx_model"]]["full_work_predicted_cycles"]
        )
        xavier_seconds = xavier_cycles / xavier_clock
        mlx_seconds = mlx_cycles / mlx_clock
        speedup = xavier_seconds / mlx_seconds
        component_checks[shape] = (
            fft["eligible"] is True
            and all(item["eligible"] for item in non_fft.values())
            and set(components) == set(config["composition"]["xavier_components"])
            and abs(xavier_cycles - sum(components.values())) < 1e-9
        )
        finite_checks[shape] = all(
            math.isfinite(value) and value > 0
            for value in (
                *components.values(),
                xavier_cycles,
                mlx_cycles,
                xavier_seconds,
                mlx_seconds,
                speedup,
            )
        )
        compositions[shape] = {
            "xavier_components": components,
            "xavier_total_cycles": xavier_cycles,
            "xavier_clock_hz": xavier_clock,
            "xavier_total_seconds": xavier_seconds,
            "mlx_model": mapping["mlx_model"],
            "mlx_cycles": mlx_cycles,
            "mlx_clock_hz": mlx_clock,
            "mlx_seconds": mlx_seconds,
            "speedup": speedup,
            "speedup_definition": config["composition"]["speedup"],
            "xavier_mapping_claim": "transparent_proxy_not_author_cuda",
        }
    clock_checks = {
        "xavier": xavier_clock == 1_377_000_000,
        "mlx": mlx_clock == 1_000_000_000,
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    source_text = "\n".join(
        (PROJECT_ROOT / path).read_text(errors="replace")
        for path in config["source_layout"].values()
    )
    target_free_checks = {
        "config": config["acceptance"]["targets_consumed"] is False,
        "no_target": "fig20" + "_speedup" not in source_text,
        "no_fit": "fit" + "_affine" not in source_text,
        "no_factor": "component" + "_factor" not in source_text,
        "no_overlap": "overlap" + "_cycles" not in source_text,
    }
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        len(compositions) == int(config["acceptance"]["required_shapes"]),
        all(component_checks.values()),
        all(
            compositions[shape]["mlx_cycles"]
            == h83["models"][mapping["mlx_model"]]["full_work_predicted_cycles"]
            for shape, mapping in config["shapes"].items()
        ),
        all(clock_checks.values()),
        all(finite_checks.values()),
        all(
            set(item["xavier_components"])
            == set(config["composition"]["xavier_components"])
            for item in compositions.values()
        ),
        all(target_free_checks.values())
        and all(item["pass"] for item in source_files.values()),
        config["acceptance"]["author_cuda_mapping_claimed"] is False,
        config["validation_eligible"] is False,
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "components": all(component_checks.values()),
        "finite": all(finite_checks.values()),
        "clocks": all(clock_checks.values()),
        "source": all(target_free_checks.values())
        and all(item["pass"] for item in source_files.values()),
        "acceptance_evaluated": len(acceptance_gates) == 10
        and all(isinstance(value, bool) for value in acceptance_gates),
    }
    integrity = all(integrity_checks.values())
    supported = integrity and all(acceptance_gates)
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "classification": config["classification"],
        "validation_eligible": config["validation_eligible"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_commit": git_commit(),
        "hypothesis_status": "supported" if supported else "rejected",
        "audit_integrity": integrity,
        "paper_performance_targets_consumed": False,
        "paper_reproduction_claim": "none_target_free_attention_composition_only",
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "component_checks": component_checks,
        "finite_checks": finite_checks,
        "clock_checks": clock_checks,
        "compositions": compositions,
        "target_free_checks": target_free_checks,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "shapes": len(compositions),
            "eligible_xavier_components": 8,
            "speedups": {
                shape: item["speedup"] for shape, item in compositions.items()
            },
            "acceptance_gates_passed": sum(acceptance_gates),
            "acceptance_gates_total": len(acceptance_gates),
            "active_simulator_figures_reproduced": 0,
            "active_simulator_figures_total": 8,
        },
        "source_files": source_files,
        "integrity_checks": integrity_checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    report = build_audit(config)
    output = PROJECT_ROOT / config["result_path"]
    if args.verify_existing:
        existing = json.loads(output.read_text())
        keys = (
            "hypothesis_status",
            "audit_integrity",
            "compositions",
            "acceptance_gates",
            "summary",
            "integrity_checks",
        )
        matches = all(
            json.dumps(existing.get(key), sort_keys=True)
            == json.dumps(report.get(key), sort_keys=True)
            for key in keys
        )
        print(json.dumps({"existing_matches": matches, **report}, indent=2))
        return 0 if matches else 1
    if args.preflight_only:
        print(json.dumps(report, indent=2))
        return 0 if report["audit_integrity"] and not output.exists() else 1
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["hypothesis_status"], **report["summary"]}, indent=2))
    return 0 if report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
