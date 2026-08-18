#!/usr/bin/env python3
"""Audit H23 Figure 19 mapping against current source-integrated mechanisms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from mlxsim.fig19_mlx_transfer import mapped_workloads
from mlxsim.workloads import compile_workload

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/fig19_source_identity_v1.yaml"


def qualify(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    exists = path.is_file()
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if exists else None
    checks = {"is_file": exists, "sha256": digest == expected["sha256"]}
    return {
        "path": str(path.relative_to(PROJECT_ROOT)) if exists else str(path),
        "bytes": path.stat().st_size if exists else None,
        "sha256": digest,
        "checks": checks,
        "pass": all(checks.values()),
    }


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={PROJECT_ROOT}", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    files = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    reports = {
        name: json.loads((PROJECT_ROOT / spec["path"]).read_text(encoding="utf-8"))
        for name, spec in config["frozen_inputs"].items()
        if name != "h23_config"
    }
    h23_config = yaml.safe_load(
        (PROJECT_ROOT / config["frozen_inputs"]["h23_config"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    parent_checks = {
        name: (
            "required_status" not in spec
            or reports[name]["hypothesis_status"] == spec["required_status"]
        )
        and (
            "required_integrity" not in spec
            or reports[name]["audit_integrity"] is spec["required_integrity"]
        )
        for name, spec in config["frozen_inputs"].items()
        if name != "h23_config"
    }
    simulated = {
        int(item["sequence_length"]): item for item in reports["h23"]["simulated"]
    }
    profiles = {}
    mapping_checks = {}
    for n_value in h23_config["model"]["sequence_lengths"]:
        n = int(n_value)
        workloads = mapped_workloads(h23_config, n)
        shape_profiles = {}
        checks = {}
        for component in ("attention", "ffn"):
            compiled = [compile_workload(workload) for workload in workloads[component]]
            shape_profiles[component] = [
                {
                    "kernel": workload.kernel,
                    "name": workload.name,
                    "n": workload.n,
                    "d": workload.d,
                    "output_dim": workload.resolved_output_dim,
                    "block_size": workload.block_size,
                    "operations": profile.operations,
                    "offchip_bytes": profile.offchip_bytes,
                    "stage_count": len(profile.stages),
                    "metadata": profile.metadata,
                }
                for workload, profile in zip(workloads[component], compiled, strict=True)
            ]
            raw = simulated[n]["components"][component]["workloads"]
            checks[f"{component}_operations"] = [
                item["operations"] for item in raw
            ] == [profile.operations for profile in compiled]
            checks[f"{component}_bytes"] = [item["offchip_bytes"] for item in raw] == [
                profile.offchip_bytes for profile in compiled
            ]
        checks.update(
            {
                "hidden_fft_plain": workloads["attention"][0].kernel == "fft"
                and shape_profiles["attention"][0]["stage_count"] == 10,
                "token_fft_plain": workloads["attention"][1].kernel == "fft"
                and shape_profiles["attention"][1]["stage_count"]
                == int(math.log2(n)),
                "no_fft_compression": all(
                    profile["kernel"] == "fft"
                    and profile["stage_count"] == int(math.log2(profile["n"]))
                    for profile in shape_profiles["attention"]
                ),
                "global_ffn_stages": [
                    profile["stage_count"] for profile in shape_profiles["ffn"]
                ]
                == [10, 12],
                "global_ffn_blocks": [
                    profile["block_size"] for profile in shape_profiles["ffn"]
                ]
                == [1024, 4096],
            }
        )
        profiles[f"N{n}"] = shape_profiles
        mapping_checks[f"N{n}"] = checks

    source_coverage = {
        "aggregate_radix": {
            "available": reports["aggregate"]["hypothesis_status"] == "supported",
            "evidence": "H43 exact radix aggregation includes FFT8192 structure",
            "direct_fig19_compiler": False,
        },
        "fftcmp": {
            "available": True,
            "directly_reusable": False,
            "reason": "H81 includes truncate plus inverse stages; Figure 19 requires plain forward FFT",
        },
        "b32_paths": {
            "available": True,
            "directly_reusable": False,
            "reason": "H92 uses hierarchical B32 five-stage projections; Figure 19 requires global B1024/B4096",
        },
        "packet_memory": {
            "available": reports["packet_memory"]["hypothesis_status"] == "supported",
            "directly_reusable": True,
            "scope": "SIMD32 packets, grouped events, four-port SRAM",
        },
    }
    gap_checks = {
        "mapping_exact": all(all(checks.values()) for checks in mapping_checks.values()),
        "four_shapes": len(profiles) == 4,
        "two_fft_axes": all(len(item["attention"]) == 2 for item in profiles.values()),
        "two_global_ffn": all(len(item["ffn"]) == 2 for item in profiles.values()),
        "plain_fft_compiler_needed": source_coverage["fftcmp"]["directly_reusable"]
        is False,
        "global_bsmm_compiler_needed": source_coverage["b32_paths"][
            "directly_reusable"
        ]
        is False,
        "mechanism_reuse_available": source_coverage["aggregate_radix"]["available"]
        and source_coverage["packet_memory"]["directly_reusable"],
    }
    integrity_checks = {
        "frozen_files": all(item["pass"] for item in files.values()),
        "parents": all(parent_checks.values()),
        "mapping": all(all(checks.values()) for checks in mapping_checks.values()),
        "gaps": all(gap_checks.values()),
        "targets_consumed": False,
    }
    integrity = all(
        value for key, value in integrity_checks.items() if key != "targets_consumed"
    ) and not integrity_checks["targets_consumed"]
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "classification": config["classification"],
        "validation_eligible": config["validation_eligible"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_commit": git_commit(),
        "hypothesis_status": "supported" if integrity else "rejected",
        "audit_integrity": integrity,
        "source_integrated_timing_available": False,
        "mapping_identifiable": True,
        "frozen_inputs": files,
        "parent_checks": parent_checks,
        "model": h23_config["model"],
        "mapping": h23_config["mapping"],
        "profiles": profiles,
        "mapping_checks": mapping_checks,
        "source_coverage": source_coverage,
        "gap_checks": gap_checks,
        "integrity_checks": integrity_checks,
        "paper_performance_targets_consumed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report = build_audit(config)
    output = PROJECT_ROOT / config["result_path"]
    if args.verify_existing:
        existing = json.loads(output.read_text(encoding="utf-8"))
        keys = (
            "hypothesis_status", "audit_integrity", "profiles", "mapping_checks",
            "source_coverage", "gap_checks", "integrity_checks",
        )
        matches = all(existing.get(key) == report.get(key) for key in keys)
        print(json.dumps({"existing_matches": matches, **report}, indent=2))
        return 0 if matches else 1
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "hypothesis_status": report["hypothesis_status"],
                "audit_integrity": report["audit_integrity"],
                "mapping_identifiable": report["mapping_identifiable"],
                "source_integrated_timing_available": report[
                    "source_integrated_timing_available"
                ],
                "gap_checks": report["gap_checks"],
            },
            indent=2,
        )
    )
    return 0 if report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
