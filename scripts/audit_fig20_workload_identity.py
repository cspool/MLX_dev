#!/usr/bin/env python3
"""Audit H57 proxy work against matched Llama2-7B Figure 20 shapes."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from mlxsim.experiments import _llama_kernel_workloads
from mlxsim.workloads import compile_workload

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/fig20_workload_identity_v1.yaml"


def qualify(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    path = path.resolve()
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


def logical_profiles(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for n in config["shape"]["sequence_lengths"]:
        workloads = _llama_kernel_workloads(
            int(n), sparse=True, batch=int(config["shape"]["batch"])
        )
        for kernel in config["shape"]["kernels"]:
            profiles = [compile_workload(workload) for workload in workloads[kernel]]
            key = f"{kernel}-N{n}"
            result[key] = {
                "operations": sum(profile.operations for profile in profiles),
                "fma_equivalents": sum(profile.operations for profile in profiles) / 2.0,
                "offchip_bytes": sum(profile.offchip_bytes for profile in profiles),
                "output_elements": sum(profile.output_elements for profile in profiles),
                "component_count": len(profiles),
                "stage_count": sum(len(profile.stages) for profile in profiles),
                "components": [
                    {
                        "operations": profile.operations,
                        "offchip_bytes": profile.offchip_bytes,
                        "output_elements": profile.output_elements,
                        "stages": len(profile.stages),
                        "metadata": profile.metadata,
                    }
                    for profile in profiles
                ],
            }
    return result


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    files = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    manifest_path = PROJECT_ROOT / config["h57_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mlx_proxy = {item["name"]: item for item in manifest["mlx_outputs"]}
    gpu_proxy = {item["name"]: item for item in manifest["gpu_jobs"]}
    logical = logical_profiles(config)
    comparisons: dict[str, Any] = {}
    for n, case in ((256, "short"), (8192, "long")):
        for kernel in config["shape"]["kernels"]:
            proxy_family = "fft" if kernel == "attention" else "bsmm"
            proxy_key = f"{proxy_family}--{case}"
            logical_key = f"{kernel}-N{n}"
            profile = logical[logical_key]
            mlx_fma = int(mlx_proxy[proxy_key]["mlx_fma_equivalents"])
            gpu_fma = int(gpu_proxy[proxy_key]["gpu_fma_equivalents"])
            comparisons[logical_key] = {
                "proxy_key": proxy_key,
                "logical": profile,
                "mlx_proxy_fma_equivalents": mlx_fma,
                "gpu_proxy_fma_equivalents": gpu_fma,
                "mlx_represented_fraction": mlx_fma / profile["fma_equivalents"],
                "gpu_represented_fraction": gpu_fma / profile["fma_equivalents"],
            }
    identity_checks = {
        "qkv_ffn_shapes_differ": all(
            logical[f"qkv-N{n}"]["fma_equivalents"]
            != logical[f"ffn1-N{n}"]["fma_equivalents"]
            for n in (256, 8192)
        ),
        "ffn_output_shapes_differ": all(
            logical[f"ffn1-N{n}"]["output_elements"]
            != logical[f"ffn2-N{n}"]["output_elements"]
            for n in (256, 8192)
        ),
        "shared_bsmm_proxy": all(
            comparisons[f"{kernel}-N{n}"]["proxy_key"] == f"bsmm--{case}"
            for n, case in ((256, "short"), (8192, "long"))
            for kernel in ("qkv", "ffn1", "ffn2")
        ),
        "proxy_fraction_below_one_percent": all(
            item["mlx_represented_fraction"] < 0.01
            and item["gpu_represented_fraction"] < 0.01
            for item in comparisons.values()
        ),
        "sequence_scaling_mismatch": (
            mlx_proxy["bsmm--long"]["mlx_fma_equivalents"]
            / mlx_proxy["bsmm--short"]["mlx_fma_equivalents"]
            != logical["qkv-N8192"]["fma_equivalents"]
            / logical["qkv-N256"]["fma_equivalents"]
        ),
        "attention_has_two_components": all(
            logical[f"attention-N{n}"]["component_count"] == 2
            for n in (256, 8192)
        ),
        "h57_attention_single_proxy": all(
            comparisons[f"attention-N{n}"]["proxy_key"] == f"fft--{case}"
            for n, case in ((256, "short"), (8192, "long"))
        ),
    }
    source_files = {
        "auditor": {
            "path": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "workloads": {
            "path": "src/mlxsim/workloads.py",
            "sha256": hashlib.sha256(
                (PROJECT_ROOT / "src/mlxsim/workloads.py").read_bytes()
            ).hexdigest(),
        },
    }
    integrity_checks = {
        "frozen_files": all(item["pass"] for item in files.values()),
        "manifest": len(mlx_proxy) == len(gpu_proxy) == 4
        and manifest["paper_target_values_consumed"] is False,
        "logical_matrix": len(logical) == 8,
        "comparison_matrix": len(comparisons) == 8,
        "identity_mismatch_proven": all(identity_checks.values()),
        "targets_consumed": False,
    }
    integrity = all(value for key, value in integrity_checks.items() if key != "targets_consumed")
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
        "frozen_inputs": files,
        "manifest": {
            "path": str(manifest_path.relative_to(PROJECT_ROOT)),
            "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        },
        "logical_profiles": logical,
        "comparisons": comparisons,
        "identity_checks": identity_checks,
        "source_files": source_files,
        "integrity_checks": integrity_checks,
        "conclusion": "H57 execution proxies do not establish matched Figure 20 workload identity",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report = build_audit(config)
    output = PROJECT_ROOT / config["result_path"]
    if args.verify_existing:
        existing = json.loads(output.read_text(encoding="utf-8"))
        matches = all(
            existing.get(key) == report.get(key)
            for key in ("hypothesis_status", "audit_integrity", "identity_checks")
        )
        print(json.dumps({"existing_matches": matches, **report}, indent=2, sort_keys=True))
        return 0 if matches else 1
    if args.preflight_only:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["audit_integrity"] and not output.exists() else 1
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
