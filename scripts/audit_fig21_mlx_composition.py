#!/usr/bin/env python3
"""Compose H92-H94 into target-free 24+8-layer Figure 21 MLX timing."""

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

from mlxsim.experiments import _llama_memory_gb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/fig21_mlx_composition_v1.yaml"


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
    parents = {
        name: json.loads((PROJECT_ROOT / spec["path"]).read_text(encoding="utf-8"))
        for name, spec in config["frozen_inputs"].items()
    }
    parent_checks = {
        name: report["hypothesis_status"] == config["frozen_inputs"][name]["required_status"]
        and report["audit_integrity"]
        is config["frozen_inputs"][name]["required_integrity"]
        for name, report in parents.items()
    }
    timed = parents["timed_paths"]["full_estimates"]
    structured_attention = parents["structured_attention"]["full_estimates"]
    dense_attention = parents["dense_attention"]["full_estimates"]
    structured_layers = int(config["structured_layers"])
    dense_layers = int(config["dense_layers"])
    clock = int(config["clock_hz"])
    rows = []
    row_checks = {}
    for n_value in config["sequence_lengths"]:
        n = int(n_value)
        shape = f"N{n}"
        structured_projection = sum(
            float(timed[f"{shape}-structured_{component}"])
            for component in ("qkv", "output", "ffn1", "ffn2")
        )
        dense_projection = sum(
            float(timed[f"{shape}-dense_{component}"])
            for component in ("qkv", "output", "ffn1", "ffn2")
        )
        elementwise = float(timed[f"{shape}-elementwise"])
        structured_attn = float(structured_attention[shape])
        dense_attn = float(dense_attention[shape])
        structured_layer = structured_projection + structured_attn + elementwise
        dense_layer = dense_projection + dense_attn + elementwise
        total = structured_layers * structured_layer + dense_layers * dense_layer
        gemm = dense_layers * dense_projection
        memory = _llama_memory_gb(n, batch=8)
        row = {
            "sequence_length": n,
            "structured_layer_cycles": structured_layer,
            "dense_layer_cycles": dense_layer,
            "component_cycles": {
                "structured_projection": structured_layers * structured_projection,
                "structured_attention": structured_layers * structured_attn,
                "dense_projection_gemm": gemm,
                "dense_attention": dense_layers * dense_attn,
                "elementwise": (structured_layers + dense_layers) * elementwise,
            },
            "mlx_total_cycles": total,
            "mlx_latency_seconds": total / clock,
            "gemm_cycles": gemm,
            "gemm_time_share": gemm / total,
            "memory": memory,
            "xavier_total_cycles": None,
            "speedup_over_xavier": None,
        }
        component_sum = sum(row["component_cycles"].values())
        checks = {
            "positive": all(
                value > 0
                for value in (
                    structured_projection,
                    dense_projection,
                    structured_attn,
                    dense_attn,
                    elementwise,
                    total,
                )
            ),
            "layer_arithmetic": total
            == structured_layers * structured_layer + dense_layers * dense_layer,
            "component_sum": math.isclose(component_sum, total, rel_tol=0, abs_tol=1e-6),
            "gemm_share": 0.0 < row["gemm_time_share"] < 1.0,
            "memory": memory["dense"] > memory["sparse"] > 0,
            "xavier_unavailable": row["xavier_total_cycles"] is None
            and row["speedup_over_xavier"] is None,
        }
        rows.append(row)
        row_checks[shape] = checks
    summary = {
        "shape_count": len(rows),
        "all_rows_pass": all(all(checks.values()) for checks in row_checks.values()),
        "mlx_composition_available": True,
        "xavier_dense_tensor_available": False,
        "figure21_speedup_available": False,
        "gemm_time_shares": [row["gemm_time_share"] for row in rows],
    }
    integrity_checks = {
        "frozen_files": all(item["pass"] for item in files.values()),
        "parents": all(parent_checks.values()),
        "five_shapes": len(rows) == 5
        and [row["sequence_length"] for row in rows]
        == [int(value) for value in config["sequence_lengths"]],
        "rows": summary["all_rows_pass"],
        "layer_count": structured_layers + dense_layers == 32,
        "no_speedup": summary["figure21_speedup_available"] is False,
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
        "frozen_inputs": files,
        "parent_checks": parent_checks,
        "rows": rows,
        "row_checks": row_checks,
        "summary": summary,
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
        keys = ("hypothesis_status", "audit_integrity", "rows", "row_checks", "summary", "integrity_checks")
        matches = all(existing.get(key) == report.get(key) for key in keys)
        print(json.dumps({"existing_matches": matches, **report}, indent=2))
        return 0 if matches else 1
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "rows": report["rows"]}, indent=2))
    return 0 if report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
