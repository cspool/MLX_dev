#!/usr/bin/env python3
"""Audit H45's target-independent SIMD/mesh scaling mechanism."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = PROJECT_ROOT / "artifacts/environment/h45"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/dsagen_mlx_scaling_mechanism_v1.yaml"
NAMES = ("baseline", "simd32_4x4", "simd8_8x8", "simd32_8x8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--verify-existing", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def qualify(path: Path, expected: dict[str, Any] | None = None) -> dict[str, Any]:
    is_file = path.is_file()
    size = path.stat().st_size if is_file else None
    digest = sha256_file(path) if is_file else None
    checks = {"is_file": is_file}
    if expected:
        checks.update(
            {
                "bytes": size == expected["bytes"],
                "sha256": digest == expected["sha256"],
            }
        )
    return {
        "path": str(path.relative_to(PROJECT_ROOT)) if is_file else str(path),
        "bytes": size,
        "sha256": digest,
        "checks": checks,
        "pass": all(checks.values()),
    }


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={PROJECT_ROOT}", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    fig22 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["fig22_result"]["path"]).read_text(
            encoding="utf-8"
        )
    ) if frozen["fig22_result"]["pass"] else {}
    compiler_path = EVIDENCE_ROOT / "scaling-compile-manifest.json"
    replay_path = EVIDENCE_ROOT / "replay/scaling-compile-manifest.json"
    compiler_file = qualify(compiler_path)
    compiler = json.loads(compiler_path.read_text(encoding="utf-8")) if compiler_file["pass"] else {}
    bindings: dict[str, bool] = {}
    replay_checks: dict[str, bool] = {}
    for name, specification in (compiler.get("outputs") or {}).items():
        path = EVIDENCE_ROOT / specification["path"]
        artifact = qualify(path)
        bindings[name] = (
            artifact["bytes"] == specification["bytes"]
            and artifact["sha256"] == specification["sha256"]
        )
        replay = EVIDENCE_ROOT / "replay" / path.name
        replay_checks[name] = replay.is_file() and sha256_file(replay) == artifact["sha256"]
    logical = compiler.get("logical_work") or {}
    reference = logical.get("baseline") or {}
    conservation_checks = {
        name: all(
            values.get(key) == reference.get(key)
            for key in (
                "logical_pair_iterations",
                "vector_instruction_lane_work",
                "memory_request_lane_work",
                "transfer_lane_work",
            )
        )
        for name, values in logical.items()
    }
    compiler_checks = {
        "manifest": compiler_file["pass"],
        "manifest_replay": replay_path.is_file()
        and sha256_file(replay_path) == compiler_file["sha256"],
        "four_outputs": set(compiler.get("outputs") or {}) == set(NAMES),
        "bindings": all(bindings.values()),
        "replays": all(replay_checks.values()),
        "compiler_checks": all(
            value
            for key, value in (compiler.get("checks") or {}).items()
            if key != "paper_targets_consumed"
        )
        and (compiler.get("checks") or {}).get("paper_targets_consumed") is False,
        "work_conservation": all(conservation_checks.values()),
        "simd_fourfold": compiler["outputs"]["baseline"]["metadata"][
            "instruction_count"
        ]
        == 4 * compiler["outputs"]["simd32_4x4"]["metadata"]["instruction_count"],
        "mesh_slots_fourfold": logical["simd8_8x8"]["active_slots_per_stage"]
        == 4 * logical["baseline"]["active_slots_per_stage"],
        "footprint": all(
            item["active_instruction_footprint"] <= 18 for item in logical.values()
        ),
    }

    run_path = EVIDENCE_ROOT / "scaling-run-manifest.json"
    run_file = qualify(run_path)
    run = json.loads(run_path.read_text(encoding="utf-8")) if run_file["pass"] else {}
    run_checks = {
        "manifest": run_file["pass"],
        "four_runs": set(run.get("runs") or {}) == set(NAMES),
        "all_internal_checks": all(
            value
            for key, value in (run.get("checks") or {}).items()
            if key != "figure23_targets_consumed"
        )
        and (run.get("checks") or {}).get("figure23_targets_consumed") is False,
        "positive_speedups": all(value > 1.0 for value in (run.get("speedups") or {}).values()),
    }
    summary_files: dict[str, Any] = {}
    summary_bindings: dict[str, bool] = {}
    for name in NAMES:
        for build in ("debug", "opt", "sanitize"):
            key = f"{build}-{name}"
            path = EVIDENCE_ROOT / "raw" / f"{key}-summary.json"
            artifact = qualify(path)
            summary_files[key] = artifact
            expected = run.get("runs", {}).get(name, {}).get(build, {})
            summary_bindings[key] = (
                artifact["sha256"] == expected.get("summary_sha256")
                and json.loads(path.read_text(encoding="utf-8")) == expected.get("summary")
            ) if artifact["pass"] else False
    run_checks["summary_files"] = all(item["pass"] for item in summary_files.values())
    run_checks["summary_bindings"] = all(summary_bindings.values())

    source_text = "\n".join(
        (PROJECT_ROOT / path).read_text(encoding="utf-8")
        for path in (
            "scripts/compile_mlx_scaling.py",
            "scripts/run_mlx_scaling_mechanism.py",
        )
    )
    source_checks = {
        "compiler_trip_mapping": "compiler_outer_trip" in source_text,
        "work_factor": "simd_work_factor" in source_text,
        "no_figure23_target_values": not any(
            token in source_text for token in ("3.88", "3.62", "14.1", "3.9", "3.6", "14.0")
        ),
        "temporary_traces_hashed_then_removed": "trace.unlink()" in source_text,
    }
    integrity_checks = {
        "frozen_inputs": all(item["pass"] for item in frozen.values()),
        "fig22_rejection_preserved": fig22.get("hypothesis_status")
        == config["frozen_inputs"]["fig22_result"]["required_status"]
        and fig22.get("audit_integrity")
        is config["frozen_inputs"]["fig22_result"]["required_integrity"],
        "compiler": all(compiler_checks.values()),
        "runner": all(run_checks.values()),
        "source": all(source_checks.values()),
        "figure23_targets_consumed": False,
    }
    audit_integrity = all(
        value for key, value in integrity_checks.items() if key != "figure23_targets_consumed"
    ) and integrity_checks["figure23_targets_consumed"] is False
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "classification": config["classification"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_commit": git_commit(),
        "hypothesis_status": "supported" if audit_integrity else "rejected",
        "audit_integrity": audit_integrity,
        "frozen_inputs": frozen,
        "compiler": {
            "artifact": compiler_file,
            "bindings": bindings,
            "replays": replay_checks,
            "conservation": conservation_checks,
            "checks": compiler_checks,
            "pass": all(compiler_checks.values()),
        },
        "execution": {
            "artifact": run_file,
            "cycles": run.get("cycles"),
            "speedups": run.get("speedups"),
            "trace_hashes": {
                name: {
                    build: run["runs"][name][build]["trace_sha256"]
                    for build in ("debug", "opt", "sanitize")
                }
                for name in NAMES
            }
            if run.get("runs")
            else {},
            "summary_files": summary_files,
            "summary_bindings": summary_bindings,
            "checks": run_checks,
            "pass": all(run_checks.values()),
        },
        "source_checks": source_checks,
        "integrity_checks": integrity_checks,
        "wall_seconds": time.perf_counter() - started,
    }


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report = build_audit(config)
    output = PROJECT_ROOT / config["result_path"]
    if args.verify_existing:
        if not output.is_file():
            raise FileNotFoundError(output)
        existing = json.loads(output.read_text(encoding="utf-8"))
        keys = ("hypothesis_status", "audit_integrity", "integrity_checks")
        matches = all(existing.get(key) == report.get(key) for key in keys)
        print(json.dumps({"existing_matches": matches, **report}, indent=2, sort_keys=True))
        return 0 if matches else 1
    if args.preflight_only:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["audit_integrity"] and not output.exists() else 1
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
