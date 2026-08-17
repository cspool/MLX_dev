#!/usr/bin/env python3
"""Audit H46's no-fit structured-proxy transfer to Figure 23."""

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
EVIDENCE_ROOT = PROJECT_ROOT / "artifacts/environment/h46"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/dsagen_mlx_fig23_v1.yaml"


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


def rel_error(actual: float, target: float) -> float:
    return abs(actual - target) / abs(target)


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    scaling = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["scaling_result"]["path"]).read_text(
            encoding="utf-8"
        )
    ) if frozen["scaling_result"]["pass"] else {}
    compile_path = EVIDENCE_ROOT / "fig23-compile-manifest.json"
    compile_file = qualify(compile_path)
    compiler = json.loads(compile_path.read_text(encoding="utf-8")) if compile_file["pass"] else {}
    replay_path = EVIDENCE_ROOT / "fig23-compiler-replay-check.json"
    replay_file = qualify(replay_path)
    replay = json.loads(replay_path.read_text(encoding="utf-8")) if replay_file["pass"] else {}
    bindings: dict[str, bool] = {}
    replay_checks: dict[str, bool] = {}
    structural: dict[str, bool] = {}
    for key, specification in (compiler.get("outputs") or {}).items():
        path = EVIDENCE_ROOT / specification["path"]
        artifact = qualify(path)
        bindings[key] = (
            artifact["bytes"] == specification["bytes"]
            and artifact["sha256"] == specification["sha256"]
        )
        replay_checks[key] = replay.get("checks", {}).get(path.name) is True
        document = json.loads(path.read_text(encoding="utf-8"))
        structural[key] = (
            document.get("record_events") is False
            and document.get("memory_backend") == "fixed"
            and document.get("metadata", {}).get("sequence_length")
            == int(key.split("-")[0])
            and document.get("metadata", {}).get("hardware_name")
            == key.removeprefix(key.split("-")[0] + "-")
        )
    compiler_checks = {
        "manifest": compile_file["pass"],
        "twenty_outputs": len(compiler.get("outputs") or {}) == 20,
        "bindings": all(bindings.values()),
        "replay_certificate": replay_file["pass"]
        and replay.get("all_identical") is True
        and replay.get("file_count") == 21,
        "replay_files": all(replay_checks.values()),
        "structural": all(structural.values()),
        "work_conservation": all(
            all(per_sequence.values())
            for per_sequence in (compiler.get("conservation") or {}).values()
        ),
        "compiler_checks": all(
            value
            for key, value in (compiler.get("checks") or {}).items()
            if key != "targets_consumed_by_compiler"
        )
        and (compiler.get("checks") or {}).get("targets_consumed_by_compiler") is False,
    }

    run_path = EVIDENCE_ROOT / "fig23-run-manifest.json"
    run_file = qualify(run_path)
    run = json.loads(run_path.read_text(encoding="utf-8")) if run_file["pass"] else {}
    summary_files: dict[str, Any] = {}
    execution_checks: dict[str, bool] = {}
    for key, details in (run.get("runs") or {}).items():
        first_path = EVIDENCE_ROOT / "runs" / f"{key}-first.json"
        second_path = EVIDENCE_ROOT / "runs" / f"{key}-second.json"
        first = qualify(first_path)
        second = qualify(second_path)
        summary_files[f"{key}-first"] = first
        summary_files[f"{key}-second"] = second
        summary = json.loads(first_path.read_text(encoding="utf-8")) if first["pass"] else {}
        metadata = (compiler.get("outputs", {}).get(key) or {}).get("metadata") or {}
        execution_checks[key] = (
            first["pass"]
            and second["pass"]
            and first["sha256"] == second["sha256"] == details.get("summary_sha256")
            and summary == details.get("summary")
            and summary.get("done") is True
            and summary.get("record_events") is False
            and summary.get("instructions_completed") == metadata.get("instruction_count")
        )
    runner_checks = {
        "manifest": run_file["pass"],
        "twenty_runs": len(run.get("runs") or {}) == 20,
        "all_summaries": all(item["pass"] for item in summary_files.values()),
        "execution": all(execution_checks.values()),
        "runner_checks": all(
            value
            for key, value in (run.get("checks") or {}).items()
            if key != "targets_consumed_by_runner"
        )
        and (run.get("checks") or {}).get("targets_consumed_by_runner") is False,
    }

    points: dict[str, list[dict[str, Any]]] = {}
    errors: list[float] = []
    passing = 0
    for name, targets in config["targets"].items():
        if name == "uncertainty_abs":
            continue
        points[name] = []
        actuals = run.get("speedups", {}).get(name) or []
        for sequence, actual, target in zip(
            config["mapping"]["sequence_lengths"], actuals, targets, strict=True
        ):
            error = rel_error(float(actual), float(target))
            passed = error <= 0.10
            passing += passed
            errors.append(error)
            points[name].append(
                {
                    "sequence_length": sequence,
                    "actual": actual,
                    "target": target,
                    "relative_error": error,
                    "pass_10pct": passed,
                }
            )
    source_text = "\n".join(
        (PROJECT_ROOT / path).read_text(encoding="utf-8")
        for path in ("scripts/compile_dsagen_fig23.py", "scripts/run_dsagen_fig23.py")
    )
    forbidden_tokens = (
        "mesh_congestion_penalty",
        "mesh_fill_penalty",
        "speedup_scale",
        "speedup_intercept",
        "paper_v1",
    )
    integrity_checks = {
        "frozen_inputs": all(item["pass"] for item in frozen.values()),
        "h45_supported": scaling.get("hypothesis_status")
        == config["frozen_inputs"]["scaling_result"]["required_status"]
        and scaling.get("audit_integrity")
        is config["frozen_inputs"]["scaling_result"]["required_integrity"],
        "compiler": all(compiler_checks.values()),
        "runner": all(runner_checks.values()),
        "forbidden_adjustments_absent": not any(
            token in source_text for token in forbidden_tokens
        ),
        "post_result_change": False,
    }
    audit_integrity = all(
        value for key, value in integrity_checks.items() if key != "post_result_change"
    ) and integrity_checks["post_result_change"] is False
    all_points = passing == 15
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "classification": config["classification"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_commit": git_commit(),
        "hypothesis_status": "supported" if all_points else "rejected",
        "audit_integrity": audit_integrity,
        "frozen_inputs": frozen,
        "compiler": {
            "artifact": compile_file,
            "replay": replay_file,
            "bindings": bindings,
            "replay_checks": replay_checks,
            "structural": structural,
            "checks": compiler_checks,
            "pass": all(compiler_checks.values()),
        },
        "execution": {
            "artifact": run_file,
            "summary_files": summary_files,
            "execution_checks": execution_checks,
            "cycles": run.get("cycles"),
            "speedups": run.get("speedups"),
            "checks": runner_checks,
            "pass": all(runner_checks.values()),
        },
        "points": points,
        "summary": {
            "passing_points": passing,
            "total_points": 15,
            "mape": sum(errors) / len(errors),
            "max_relative_error": max(errors),
            "validation_eligible": False,
            "proxy_mapping": True,
        },
        "integrity_checks": integrity_checks,
        "numerical_gate": {"all_15_within_10pct": all_points},
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
        keys = ("hypothesis_status", "audit_integrity", "numerical_gate")
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
