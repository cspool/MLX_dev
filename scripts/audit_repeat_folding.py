#!/usr/bin/env python3
"""Audit H76 affine repeat folding across three memory mechanisms."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from mlxsim.repeat_folding import fit_affine, relative_error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/repeat_folding_v1.yaml"
RUN_ROOTS = {
    "fixed": PROJECT_ROOT / "artifacts/environment/h64/runs",
    "single_buffer": PROJECT_ROOT / "artifacts/environment/h67/runs",
    "column_port": PROJECT_ROOT / "artifacts/environment/h69/runs",
}


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


def load_summary(backend: str, sequence: int, hardware: str) -> dict[str, Any]:
    path = RUN_ROOTS[backend] / f"{sequence}-{hardware}-first.json"
    return json.loads(path.read_text(encoding="utf-8"))


def exact_work_conservation(summaries: list[dict[str, Any]], sequences: list[int]) -> bool:
    scalar_fields = (
        "instructions_issued",
        "instructions_completed",
        "boundary_events_emitted",
        "route_hops",
        "skip_hops",
        "unit_hops",
    )
    base_sequence = sequences[0]
    base = summaries[0]
    for sequence, summary in zip(sequences, summaries, strict=True):
        for field in scalar_fields:
            if summary[field] * base_sequence != base[field] * sequence:
                return False
        for name, value in summary["issued_by_pipeline"].items():
            if value * base_sequence != base["issued_by_pipeline"][name] * sequence:
                return False
    return True


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    files = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    parents = {
        name: json.loads((PROJECT_ROOT / spec["path"]).read_text(encoding="utf-8"))
        for name, spec in config["frozen_inputs"].items()
    }
    fit_sequences = [int(value) for value in config["fit_sequences"]]
    holdouts = [int(value) for value in config["holdout_sequences"]]
    all_sequences = [*fit_sequences, *holdouts]
    reports: dict[str, Any] = {}
    errors: list[float] = []
    conservation_checks: dict[str, bool] = {}
    for backend in config["frozen_inputs"]:
        reports[backend] = {}
        for hardware in config["hardware"]:
            model = fit_affine(
                fit_sequences[0],
                parents[backend]["cycles"][str(fit_sequences[0])][hardware],
                fit_sequences[1],
                parents[backend]["cycles"][str(fit_sequences[1])][hardware],
            )
            points = []
            for sequence in holdouts:
                actual = float(parents[backend]["cycles"][str(sequence)][hardware])
                predicted = model.predict(sequence)
                error = relative_error(predicted, actual)
                errors.append(error)
                points.append(
                    {
                        "sequence": sequence,
                        "actual_cycles": actual,
                        "predicted_cycles": predicted,
                        "relative_error": error,
                        "pass": error <= config["cycle_relative_error_limit"],
                    }
                )
            summaries = [
                load_summary(backend, sequence, hardware) for sequence in all_sequences
            ]
            conservation = exact_work_conservation(summaries, all_sequences)
            conservation_checks[f"{backend}-{hardware}"] = conservation
            reports[backend][hardware] = {
                "intercept": model.intercept,
                "slope_cycles_per_sequence_unit": model.slope,
                "holdouts": points,
                "work_conservation": conservation,
            }
    parent_checks = {
        name: parents[name]["hypothesis_status"] == spec["required_status"]
        and parents[name]["audit_integrity"] is spec["required_integrity"]
        for name, spec in config["frozen_inputs"].items()
    }
    source_files = {
        name: {
            "path": path,
            "sha256": hashlib.sha256((PROJECT_ROOT / path).read_bytes()).hexdigest(),
        }
        for name, path in config["source_layout"].items()
    }
    summary = {
        "passing_predictions": sum(
            error <= config["cycle_relative_error_limit"] for error in errors
        ),
        "total_predictions": len(errors),
        "max_relative_error": max(errors),
        "mape": sum(errors) / len(errors),
        "all_predictions_pass": all(
            error <= config["cycle_relative_error_limit"] for error in errors
        ),
        "all_work_conserved": all(conservation_checks.values()),
    }
    integrity_checks = {
        "frozen_files": all(item["pass"] for item in files.values()),
        "parents": all(parent_checks.values()),
        "prediction_count": len(errors) == 36,
        "predictions": summary["all_predictions_pass"],
        "work_conservation": summary["all_work_conserved"],
        "source_files": all((PROJECT_ROOT / item["path"]).is_file() for item in source_files.values()),
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
        "models": reports,
        "work_conservation_checks": conservation_checks,
        "summary": summary,
        "source_files": source_files,
        "integrity_checks": integrity_checks,
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
            for key in ("hypothesis_status", "audit_integrity", "summary")
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
