#!/usr/bin/env python3
"""Audit H92 timed Figure 21 non-Attention paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from mlxsim.dsagen_overlay import canonical_json
from mlxsim.fig21_timed_paths import compile_timed_path
from mlxsim.repeat_folding import fit_affine, relative_error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig21_timed_paths_v1.yaml"


def qualify(path: Path, expected: dict[str, Any] | None = None) -> dict[str, Any]:
    exists = path.is_file()
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if exists else None
    checks = {"is_file": exists}
    if expected and "sha256" in expected:
        checks["sha256"] = digest == expected["sha256"]
    try:
        display = str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        display = str(path)
    return {
        "path": display,
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
        if name in {"layer_contract", "spad", "column_ports", "full_block"}
    }
    parent_checks = {
        name: report["hypothesis_status"] == config["frozen_inputs"][name]["required_status"]
        and report["audit_integrity"]
        is config["frozen_inputs"][name]["required_integrity"]
        for name, report in parents.items()
    }
    output_root = PROJECT_ROOT / config["output_root"]
    compile_path = output_root / "fig21-timed-paths-compile-manifest.json"
    run_path = output_root / "fig21-timed-paths-run-manifest.json"
    compile_file = qualify(compile_path)
    run_file = qualify(run_path)
    compiler = json.loads(compile_path.read_text(encoding="utf-8"))
    run = json.loads(run_path.read_text(encoding="utf-8"))

    compile_checks = {}
    execution_checks = {}
    measurements = {}
    for run_key, item in compiler["outputs"].items():
        path_key, scale_text = run_key.rsplit("-q", maxsplit=1)
        scale = int(scale_text)
        document, metadata = compile_timed_path(
            name=run_key,
            normalized=compiler["path_contracts"][path_key],
            scale=scale,
            active_window=int(config["hardware"]["active_window"]),
        )
        path = PROJECT_ROOT / item["artifact"]["path"]
        artifact = qualify(path, item["artifact"])
        compile_checks[run_key] = (
            artifact["pass"]
            and path.read_text(encoding="utf-8") == canonical_json(document)
            and item["metadata"] == metadata
        )
        first = run["records"][run_key]["first"]
        second = run["records"][run_key]["second"]
        first_summary = qualify(
            PROJECT_ROOT / first["summary_path"],
            {"sha256": first["summary_sha256"]},
        )
        second_summary = qualify(
            PROJECT_ROOT / second["summary_path"],
            {"sha256": second["summary_sha256"]},
        )
        first_adapter = qualify(
            PROJECT_ROOT / first["adapter_path"],
            {"sha256": first["adapter_sha256"]},
        )
        second_adapter = qualify(
            PROJECT_ROOT / second["adapter_path"],
            {"sha256": second["adapter_sha256"]},
        )
        summary = first["summary"]
        adapter = first["adapter"]
        expected_instructions = sum(metadata["pipeline_counts"].values())
        checks = {
            "files": first_summary["pass"]
            and second_summary["pass"]
            and first_adapter["pass"]
            and second_adapter["pass"],
            "replay": first["summary_sha256"] == second["summary_sha256"]
            and first["adapter_sha256"] == second["adapter_sha256"],
            "done": summary["done"] is True,
            "instructions": summary["instructions_issued"]
            == summary["instructions_completed"]
            == expected_instructions,
            "pipelines": summary["issued_by_pipeline"] == metadata["pipeline_counts"],
            "events": summary["boundary_events_emitted"]
            == metadata["dynamic_event_count"],
            "memory": summary["external_memory_requests"]
            == summary["external_memory_completions"]
            == metadata["memory_requests"]
            == adapter["requests"]
            == adapter["responses"],
            "ports": adapter["ports"] == int(config["hardware"]["spad_ports"]),
            "axis": adapter["axis"] == config["hardware"]["spad_axis"],
        }
        execution_checks[run_key] = all(checks.values())
        measurements[run_key] = {
            "cycles": int(summary["cycles"]),
            "metadata": metadata,
            "checks": checks,
        }

    fit_scales = [int(value) for value in config["fit_scales"]]
    holdout_scales = [int(value) for value in config["holdout_scales"]]
    limit = float(config["cycle_relative_error_limit"])
    models = {}
    errors = []
    full_estimates = {}
    for path_key, normalized in compiler["path_contracts"].items():
        model = fit_affine(
            fit_scales[0],
            measurements[f"{path_key}-q{fit_scales[0]}"]["cycles"],
            fit_scales[1],
            measurements[f"{path_key}-q{fit_scales[1]}"]["cycles"],
        )
        holdouts = []
        for scale in holdout_scales:
            actual = measurements[f"{path_key}-q{scale}"]["cycles"]
            predicted = model.predict(scale)
            error = relative_error(predicted, actual)
            errors.append(error)
            holdouts.append(
                {
                    "scale": scale,
                    "actual_cycles": actual,
                    "predicted_cycles": predicted,
                    "relative_error": error,
                    "pass_5pct": error <= limit,
                }
            )
        full_scale = int(normalized["full_scale"])
        models[path_key] = {
            "intercept": model.intercept,
            "slope_cycles_per_scale": model.slope,
            "holdouts": holdouts,
            "full_scale": full_scale,
            "full_predicted_cycles": model.predict(full_scale),
        }
        full_estimates[path_key] = model.predict(full_scale)
    numerical = {
        "passing_holdouts": sum(error <= limit for error in errors),
        "total_holdouts": len(errors),
        "mape": sum(errors) / len(errors),
        "max_error": max(errors),
        "all_holdouts_pass": all(error <= limit for error in errors),
    }
    full_work_checks = {
        key: all(checks.values()) for key, checks in compiler["checks"].items()
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    implementation_text = "\n".join(
        (PROJECT_ROOT / path).read_text(encoding="utf-8")
        for name, path in config["source_layout"].items()
        if name != "auditor"
    ).lower()
    summary = {
        "path_count": len(models),
        "run_count": len(measurements),
        "full_estimate_count": len(full_estimates),
        **numerical,
    }
    integrity_checks = {
        "frozen_files": all(item["pass"] for item in files.values()),
        "parents": all(parent_checks.values()),
        "compile_manifest": compile_file["pass"]
        and compiler["paper_performance_targets_consumed"] is False
        and len(compiler["outputs"]) == 180,
        "run_manifest": run_file["pass"]
        and run["paper_performance_targets_consumed"] is False
        and len(run["records"]) == 180,
        "compiler_replay": all(compile_checks.values()),
        "executions": all(execution_checks.values()),
        "run_replays": all(run["checks"].values()),
        "full_work": all(full_work_checks.values()),
        "source_files": all(item["pass"] for item in source_files.values()),
        "targets_absent": "paper_targets" not in implementation_text,
        "counts": summary["path_count"] == 45
        and summary["run_count"] == 180
        and summary["total_holdouts"] == 90,
    }
    integrity = all(integrity_checks.values())
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "classification": config["classification"],
        "validation_eligible": config["validation_eligible"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_commit": git_commit(),
        "hypothesis_status": (
            "supported" if integrity and numerical["all_holdouts_pass"] else "rejected"
        ),
        "audit_integrity": integrity,
        "frozen_inputs": files,
        "parent_checks": parent_checks,
        "compile_manifest": compile_file,
        "run_manifest": run_file,
        "compile_checks": compile_checks,
        "execution_checks": execution_checks,
        "full_work_checks": full_work_checks,
        "measurements": measurements,
        "models": models,
        "full_estimates": full_estimates,
        "summary": summary,
        "source_files": source_files,
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
            "hypothesis_status",
            "audit_integrity",
            "full_work_checks",
            "models",
            "full_estimates",
            "summary",
            "integrity_checks",
        )
        matches = all(existing.get(key) == report.get(key) for key in keys)
        print(json.dumps({"existing_matches": matches, **report}, indent=2))
        return 0 if matches else 1
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0 if report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
