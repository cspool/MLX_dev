#!/usr/bin/env python3
"""Audit H101 exact batch-32 Figure 24/25 MLX paths."""

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
from mlxsim.fig24_25_exact_paths import compile_fft_cmp_path, compile_swa_path
from mlxsim.repeat_folding import fit_affine, relative_error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig24_25_exact_paths_v1.yaml"


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


def recompile(
    *, run_key: str, contract: dict[str, Any], scale: int, active_window: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    family = contract["family"]
    case = contract["case"]
    operator = contract["operator"]
    if family == "qkv_bsmm":
        return compile_timed_path(
            name=run_key,
            normalized=contract["normalized"],
            scale=scale,
            active_window=active_window,
        )
    if family == "fft":
        return compile_fft_cmp_path(
            name=run_key,
            sequence_length=int(case["n"]),
            hidden_dimension=int(case["d"]),
            batch=int(case["batch"]),
            scale=scale,
        )
    if family == "swa":
        return compile_swa_path(
            name=run_key,
            sequence_length=int(case["n"]),
            hidden_dimension=int(case["d"]),
            batch=int(case["batch"]),
            window=int(operator["window"]),
            query_tile=int(operator["query_tile"]),
            scale=scale,
        )
    raise ValueError(f"unknown family: {family}")


def reconstruct_full_work(
    *, contract: dict[str, Any], metadata: dict[str, Any], vector_bytes: int,
    simd_width: int,
) -> dict[str, Any]:
    family = contract["family"]
    case = contract["case"]
    scale = int(metadata["scale"])
    if family == "qkv_bsmm":
        full_scale = int(contract["normalized"]["full_scale"])
        expected_load = int(contract["normalized"]["full_load_bytes"])
        expected_store = int(contract["normalized"]["full_store_bytes"])
    else:
        full_scale = int(contract["full_scale"])
        batch = int(case["batch"])
        n = int(case["n"])
        d = int(case["d"])
        if family == "fft":
            expected_load = batch * n * d * 3 * 2
            expected_store = batch * (n // 2) * d * 3 * 2
        elif family == "swa":
            expected_load = batch * n * d * 3 * 2
            expected_store = batch * n * d * 2
        else:
            raise ValueError(family)
    reconstructed_fu = {
        operation: int(count) * simd_width * full_scale // scale
        for operation, count in metadata["operation_counts"].items()
    }
    reconstructed_load = (
        int(metadata["pipeline_counts"]["load"])
        * vector_bytes
        * full_scale
        // scale
    )
    reconstructed_store = (
        int(metadata["pipeline_counts"]["store"])
        * vector_bytes
        * full_scale
        // scale
    )
    expected_fu = {name: int(value) for name, value in contract["actual"]["fu"].items()}
    reconstructed_stages = metadata.get("stage_count")
    if reconstructed_stages is None:
        reconstructed_stages = metadata.get("normalized", {}).get("stage_count")
    checks = {
        "fu": reconstructed_fu == expected_fu,
        "load_bytes": reconstructed_load == expected_load,
        "store_bytes": reconstructed_store == expected_store,
        "stage_count": reconstructed_stages is not None
        and int(reconstructed_stages) == int(contract["actual"]["stage_count"]),
    }
    if family == "swa":
        checks["query_tile"] = int(metadata["query_tile"]) == int(
            contract["actual"]["query_tile"]
        )
    return {
        "family": family,
        "full_scale": full_scale,
        "reconstructed_fu": reconstructed_fu,
        "expected_fu": expected_fu,
        "reconstructed_load_bytes": reconstructed_load,
        "expected_load_bytes": expected_load,
        "reconstructed_store_bytes": reconstructed_store,
        "expected_store_bytes": expected_store,
        "checks": checks,
        "pass": all(checks.values()),
    }


def expected_fu_classes(document: dict[str, Any], metadata: dict[str, Any]) -> list[str]:
    units = document["functional_units"]
    return sorted(
        {
            str(units[operation]["class"])
            for operation, count in metadata["operation_counts"].items()
            if int(count) > 0
        }
    )


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
        name: report["hypothesis_status"]
        == config["frozen_inputs"][name]["required_status"]
        and report["audit_integrity"]
        is config["frozen_inputs"][name]["required_integrity"]
        for name, report in parents.items()
    }
    output_root = PROJECT_ROOT / config["output_root"]
    compile_path = output_root / "fig24-25-exact-compile-manifest.json"
    run_path = output_root / "fig24-25-exact-run-manifest.json"
    compile_file = qualify(compile_path)
    run_file = qualify(run_path)
    compiler = json.loads(compile_path.read_text(encoding="utf-8"))
    run = json.loads(run_path.read_text(encoding="utf-8"))

    compile_checks: dict[str, bool] = {}
    execution_checks: dict[str, bool] = {}
    execution_details: dict[str, dict[str, Any]] = {}
    measurements: dict[str, dict[str, Any]] = {}
    scale_work: dict[str, dict[str, Any]] = {}
    for run_key, item in compiler["outputs"].items():
        path_key, scale_text = run_key.rsplit("-q", maxsplit=1)
        scale = int(scale_text)
        contract = compiler["path_contracts"][path_key]
        document, metadata = recompile(
            run_key=run_key,
            contract=contract,
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
            PROJECT_ROOT / first["summary_path"], {"sha256": first["summary_sha256"]}
        )
        second_summary = qualify(
            PROJECT_ROOT / second["summary_path"], {"sha256": second["summary_sha256"]}
        )
        first_adapter = qualify(
            PROJECT_ROOT / first["adapter_path"], {"sha256": first["adapter_sha256"]}
        )
        second_adapter = qualify(
            PROJECT_ROOT / second["adapter_path"], {"sha256": second["adapter_sha256"]}
        )
        summary = first["summary"]
        adapter = first["adapter"]
        classes = expected_fu_classes(document, metadata)
        physical = summary.get("productive_pe_cycles_by_fu_class", {})
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
            == sum(metadata["pipeline_counts"].values()),
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
            "physical_pes": summary["physical_pe_count"] == 16
            and 0 < summary["mapped_pe_count"] <= 16,
            "physical_fu_classes": sorted(physical) == classes
            and all(int(physical[name]) > 0 for name in classes),
        }
        execution_checks[run_key] = all(checks.values())
        execution_details[run_key] = {
            "checks": checks,
            "expected_fu_classes": classes,
            "physical_fu_cycles": physical,
        }
        measurements[run_key] = {
            "cycles": int(summary["cycles"]),
            "metadata": metadata,
            "physical_fu_cycles": physical,
        }
        scale_work[run_key] = reconstruct_full_work(
            contract=contract,
            metadata=metadata,
            vector_bytes=int(config["hardware"]["vector_bytes"]),
            simd_width=int(config["hardware"]["simd_width"]),
        )

    fit_scales = [int(value) for value in config["fit_scales"]]
    holdout_scales = [int(value) for value in config["holdout_scales"]]
    limit = float(config["cycle_relative_error_limit"])
    models: dict[str, dict[str, Any]] = {}
    errors: list[float] = []
    full_estimates: dict[str, float] = {}
    for path_key, contract in compiler["path_contracts"].items():
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
        full_scale = (
            int(contract["normalized"]["full_scale"])
            if contract["family"] == "qkv_bsmm"
            else int(contract["full_scale"])
        )
        prediction = model.predict(full_scale)
        models[path_key] = {
            "family": contract["family"],
            "intercept": model.intercept,
            "slope_cycles_per_scale": model.slope,
            "holdouts": holdouts,
            "full_scale": full_scale,
            "full_predicted_cycles": prediction,
        }
        full_estimates[path_key] = prediction
    numerical = {
        "passing_holdouts": sum(error <= limit for error in errors),
        "total_holdouts": len(errors),
        "mape": sum(errors) / len(errors),
        "max_error": max(errors),
        "all_holdouts_pass": all(error <= limit for error in errors),
    }
    full_work_checks = {
        key: all(
            scale_work[f"{key}-q{scale}"]["pass"]
            for scale in [*fit_scales, *holdout_scales]
        )
        for key in compiler["path_contracts"]
    }
    summary = {
        "path_count": len(models),
        "run_count": len(measurements),
        "full_estimate_count": len(full_estimates),
        "full_work_passing_paths": sum(full_work_checks.values()),
        "physical_fu_passing_runs": sum(
            item["checks"]["physical_fu_classes"] for item in execution_details.values()
        ),
        **numerical,
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
    integrity_checks = {
        "frozen_files": all(item["pass"] for item in files.values()),
        "parents": all(parent_checks.values()),
        "compile_manifest": compile_file["pass"]
        and compiler["paper_performance_targets_consumed"] is False
        and len(compiler["outputs"]) == 192,
        "run_manifest": run_file["pass"]
        and run["paper_performance_targets_consumed"] is False
        and len(run["records"]) == 192,
        "compiler_replay": all(compile_checks.values()),
        "executions": all(execution_checks.values()),
        "run_replays": all(run["checks"].values()),
        "full_work": all(full_work_checks.values()),
        "source_files": all(item["pass"] for item in source_files.values()),
        "targets_absent": "paper_targets" not in implementation_text,
        "counts": summary["path_count"] == 48
        and summary["run_count"] == 192
        and summary["total_holdouts"] == 96
        and summary["physical_fu_passing_runs"] == 192,
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
        "execution_details": execution_details,
        "scale_work": scale_work,
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
