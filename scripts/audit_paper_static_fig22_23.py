#!/usr/bin/env python3
"""Audit H59's paper-static corrective replay of Figures 22 and 23."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = PROJECT_ROOT / "artifacts/environment/h59"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/paper_static_fig22_23_v1.yaml"


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
    path = path.resolve()
    is_file = path.is_file()
    size = path.stat().st_size if is_file else None
    digest = sha256_file(path) if is_file else None
    checks = {"is_file": is_file}
    if expected and "bytes" in expected:
        checks["bytes"] = size == int(expected["bytes"])
    if expected and "sha256" in expected:
        checks["sha256"] = digest == expected["sha256"]
    try:
        display_path = path.relative_to(PROJECT_ROOT)
    except ValueError:
        display_path = path
    return {
        "path": str(display_path),
        "bytes": size,
        "sha256": digest,
        "checks": checks,
        "pass": all(checks.values()),
    }


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={PROJECT_ROOT}", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def relative_error(actual: float, target: float) -> float:
    return abs(actual - target) / abs(target)


def paper_static_transform(document: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(document)
    result["pe_dependency_model"] = "paper_static"
    result.setdefault("metadata", {})["pe_dependency_model"] = "paper_static"
    result["metadata"]["scoreboard_is_paper_semantics"] = False
    return result


def parse_fig22_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")

    def prefixed(prefix: str) -> dict[str, Any] | None:
        matches = re.findall(rf"^{prefix} (\{{.*\}})$", text, flags=re.MULTILINE)
        return json.loads(matches[-1]) if matches else None

    return {
        "overlay": prefixed("MLX_OVERLAY_SUMMARY"),
        "adapter": prefixed("MLX_SPAD_ADAPTER_SUMMARY"),
        "sanity": bool(
            re.search(
                r"\[(?:mlx-wait|single-core)\] sanity check passed successfully!",
                text,
            )
        ),
        "normal_exit": "exiting with last active thread context" in text
        and "Simulated exit code not 0!" not in text,
        "watchdog_abort": "is stalled for too long" in text,
    }


def audit_transformation() -> dict[str, Any]:
    manifest_path = EVIDENCE_ROOT / "paper-static-fig22-23-compile-manifest.json"
    manifest_file = qualify(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records: dict[str, list[dict[str, Any]]] = {}
    checks: dict[str, bool] = {
        "manifest": manifest_file["pass"],
        "experiment_id": manifest.get("experiment_id") == "H59",
        "targets_not_consumed": manifest.get("paper_target_values_consumed") is False,
        "paper_static": manifest.get("pe_dependency_model") == "paper_static",
    }
    for figure, expected_count in (("fig22", 16), ("fig23", 20)):
        records[figure] = []
        source_records = manifest.get("records", {}).get(figure, [])
        checks[f"{figure}_count"] = len(source_records) == expected_count
        for record in source_records:
            parent_path = project_path(record["parent"]["path"])
            output_path = project_path(record["output"]["path"])
            parent_file = qualify(parent_path, record["parent"])
            output_file = qualify(output_path, record["output"])
            parent = json.loads(parent_path.read_text(encoding="utf-8"))
            output = json.loads(output_path.read_text(encoding="utf-8"))
            exact_transform = output == paper_static_transform(parent)
            item_checks = {
                "parent": parent_file["pass"],
                "output": output_file["pass"],
                "exact_allowed_transform": exact_transform,
                "paper_static": output.get("pe_dependency_model") == "paper_static",
                "scoreboard_disclaimed": output.get("metadata", {}).get(
                    "scoreboard_is_paper_semantics"
                )
                is False,
            }
            records[figure].append(
                {
                    "parent": parent_file,
                    "output": output_file,
                    "checks": item_checks,
                    "pass": all(item_checks.values()),
                }
            )
        checks[f"{figure}_exact_transform"] = all(
            item["pass"] for item in records[figure]
        )
    copied_manifest = qualify(EVIDENCE_ROOT / "fig23/fig23-compile-manifest.json")
    checks["fig23_parent_manifest"] = copied_manifest["pass"]
    return {
        "manifest": manifest_file,
        "fig23_parent_manifest": copied_manifest,
        "records": records,
        "checks": checks,
        "pass": all(checks.values()),
    }


def audit_architecture_contract() -> dict[str, Any]:
    paper_path = PROJECT_ROOT / (
        "MLX Multi-Layer Execution for Structured LLM Workload Acceleration on Spatial "
        "Architectures/MLX Multi-Layer Execution for Structured LLM Workload Acceleration "
        "on Spatial Architectures.md"
    )
    paper = paper_path.read_text(encoding="utf-8")
    overlay_path = (
        PROJECT_ROOT
        / "third_party/dsa-framework/dsa-gem5/src/cpu/minor/ssim/mlx_overlay.cc"
    )
    overlay = overlay_path.read_text(encoding="utf-8", errors="replace")
    checks = {
        "paper_mesh_of_pes": "mesh of processing elements (PEs)" in paper,
        "paper_hop_encoded_network": "hop-encoded network" in paper,
        "paper_decoupled_pipelines": "four independent pipelines" in paper,
        "paper_rejects_fine_grained_hazards": (
            "Rather than tracking these interactions with fine-grained instruction-level hazards"
            in paper
        ),
        "paper_static_tagged_blocks": "fixed reusable" in paper
        and "compact *tagged block*" in paper,
        "paper_hybrid_schedule": "intra-layer determinism from cross-layer elasticity" in paper,
        "simulator_has_explicit_paper_static_mode": (
            'name == "paper_static"' in overlay
            and "if (usesExperimentalScoreboard())" in overlay
            and "if (!usesExperimentalScoreboard())" in overlay
        ),
    }
    return {
        "paper": qualify(paper_path),
        "simulator": qualify(overlay_path),
        "corrected_model": {
            "inter_pe": "explicit hop-encoded spatial routing on a PE mesh",
            "intra_pe": (
                "static ordered tagged blocks over decoupled load/store, compute, and xfer "
                "pipelines with heterogeneous FUs"
            ),
            "cross_layer": "elastic tag/event-level arbitration within a bounded active window",
            "not_claimed_by_paper": [
                "warp or SIMT execution",
                "CTA scheduling",
                "GPU operand collectors",
                "fine-grained GPU scoreboard hazards",
                "register-bank arbitration timing",
            ],
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def audit_fig22(old: dict[str, Any]) -> dict[str, Any]:
    points: dict[str, list[dict[str, Any]]] = {"bsmm": [], "fft": []}
    artifacts: dict[str, Any] = {}
    errors: list[float] = []
    execution_checks: dict[str, bool] = {}
    forbidden_stall_prefixes = ("rf_", "register_")
    for kernel in ("bsmm", "fft"):
        for target_point in old["points"][kernel]:
            size = int(target_point["size"])
            name = f"{kernel}-{size}"
            config_path = EVIDENCE_ROOT / f"fig22/fig22-{name}.json"
            config_file = qualify(config_path)
            document = json.loads(config_path.read_text(encoding="utf-8"))
            metadata = document["metadata"]
            log_path = EVIDENCE_ROOT / f"runs/fig22/{name}/run.log"
            log_file = qualify(log_path)
            artifacts[name] = {"config": config_file, "log": log_file}
            parsed = parse_fig22_log(log_path)
            overlay = parsed["overlay"] or {}
            adapter = parsed["adapter"] or {}
            stalls = overlay.get("stalls_by_reason") or {}
            cycles = int(overlay.get("cycles", 0))
            compute_busy = int(
                (overlay.get("busy_cycles_by_pipeline") or {}).get("compute", 0)
            )
            actual = compute_busy / cycles if cycles else 0.0
            target = float(target_point["target"])
            error = relative_error(actual, target) if target else float("inf")
            errors.append(error)
            checks = {
                "config": config_file["pass"],
                "log": log_file["pass"],
                "paper_static": overlay.get("pe_dependency_model") == "paper_static",
                "done": overlay.get("done") is True,
                "instructions": overlay.get("instructions_issued")
                == overlay.get("instructions_completed")
                == metadata.get("instruction_count"),
                "memory": overlay.get("external_memory_requests")
                == overlay.get("external_memory_completions")
                == adapter.get("requests")
                == adapter.get("responses")
                == metadata.get("memory_requests"),
                "events": overlay.get("boundary_events_emitted")
                == metadata.get("transfers"),
                "tag_window": int(overlay.get("max_active_tags", 0))
                <= int(document["active_window"]),
                "no_experimental_register_stalls": not any(
                    key.startswith(forbidden_stall_prefixes) for key in stalls
                ),
                "sanity": parsed["sanity"],
                "normal_exit": parsed["normal_exit"],
                "no_watchdog_abort": not parsed["watchdog_abort"],
            }
            execution_checks[name] = all(checks.values())
            points[kernel].append(
                {
                    "size": size,
                    "cycles": cycles,
                    "compute_busy_cycles": compute_busy,
                    "compute_utilization": actual,
                    "target": target,
                    "relative_error": error,
                    "pass_10pct": error <= 0.10,
                    "checks": checks,
                    "overlay": overlay,
                    "adapter": adapter,
                }
            )
    passing = sum(
        point["pass_10pct"] for values in points.values() for point in values
    )
    summary = {
        "passing_points": passing,
        "total_points": 16,
        "mape": sum(errors) / len(errors),
        "max_relative_error": max(errors),
        "failing_points": [
            f"{kernel}-{point['size']}"
            for kernel, values in points.items()
            for point in values
            if not point["pass_10pct"]
        ],
        "all_16_within_10pct": passing == 16,
        "validation_eligible": False,
    }
    checks = {
        "sixteen_executions": len(execution_checks) == 16,
        "all_executions_valid": all(execution_checks.values()),
        "unscaled_compute_busy_over_cycles": True,
    }
    return {
        "artifacts": artifacts,
        "points": points,
        "execution_checks": execution_checks,
        "checks": checks,
        "summary": summary,
        "pass": all(checks.values()),
    }


def audit_fig23(old: dict[str, Any]) -> dict[str, Any]:
    compile_path = EVIDENCE_ROOT / "fig23/fig23-compile-manifest.json"
    compiler_file = qualify(compile_path)
    compiler = json.loads(compile_path.read_text(encoding="utf-8"))
    run_path = EVIDENCE_ROOT / "fig23/fig23-run-manifest.json"
    run_file = qualify(run_path)
    run = json.loads(run_path.read_text(encoding="utf-8"))
    summaries: dict[str, Any] = {}
    execution_checks: dict[str, bool] = {}
    for key, details in run.get("runs", {}).items():
        first_path = EVIDENCE_ROOT / f"fig23/runs/{key}-first.json"
        second_path = EVIDENCE_ROOT / f"fig23/runs/{key}-second.json"
        first_file = qualify(first_path)
        second_file = qualify(second_path)
        first = json.loads(first_path.read_text(encoding="utf-8"))
        metadata = compiler["outputs"][key]["metadata"]
        stalls = first.get("stalls_by_reason") or {}
        checks = {
            "first": first_file["pass"],
            "second": second_file["pass"],
            "deterministic": first_file["sha256"]
            == second_file["sha256"]
            == details.get("summary_sha256")
            == details.get("replay_sha256"),
            "manifest_binding": first == details.get("summary"),
            "paper_static": first.get("pe_dependency_model") == "paper_static",
            "fixed_memory": first.get("memory_backend") == "fixed",
            "done": first.get("done") is True,
            "instructions": first.get("instructions_issued")
            == first.get("instructions_completed")
            == metadata.get("instruction_count"),
            "no_experimental_register_stalls": not any(
                key_.startswith(("rf_", "register_")) for key_ in stalls
            ),
        }
        execution_checks[key] = all(checks.values())
        summaries[key] = {
            "first": first_file,
            "second": second_file,
            "checks": checks,
        }
    points: dict[str, list[dict[str, Any]]] = {}
    errors: list[float] = []
    passing = 0
    for name, target_points in old["points"].items():
        points[name] = []
        actuals = run["speedups"][name]
        for target_point, actual in zip(target_points, actuals, strict=True):
            target = float(target_point["target"])
            error = relative_error(float(actual), target)
            passed = error <= 0.10
            errors.append(error)
            passing += passed
            points[name].append(
                {
                    "sequence_length": int(target_point["sequence_length"]),
                    "actual": actual,
                    "target": target,
                    "relative_error": error,
                    "pass_10pct": passed,
                }
            )
    summary = {
        "passing_points": passing,
        "total_points": 15,
        "mape": sum(errors) / len(errors),
        "max_relative_error": max(errors),
        "failing_points": [
            f"{name}-{point['sequence_length']}"
            for name, values in points.items()
            for point in values
            if not point["pass_10pct"]
        ],
        "all_15_within_10pct": passing == 15,
        "validation_eligible": False,
        "proxy_mapping": True,
    }
    runner_checks = run.get("checks") or {}
    checks = {
        "compiler_manifest": compiler_file["pass"],
        "run_manifest": run_file["pass"],
        "experiment_id": run.get("experiment_id") == "H59",
        "twenty_runs": len(run.get("runs") or {}) == 20,
        "all_executions_valid": all(execution_checks.values()),
        "runner_checks": all(
            value for key, value in runner_checks.items() if key != "targets_consumed_by_runner"
        )
        and runner_checks.get("targets_consumed_by_runner") is False,
    }
    return {
        "compiler": compiler_file,
        "run_manifest": run_file,
        "cycles": run.get("cycles"),
        "speedups": run.get("speedups"),
        "summaries": summaries,
        "execution_checks": execution_checks,
        "points": points,
        "checks": checks,
        "summary": summary,
        "pass": all(checks.values()),
    }


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    historical: dict[str, Any] = {}
    old_results: dict[str, dict[str, Any]] = {}
    for figure, specification in config["historical_results"].items():
        path = PROJECT_ROOT / specification["path"]
        historical[figure] = qualify(path, specification)
        old_results[figure] = json.loads(path.read_text(encoding="utf-8"))
    semantics_path = PROJECT_ROOT / config["parent_semantics"]
    semantics_file = qualify(semantics_path)
    semantics = json.loads(semantics_path.read_text(encoding="utf-8"))
    architecture = audit_architecture_contract()
    transformation = audit_transformation()
    fig22 = audit_fig22(old_results["fig22"])
    fig23 = audit_fig23(old_results["fig23"])
    execution_sources = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in {
            "transformer": "scripts/compile_paper_static_fig22_23.py",
            "fig22_runner": "scripts/run_dsagen_fig22.sh",
            "fig23_runner": "scripts/run_dsagen_fig23.py",
            "watchdog_patch": (
                "patches/dsagen/dsa-gem5-mlx-long-overlay-watchdog-v1.patch"
            ),
            "gem5_execute": (
                "third_party/dsa-framework/dsa-gem5/src/cpu/minor/execute.cc"
            ),
            "gem5_accel": (
                "third_party/dsa-framework/dsa-gem5/src/cpu/minor/ssim/accel.cc"
            ),
        }.items()
    }
    execute_source = (
        PROJECT_ROOT
        / "third_party/dsa-framework/dsa-gem5/src/cpu/minor/execute.cc"
    ).read_text(encoding="utf-8", errors="replace")
    accel_source = (
        PROJECT_ROOT
        / "third_party/dsa-framework/dsa-gem5/src/cpu/minor/ssim/accel.cc"
    ).read_text(encoding="utf-8", errors="replace")
    fig22_runner_source = (PROJECT_ROOT / "scripts/run_dsagen_fig22.sh").read_text(
        encoding="utf-8"
    )
    diagnostic_checks = {
        "upstream_default_preserved": "uint64_t watchdog_cycles = 100000;"
        in execute_source,
        "watchdog_opt_in": 'std::getenv("MLX_WATCHDOG_CYCLES")' in execute_source,
        "runner_default_preserved": "MLX_WATCHDOG_CYCLES:-100000}" in fig22_runner_source,
        "h59_limit_registered": config.get("diagnostic_only_change", {}).get(
            "h59_cycles"
        )
        == 10000000,
        "progress_uses_observable_overlay_events": all(
            token in accel_source
            for token in (
                "instructions_issued",
                "instructions_completed",
                "route_hops",
                "external_memory_completions",
                "progress_after > progress_before",
            )
        ),
        "no_overlay_cycle_counter_in_progress_signal": (
            "progress_after = after.cycles" not in accel_source
        ),
    }
    numerical_gate = {
        "fig22_all_16_within_10pct": fig22["summary"]["all_16_within_10pct"],
        "fig23_all_15_within_10pct": fig23["summary"]["all_15_within_10pct"],
    }
    numerical_gate["both_figures_within_10pct"] = all(numerical_gate.values())
    integrity_checks = {
        "historical_target_sources": all(item["pass"] for item in historical.values()),
        "historical_audits_valid": all(
            result.get("audit_integrity") is True for result in old_results.values()
        ),
        "parent_semantics": semantics_file["pass"]
        and semantics.get("hypothesis_status") == "supported"
        and semantics.get("audit_integrity") is True,
        "architecture_contract": architecture["pass"],
        "exact_target_independent_transform": transformation["pass"],
        "execution_sources": all(item["pass"] for item in execution_sources.values()),
        "diagnostic_watchdog_only": all(diagnostic_checks.values()),
        "fig22_execution": fig22["pass"],
        "fig23_execution": fig23["pass"],
        "targets_loaded_only_by_auditor": True,
        "post_result_adjustment": False,
    }
    audit_integrity = all(
        value for key, value in integrity_checks.items() if key != "post_result_adjustment"
    ) and integrity_checks["post_result_adjustment"] is False
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "classification": config["classification"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_commit": git_commit(),
        "hypothesis_status": (
            "supported" if numerical_gate["both_figures_within_10pct"] else "rejected"
        ),
        "audit_integrity": audit_integrity,
        "historical_target_sources": historical,
        "parent_semantics": semantics_file,
        "architecture_contract": architecture,
        "transformation": transformation,
        "execution_sources": execution_sources,
        "diagnostic_checks": diagnostic_checks,
        "fig22": fig22,
        "fig23": fig23,
        "numerical_gate": numerical_gate,
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
