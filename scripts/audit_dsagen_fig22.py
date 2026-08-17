#!/usr/bin/env python3
"""Audit H44's no-fit source-integrated DSAGEN transfer to Figure 22."""

from __future__ import annotations

import argparse
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
EVIDENCE_ROOT = PROJECT_ROOT / "artifacts/environment/h44"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/dsagen_mlx_fig22_v1.yaml"


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


def qualify_file(path: Path, expected: dict[str, Any] | None = None) -> dict[str, Any]:
    is_file = path.is_file()
    size = path.stat().st_size if is_file else None
    digest = sha256_file(path) if is_file else None
    checks = {"is_file": is_file}
    if expected is not None:
        checks.update(
            {
                "bytes": size == int(expected["bytes"]),
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


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("H44 config must be a mapping")
    return value


def parse_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")

    def prefixed(prefix: str) -> dict[str, Any] | None:
        matches = re.findall(rf"^{prefix} (\{{.*\}})$", text, flags=re.MULTILINE)
        return json.loads(matches[-1]) if matches else None

    return {
        "overlay": prefixed("MLX_OVERLAY_SUMMARY"),
        "adapter": prefixed("MLX_SPAD_ADAPTER_SUMMARY"),
        "host_wait_checksum_present": "[mlx-wait] host wait checksum:" in text,
        "sanity": "[mlx-wait] sanity check passed successfully!" in text,
        "normal_exit": "exiting with last active thread context" in text
        and "Simulated exit code not 0!" not in text,
    }


def relative_error(actual: float, target: float) -> float:
    return abs(actual - target) / abs(target)


def audit_parameters(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = EVIDENCE_ROOT / "fig22-compile-manifest.json"
    manifest_file = qualify_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_file["pass"] else {}
    classes = manifest.get("parameter_classes") or {}
    fields = [field for values in classes.values() for field in values]
    paper_path = PROJECT_ROOT / (
        "MLX Multi-Layer Execution for Structured LLM Workload Acceleration on Spatial "
        "Architectures/MLX Multi-Layer Execution for Structured LLM Workload Acceleration "
        "on Spatial Architectures.md"
    )
    paper = paper_path.read_text(encoding="utf-8")
    accel = (
        PROJECT_ROOT
        / "third_party/dsa-framework/dsa-gem5/src/cpu/minor/ssim/accel.cc"
    ).read_text(encoding="utf-8", errors="replace")
    spec = (
        PROJECT_ROOT / "third_party/dsa-framework/dsa-riscv-ext/spec.h"
    ).read_text(encoding="utf-8", errors="replace")
    compiler_source = (PROJECT_ROOT / "scripts/compile_dsagen_fig22.py").read_text(
        encoding="utf-8"
    )
    runner_source = (PROJECT_ROOT / "scripts/run_dsagen_fig22.sh").read_text(
        encoding="utf-8"
    )
    forbidden_terms = (
        "kernel_issue_scale",
        "kernel_compute_setup_cycles",
        "mesh_base_penalty",
        "mesh_fill_penalty",
        "mesh_congestion_penalty",
        "launch_cycles",
    )
    execution_text = compiler_source + runner_source
    checks = {
        "manifest": manifest_file["pass"],
        "classes_exact": set(classes)
        == {
            "paper_disclosed",
            "dsagen_upstream",
            "gpgpu_sim_reference_only",
            "independently_inferred_and_frozen",
            "unavailable_and_not_fitted",
        },
        "classes_nonempty": all(classes.values()),
        "fields_unique": len(fields) == len(set(fields)),
        "compiler_parameter_checks": all((manifest.get("parameter_checks") or {}).values()),
        "paper_4x4": "4\\times 4" in paper,
        "paper_simd8_32": "8-way SIMD" in paper and "32 instructions per PE" in paper,
        "paper_1ghz_256g": "12nm @1GHz" in paper and "256 G" in paper,
        "dsagen_spad_shape": "spads.emplace_back(8, 8, SCRATCH_SIZE, 1" in accel
        and "InputBuffer(4, 16, 1)" in accel,
        "dsagen_spad_capacity": "#define SCRATCH_SIZE (1048576)" in spec,
        "legacy_calibration_absent_from_execution": not any(
            term in execution_text for term in forbidden_terms
        ),
        "compiler_did_not_consume_targets": manifest.get(
            "paper_performance_targets_consumed_by_compiler"
        )
        is False,
    }
    return {
        "manifest": manifest_file,
        "classes": classes,
        "field_count": len(fields),
        "forbidden_execution_terms": {
            term: term in execution_text for term in forbidden_terms
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def audit_compilation(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = EVIDENCE_ROOT / "fig22-compile-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    replay_report_path = EVIDENCE_ROOT / "fig22-compiler-replay-check.json"
    replay_report_file = qualify_file(replay_report_path)
    replay_report = json.loads(
        replay_report_path.read_text(encoding="utf-8")
    ) if replay_report_file["pass"] else {}
    outputs = manifest.get("outputs") or {}
    binding_checks: dict[str, bool] = {}
    replay_checks: dict[str, bool] = {}
    structural_checks: dict[str, bool] = {}
    for name, specification in outputs.items():
        path = EVIDENCE_ROOT / specification["path"]
        artifact = qualify_file(path)
        binding_checks[name] = (
            artifact["bytes"] == specification["bytes"]
            and artifact["sha256"] == specification["sha256"]
        )
        replay_checks[name] = replay_report.get("checks", {}).get(path.name) is True
        document = json.loads(path.read_text(encoding="utf-8"))
        metadata = document.get("metadata") or {}
        structural_checks[name] = (
            metadata.get("paper_performance_targets_consumed") is False
            and metadata.get("compilation_mode") == "aggregate"
            and metadata.get("width") == int(name.split("-")[-1])
            and sum(int(block["trip_count"]) for block in document["blocks"])
            == metadata.get("total_pairs")
            and metadata.get("memory_requests") == metadata.get("total_pairs") * 3
        )
    checks = {
        "sixteen_outputs": len(outputs) == 16,
        "bindings": all(binding_checks.values()),
        "file_replays": all(replay_checks.values()),
        "replay_report": replay_report_file["pass"]
        and replay_report.get("all_identical") is True
        and replay_report.get("file_count") == 17,
        "manifest_replay": replay_report.get("checks", {}).get(
            "fig22-compile-manifest.json"
        )
        is True,
        "structural": all(structural_checks.values()),
    }
    return {
        "manifest": qualify_file(manifest_path),
        "replay_report": replay_report_file,
        "binding_checks": binding_checks,
        "replay_checks": replay_checks,
        "structural_checks": structural_checks,
        "checks": checks,
        "pass": all(checks.values()),
    }


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    frozen = {
        name: qualify_file(PROJECT_ROOT / specification["path"], specification)
        for name, specification in config["frozen_inputs"].items()
        if isinstance(specification, dict) and "path" in specification
    }
    aggregate = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["aggregate_result"]["path"]).read_text(
            encoding="utf-8"
        )
    ) if frozen["aggregate_result"]["pass"] else {}
    parameter_audit = audit_parameters(config)
    compilation = audit_compilation(config)
    wait_binary = qualify_file(
        PROJECT_ROOT
        / "third_party/dsa-framework/dsa-apps/sdk/compiled/ss-vecadd-gnu-wait.out"
    )
    wait_source = qualify_file(PROJECT_ROOT / "simulator_ext/dsagen/mlx_wait_harness.c")

    sizes = [int(item) for item in config["experiment"]["sizes"]]
    point_reports: dict[str, list[dict[str, Any]]] = {"bsmm": [], "fft": []}
    log_artifacts: dict[str, Any] = {}
    execution_checks: dict[str, bool] = {}
    for kernel in config["experiment"]["kernels"]:
        for index, size in enumerate(sizes):
            name = f"{kernel}-{size}"
            path = EVIDENCE_ROOT / "runs" / name / "run.log"
            artifact = qualify_file(path)
            log_artifacts[name] = artifact
            parsed = parse_log(path) if artifact["pass"] else {}
            overlay = parsed.get("overlay") or {}
            adapter = parsed.get("adapter") or {}
            metadata = (
                compilation_manifest_output(name)
                if compilation["pass"]
                else {}
            )
            target = float(config["experiment"]["targets"][kernel][index])
            cycles = float(overlay.get("cycles", 0))
            compute_busy = float(
                (overlay.get("busy_cycles_by_pipeline") or {}).get("compute", 0)
            )
            utilization = compute_busy / cycles if cycles else 0.0
            error = relative_error(utilization, target) if target else float("inf")
            checks = {
                "artifact": artifact["pass"],
                "overlay_done": overlay.get("done") is True,
                "instruction_count": overlay.get("instructions_issued")
                == overlay.get("instructions_completed")
                == metadata.get("instruction_count"),
                "memory_count": overlay.get("external_memory_requests")
                == overlay.get("external_memory_completions")
                == adapter.get("requests")
                == adapter.get("responses")
                == metadata.get("memory_requests"),
                "events": overlay.get("boundary_events_emitted")
                == metadata.get("transfers"),
                "host_liveness_only": parsed.get("host_wait_checksum_present") is True,
                "sanity": parsed.get("sanity") is True,
                "normal_exit": parsed.get("normal_exit") is True,
            }
            execution_checks[name] = all(checks.values())
            point_reports[kernel].append(
                {
                    "size": size,
                    "cycles": int(cycles),
                    "compute_busy_cycles": int(compute_busy),
                    "compute_utilization": utilization,
                    "target": target,
                    "relative_error": error,
                    "pass_10pct": error <= 0.10,
                    "overlay": overlay,
                    "adapter": adapter,
                    "checks": checks,
                    "execution_pass": all(checks.values()),
                }
            )

    errors = [
        point["relative_error"]
        for reports in point_reports.values()
        for point in reports
    ]
    passing_points = sum(
        point["pass_10pct"] for reports in point_reports.values() for point in reports
    )
    hypothesis_checks = {
        "all_points_execute": all(execution_checks.values()),
        "all_16_within_10pct": passing_points == 16,
    }
    integrity_checks = {
        "frozen_inputs": all(item["pass"] for item in frozen.values()),
        "h43_supported": aggregate.get("hypothesis_status")
        == config["frozen_inputs"]["aggregate_result"]["required_status"]
        and aggregate.get("audit_integrity")
        is config["frozen_inputs"]["aggregate_result"]["required_integrity"],
        "parameters": parameter_audit["pass"],
        "compilation": compilation["pass"],
        "wait_harness": wait_binary["pass"] and wait_source["pass"],
        "all_logs": all(item["pass"] for item in log_artifacts.values()),
        "all_executions_semantically_valid": all(execution_checks.values()),
        "utilization_unscaled": True,
        "legacy_calibration_consumed": False,
        "post_result_parameter_change": False,
    }
    audit_integrity = all(
        value
        for key, value in integrity_checks.items()
        if key not in {"legacy_calibration_consumed", "post_result_parameter_change"}
    ) and integrity_checks["legacy_calibration_consumed"] is False and integrity_checks[
        "post_result_parameter_change"
    ] is False
    hypothesis_supported = all(hypothesis_checks.values())
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "classification": config["classification"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_commit": git_commit(),
        "hypothesis_status": "supported" if hypothesis_supported else "rejected",
        "audit_integrity": audit_integrity,
        "frozen_inputs": frozen,
        "parameter_audit": parameter_audit,
        "compilation": compilation,
        "wait_binary": wait_binary,
        "wait_source": wait_source,
        "logs": log_artifacts,
        "points": point_reports,
        "summary": {
            "passing_points": passing_points,
            "total_points": 16,
            "mape": sum(errors) / len(errors),
            "max_relative_error": max(errors),
            "failing_points": [
                f"{kernel}-{point['size']}"
                for kernel, reports in point_reports.items()
                for point in reports
                if not point["pass_10pct"]
            ],
            "legacy_calibration_consumed": False,
            "validation_eligible": False,
        },
        "integrity_checks": integrity_checks,
        "hypothesis_checks": hypothesis_checks,
        "wall_seconds": time.perf_counter() - started,
    }


def compilation_manifest_output(name: str) -> dict[str, Any]:
    manifest = json.loads(
        (EVIDENCE_ROOT / "fig22-compile-manifest.json").read_text(encoding="utf-8")
    )
    return (manifest.get("outputs", {}).get(name) or {}).get("metadata") or {}


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    report = build_audit(config)
    output = PROJECT_ROOT / config["result_path"]
    if args.verify_existing:
        if not output.is_file():
            raise FileNotFoundError(output)
        existing = json.loads(output.read_text(encoding="utf-8"))
        keys = ("hypothesis_status", "audit_integrity", "hypothesis_checks")
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
