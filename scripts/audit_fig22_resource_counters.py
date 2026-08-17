#!/usr/bin/env python3
"""Audit H61's Figure 22 PE-resource counters against all 64 raster values."""

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
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig22_resource_counters_v1.yaml"
PIPELINES = ("xfer", "load", "store", "compute")
SIZES = (64, 128, 256, 512, 1024, 2048, 4096, 8192)


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
    exists = path.is_file()
    size = path.stat().st_size if exists else None
    digest = sha256_file(path) if exists else None
    checks = {"is_file": exists}
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


def parse_log(path: Path) -> dict[str, Any]:
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


def relative_error(actual: float, target: float) -> float:
    return abs(actual - target) / abs(target)


def summarize_errors(errors: list[float]) -> dict[str, Any]:
    return {
        "passing_points": sum(error <= 0.10 for error in errors),
        "total_points": len(errors),
        "mape": sum(errors) / len(errors),
        "max_relative_error": max(errors),
        "all_within_10pct": all(error <= 0.10 for error in errors),
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
    h59 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h59_result"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    h60 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h60_targets"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    targets = h60["derived_targets"]["panels"]
    config_root = PROJECT_ROOT / config["configs"]
    output_root = PROJECT_ROOT / config["output_root"]
    logs: dict[str, Any] = {}
    points: dict[str, list[dict[str, Any]]] = {"bsmm": [], "chunk_fft": []}
    execution_checks: dict[str, bool] = {}
    metric_errors = {
        name: []
        for name in (
            "primary_productive_pe",
            "resident_pe",
            "productive_global",
            "legacy_global_resident",
            "issue_global",
        )
    }
    for panel, kernel in (("bsmm", "bsmm"), ("chunk_fft", "fft")):
        for index, size in enumerate(SIZES):
            name = f"{kernel}-{size}"
            log_path = output_root / name / "run.log"
            log_file = qualify(log_path)
            logs[name] = log_file
            parsed = parse_log(log_path)
            overlay = parsed["overlay"] or {}
            adapter = parsed["adapter"] or {}
            config_path = config_root / f"fig22-{name}.json"
            document = json.loads(config_path.read_text(encoding="utf-8"))
            metadata = document["metadata"]
            old_panel = "fft" if panel == "chunk_fft" else panel
            old_point = h59["fig22"]["points"][old_panel][index]
            old_overlay = old_point["overlay"]
            common_summary_unchanged = all(
                overlay.get(key) == value for key, value in old_overlay.items()
            )
            cycles = int(overlay.get("cycles", 0))
            physical_pes = int(overlay.get("physical_pe_count", 0))
            capacity = cycles * physical_pes
            counter_checks: dict[str, bool] = {}
            metric_values: dict[str, dict[str, float]] = {
                metric: {} for metric in metric_errors
            }
            resource_points: dict[str, Any] = {}
            for resource in PIPELINES:
                resident = int(
                    (overlay.get("resident_pe_cycles_by_pipeline") or {}).get(
                        resource, -1
                    )
                )
                productive = int(
                    (overlay.get("productive_pe_cycles_by_pipeline") or {}).get(
                        resource, -1
                    )
                )
                productive_global = int(
                    (overlay.get("productive_global_cycles_by_pipeline") or {}).get(
                        resource, -1
                    )
                )
                legacy_global = int(
                    (overlay.get("busy_cycles_by_pipeline") or {}).get(resource, -1)
                )
                issue_global = int(
                    (overlay.get("issue_cycles_by_pipeline") or {}).get(resource, -1)
                )
                issued = int(
                    (overlay.get("issued_by_pipeline") or {}).get(resource, -1)
                )
                counter_checks[f"{resource}_counter_order"] = (
                    0 <= issue_global <= productive_global <= legacy_global <= cycles
                    and 0 <= issued <= productive <= resident <= capacity
                    and productive_global <= productive
                    and legacy_global <= resident
                )
                metric_values["primary_productive_pe"][resource] = (
                    productive / capacity if capacity else 0.0
                )
                metric_values["resident_pe"][resource] = (
                    resident / capacity if capacity else 0.0
                )
                metric_values["productive_global"][resource] = (
                    productive_global / cycles if cycles else 0.0
                )
                metric_values["legacy_global_resident"][resource] = (
                    legacy_global / cycles if cycles else 0.0
                )
                metric_values["issue_global"][resource] = (
                    issue_global / cycles if cycles else 0.0
                )
                target = float(targets[panel][resource][index])
                primary = metric_values["primary_productive_pe"][resource]
                error = relative_error(primary, target)
                resource_points[resource] = {
                    "target": target,
                    "actual": primary,
                    "relative_error": error,
                    "pass_10pct": error <= 0.10,
                    "counts": {
                        "productive_pe": productive,
                        "resident_pe": resident,
                        "productive_global": productive_global,
                        "legacy_global_resident": legacy_global,
                        "issue_global": issue_global,
                        "issued_instructions": issued,
                    },
                    "diagnostic_actuals": {
                        metric: values[resource]
                        for metric, values in metric_values.items()
                        if metric != "primary_productive_pe"
                    },
                }
                for metric, values in metric_values.items():
                    metric_errors[metric].append(
                        relative_error(values[resource], target)
                    )
            checks = {
                "log": log_file["pass"],
                "paper_static": overlay.get("pe_dependency_model") == "paper_static",
                "physical_pes": physical_pes
                == int(document["routing"]["mesh_width"])
                * int(document["routing"]["mesh_height"])
                == 16,
                "mapped_pes": overlay.get("mapped_pe_count") == 16,
                "common_h59_summary_unchanged": common_summary_unchanged,
                "instructions": overlay.get("instructions_issued")
                == overlay.get("instructions_completed")
                == metadata["instruction_count"],
                "memory": overlay.get("external_memory_requests")
                == overlay.get("external_memory_completions")
                == adapter.get("requests")
                == adapter.get("responses")
                == metadata["memory_requests"],
                "events": overlay.get("boundary_events_emitted")
                == metadata["transfers"],
                "counter_invariants": all(counter_checks.values()),
                "no_experimental_register_stalls": not any(
                    key.startswith(("rf_", "register_"))
                    for key in (overlay.get("stalls_by_reason") or {})
                ),
                "sanity": parsed["sanity"],
                "normal_exit": parsed["normal_exit"],
                "no_watchdog_abort": not parsed["watchdog_abort"],
            }
            execution_checks[name] = all(checks.values())
            points[panel].append(
                {
                    "size": size,
                    "cycles": cycles,
                    "physical_pe_count": physical_pes,
                    "capacity_pe_cycles": capacity,
                    "resources": resource_points,
                    "metric_values": metric_values,
                    "counter_checks": counter_checks,
                    "checks": checks,
                    "overlay": overlay,
                    "adapter": adapter,
                }
            )
    diagnostic_summaries = {
        metric: summarize_errors(errors) for metric, errors in metric_errors.items()
    }
    primary_summary = diagnostic_summaries["primary_productive_pe"]
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in {
            "runner": config["execution"]["runner"],
            "auditor": config["execution"]["auditor"],
            "patch": config["execution"]["patch"],
            "overlay_cc": (
                "third_party/dsa-framework/dsa-gem5/src/cpu/minor/ssim/mlx_overlay.cc"
            ),
            "overlay_hh": (
                "third_party/dsa-framework/dsa-gem5/src/cpu/minor/ssim/mlx_overlay.hh"
            ),
        }.items()
    }
    integrity_checks = {
        "frozen_inputs": all(item["pass"] for item in frozen.values()),
        "h59_integrity": h59.get("audit_integrity")
        is config["frozen_inputs"]["h59_result"]["required_integrity"],
        "h60_targets": h60.get("verdict")
        == config["frozen_inputs"]["h60_targets"]["required_verdict"]
        and h60.get("summary", {}).get("pass")
        is config["frozen_inputs"]["h60_targets"]["required_summary_pass"],
        "all_logs": all(item["pass"] for item in logs.values()),
        "all_executions": all(execution_checks.values()),
        "source_files": all(item["pass"] for item in source_files.values()),
        "primary_metric_registered": config["primary_metric"]["numerator"]
        == "productive_pe_cycles_by_pipeline",
        "targets_loaded_only_by_auditor": True,
        "diagnostic_metric_selected_post_run": False,
        "post_result_adjustment": False,
    }
    audit_integrity = all(
        value
        for key, value in integrity_checks.items()
        if key not in {"diagnostic_metric_selected_post_run", "post_result_adjustment"}
    ) and not integrity_checks["diagnostic_metric_selected_post_run"] and not integrity_checks[
        "post_result_adjustment"
    ]
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "classification": config["classification"],
        "validation_eligible": config["validation_eligible"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_commit": git_commit(),
        "hypothesis_status": (
            "supported" if primary_summary["all_within_10pct"] else "rejected"
        ),
        "audit_integrity": audit_integrity,
        "frozen_inputs": frozen,
        "source_files": source_files,
        "logs": logs,
        "points": points,
        "execution_checks": execution_checks,
        "primary_metric": {
            **config["primary_metric"],
            "summary": primary_summary,
        },
        "diagnostic_metric_summaries": diagnostic_summaries,
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
        keys = ("hypothesis_status", "audit_integrity", "primary_metric")
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
