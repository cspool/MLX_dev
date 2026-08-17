#!/usr/bin/env python3
"""Audit H62's target-free Figure 10 compiler and execution smokes."""

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
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig10_mapping_v1.yaml"


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
    if expected and "bytes" in expected:
        checks["bytes"] = size == int(expected["bytes"])
    if expected and "sha256" in expected:
        checks["sha256"] = digest == expected["sha256"]
    try:
        display = path.relative_to(PROJECT_ROOT)
    except ValueError:
        display = path
    return {
        "path": str(display),
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
        "sanity": "sanity check passed successfully!" in text,
        "normal_exit": "exiting with last active thread context" in text
        and "Simulated exit code not 0!" not in text,
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


def structural_checks(document: dict[str, Any], metadata: dict[str, Any]) -> dict[str, bool]:
    width = int(metadata["width"])
    stages = int(metadata["stages"])
    operator = metadata["operator"]
    expected_template = 5 if operator == "bsmm" else 6
    blocks = document["blocks"]
    cdc_starts = set(metadata["cdc_starts"])
    cdc_ends = set(metadata["cdc_ends"])
    expected_outputs = width * stages
    local_loads = 0
    external_loads = 0
    stores = 0
    xfers = 0
    for block in blocks:
        stage = int(block["tag"]) - 1
        instructions = block["instructions"]
        if len(instructions) != expected_template:
            return {"template_length": False}
        for instruction in instructions:
            if instruction["pipeline"] == "load":
                if instruction["memory_external"]:
                    external_loads += int(block["trip_count"])
                else:
                    local_loads += int(block["trip_count"])
            if instruction["pipeline"] == "store":
                stores += int(block["trip_count"])
            if instruction["pipeline"] == "xfer":
                xfers += int(block["trip_count"])
        terminal = instructions[-1]["operation"]
        if (stage in cdc_ends) != (terminal == "store"):
            return {"terminal_by_cdc": False}
        if stage in cdc_starts:
            if not all(item["memory_external"] for item in instructions[:2]):
                return {"cdc_start_external": False}
        elif not all(not item["memory_external"] for item in instructions[:2]):
            return {"internal_load_local": False}
    expected_pipeline = metadata["expected_pipeline_instructions"]
    return {
        "block_count": len(blocks) == 16 * stages,
        "trip_count": {int(block["trip_count"]) for block in blocks} == {width // 16},
        "output_conservation": sum(int(block["trip_count"]) for block in blocks)
        == expected_outputs,
        "instruction_conservation": sum(
            len(block["instructions"]) * int(block["trip_count"])
            for block in blocks
        )
        == int(metadata["instruction_count"]),
        "external_loads": external_loads == int(metadata["external_loads"]),
        "local_loads": local_loads
        == 2 * expected_outputs - int(metadata["external_loads"]),
        "stores": stores == int(metadata["external_stores"]),
        "xfers": xfers == int(metadata["transfers"]),
        "pipeline_sum": sum(int(value) for value in expected_pipeline.values())
        == int(metadata["instruction_count"]),
        "footprint": int(metadata["max_active_instruction_footprint_per_pe"]) <= 32,
        "paper_loops": metadata["local_i1_trip"] == 4
        and metadata["spatial_i2_trip"] == 16
        and metadata["outer_i0_trip"] == width // 64,
        "no_targets": metadata["paper_performance_targets_consumed"] is False,
    }


def smoke_checks(
    summary: dict[str, Any], metadata: dict[str, Any], *, adapter: dict[str, Any] | None
) -> dict[str, bool]:
    expected_pipeline = metadata["expected_pipeline_instructions"]
    checks = {
        "done": summary.get("done") is True,
        "paper_static": summary.get("pe_dependency_model") == "paper_static",
        "physical_pes": summary.get("physical_pe_count") == 16,
        "mapped_pes": summary.get("mapped_pe_count") == 16,
        "instructions": summary.get("instructions_issued")
        == summary.get("instructions_completed")
        == metadata["instruction_count"],
        "pipeline_counts": all(
            summary.get("issued_by_pipeline", {}).get(name) == count
            for name, count in expected_pipeline.items()
        ),
        "events": summary.get("boundary_events_emitted")
        == metadata["boundary_events"],
        "routes": summary.get("route_hops") == metadata["route_hops"],
        "counter_order": all(
            0
            <= summary["productive_pe_cycles_by_pipeline"][name]
            <= summary["resident_pe_cycles_by_pipeline"][name]
            <= summary["cycles"] * 16
            for name in ("load", "store", "compute", "xfer")
        ),
    }
    if adapter is None:
        checks["fixed_memory"] = summary.get("memory_backend") == "fixed"
        checks["no_external_requests"] = summary.get("external_memory_requests") == 0
    else:
        checks["adapter_memory"] = summary.get("memory_backend") == "adapter"
        checks["external_requests"] = (
            summary.get("external_memory_requests")
            == summary.get("external_memory_completions")
            == adapter.get("requests")
            == adapter.get("responses")
            == metadata["memory_requests"]
        )
    return checks


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    root = PROJECT_ROOT / config["experiment"]["output_root"]
    manifest_path = root / "fig10-compile-manifest.json"
    manifest_file = qualify(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    outputs: dict[str, Any] = {}
    compilation_checks: dict[str, bool] = {}
    for key, record in manifest["outputs"].items():
        primary_path = PROJECT_ROOT / record["primary"]["path"]
        replay_path = PROJECT_ROOT / record["replay"]["path"]
        primary_file = qualify(primary_path, record["primary"])
        replay_file = qualify(replay_path, record["replay"])
        document = json.loads(primary_path.read_text(encoding="utf-8"))
        checks = structural_checks(document, record["metadata"])
        checks.update(
            {
                "primary": primary_file["pass"],
                "replay": replay_file["pass"],
                "byte_identical": primary_file["sha256"] == replay_file["sha256"],
                "record_identical": record["identical"] is True,
            }
        )
        compilation_checks[key] = all(checks.values())
        outputs[key] = {
            "primary": primary_file,
            "replay": replay_file,
            "checks": checks,
        }

    standalone: dict[str, Any] = {}
    gem5: dict[str, Any] = {}
    smoke_passes: list[bool] = []
    for operator in config["experiment"]["operators"]:
        key = f"{operator}-64"
        metadata = manifest["outputs"][key]["metadata"]
        first_path = root / f"runs/standalone/{operator}-64-first.json"
        second_path = root / f"runs/standalone/{operator}-64-second.json"
        first_file = qualify(first_path)
        second_file = qualify(second_path)
        summary = json.loads(first_path.read_text(encoding="utf-8"))
        checks = smoke_checks(summary, metadata, adapter=None)
        checks["replay"] = first_file["sha256"] == second_file["sha256"]
        standalone[operator] = {
            "first": first_file,
            "second": second_file,
            "summary": summary,
            "checks": checks,
            "pass": all(checks.values()),
        }
        smoke_passes.append(all(checks.values()))

        log_path = root / f"runs/gem5/{operator}-64/run.log"
        log_file = qualify(log_path)
        parsed = parse_log(log_path)
        summary = parsed["overlay"] or {}
        adapter = parsed["adapter"] or {}
        checks = smoke_checks(summary, metadata, adapter=adapter)
        checks.update(
            {
                "log": log_file["pass"],
                "sanity": parsed["sanity"],
                "normal_exit": parsed["normal_exit"],
            }
        )
        gem5[operator] = {
            "log": log_file,
            "summary": summary,
            "adapter": adapter,
            "checks": checks,
            "pass": all(checks.values()),
        }
        smoke_passes.append(all(checks.values()))

    h52_old = json.loads(
        (PROJECT_ROOT / "artifacts/environment/h52/runs/standalone/opt-summary.json").read_text(
            encoding="utf-8"
        )
    )
    h52_new_path = root / "runs/compat/h52-first.json"
    h52_replay_path = root / "runs/compat/h52-second.json"
    h52_new = json.loads(h52_new_path.read_text(encoding="utf-8"))
    h52_checks = {
        "files": qualify(h52_new_path)["pass"] and qualify(h52_replay_path)["pass"],
        "replay": sha256_file(h52_new_path) == sha256_file(h52_replay_path),
        "legacy_fields": all(h52_new.get(key) == value for key, value in h52_old.items()),
    }
    old_log = PROJECT_ROOT / "artifacts/environment/h61/runs/bsmm-64/run.log"
    new_log = root / "runs/compat-gem5/bsmm-64/run.log"
    old_parsed = parse_log(old_log)
    new_parsed = parse_log(new_log)
    compat_gem5_checks = {
        "log": qualify(new_log)["pass"],
        "overlay": new_parsed["overlay"] == old_parsed["overlay"],
        "adapter": new_parsed["adapter"] == old_parsed["adapter"],
        "sanity": new_parsed["sanity"],
        "normal_exit": new_parsed["normal_exit"],
    }
    paper_path = PROJECT_ROOT / (
        "MLX Multi-Layer Execution for Structured LLM Workload Acceleration on Spatial "
        "Architectures/MLX Multi-Layer Execution for Structured LLM Workload Acceleration "
        "on Spatial Architectures.md"
    )
    paper = paper_path.read_text(encoding="utf-8")
    paper_checks = {
        "i2_spatial": "innermost loop i2 is fully unrolled" in paper,
        "i1_local": "middle loop i1 runs locally within each PE" in paper,
        "closed_64": "closed-set sample of 64 output elements" in paper,
        "tagged_order": "loads at the beginning, comps in the middle, and xfers at the end"
        in paper,
        "stride_vertical": "stride=4, 8" in paper and "vertically" in paper,
        "simd8": "8-way SIMD is a necessary *lower bound*" in paper,
        "instruction_store": "32 instructions per PE" in paper,
    }
    sources = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in {
            "paper": paper_path.relative_to(PROJECT_ROOT),
            "compiler_module": config["source_layout"]["compiler_module"],
            "compiler": config["source_layout"]["compiler"],
            "runner": config["source_layout"]["runner"],
            "auditor": config["source_layout"]["auditor"],
            "memory_patch": config["source_layout"]["memory_patch"],
        }.items()
    }
    source_text = "\n".join(
        (PROJECT_ROOT / path).read_text(encoding="utf-8")
        for path in (
            config["source_layout"]["compiler_module"],
            config["source_layout"]["compiler"],
            config["source_layout"]["runner"],
        )
    )
    integrity_checks = {
        "manifest": manifest_file["pass"],
        "sixteen_outputs": manifest.get("output_count") == 16,
        "manifest_replay": manifest.get("all_identical") is True,
        "manifest_no_targets": manifest.get("paper_performance_targets_consumed")
        is False,
        "all_compilations": all(compilation_checks.values()),
        "all_smokes": all(smoke_passes),
        "h52_compat": all(h52_checks.values()),
        "gem5_8byte_compat": all(compat_gem5_checks.values()),
        "paper_contract": all(paper_checks.values()),
        "sources": all(item["pass"] for item in sources.values()),
        "target_paths_absent": "paper_targets" not in source_text
        and "fig22_resource" not in source_text,
        "numerical_figure22_comparison_performed": False,
        "post_result_adjustment": False,
    }
    audit_integrity = all(
        value
        for key, value in integrity_checks.items()
        if key not in {
            "numerical_figure22_comparison_performed",
            "post_result_adjustment",
        }
    ) and not integrity_checks["numerical_figure22_comparison_performed"] and not integrity_checks[
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
        "hypothesis_status": "supported" if audit_integrity else "rejected",
        "audit_integrity": audit_integrity,
        "manifest": manifest_file,
        "outputs": outputs,
        "compilation_checks": compilation_checks,
        "standalone_smokes": standalone,
        "gem5_smokes": gem5,
        "compatibility": {
            "h52": h52_checks,
            "gem5_8byte": compat_gem5_checks,
        },
        "paper_checks": paper_checks,
        "sources": sources,
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
