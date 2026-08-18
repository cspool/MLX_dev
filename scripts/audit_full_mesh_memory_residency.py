#!/usr/bin/env python3
"""Audit H107 full batch-32 memory residency and operational intensity."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from mlxsim.dsagen_overlay import canonical_json
from mlxsim.full_mesh_memory_residency import compile_residency_path

try:
    from scripts.collect_h107_memory_contracts import build_snapshot
except ModuleNotFoundError:  # Direct script execution places scripts/ on sys.path.
    from collect_h107_memory_contracts import build_snapshot

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/simulators/full_mesh_memory_residency_v1.yaml"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qualify(path: Path, expected: dict[str, Any] | None = None) -> dict[str, Any]:
    exists = path.is_file()
    digest = sha256_file(path) if exists else None
    checks = {"is_file": exists}
    if expected and "sha256" in expected:
        checks["sha256"] = digest == expected["sha256"]
    if expected and "bytes" in expected:
        checks["bytes"] = exists and path.stat().st_size == int(expected["bytes"])
    if exists and expected and (
        "required_status" in expected or "required_integrity" in expected
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "required_status" in expected:
            checks["status"] = (
                payload.get("hypothesis_status") == expected["required_status"]
            )
        if "required_integrity" in expected:
            checks["integrity"] = (
                payload.get("audit_integrity") is expected["required_integrity"]
            )
    return {
        "path": str(path.relative_to(PROJECT_ROOT)) if exists else str(path),
        "bytes": path.stat().st_size if exists else None,
        "sha256": digest,
        "checks": checks,
        "pass": all(checks.values()),
    }


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def debug_records(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        item["path"]: item
        for item in run["records"]
        if item["mode"] == "debug" and item["replay"] == 1
    }


def formula_checks(metadata: dict[str, Any]) -> dict[str, bool]:
    family = metadata["family"]
    case, operator = metadata["case"], metadata["operator"]
    batch, n, dimension = (
        int(case["batch"]),
        int(case["n"]),
        int(case["d"]),
    )
    element_bytes = 2
    if family == "fft":
        retained = n // 2
        fma = (
            4
            * 3
            * batch
            * dimension
            * (
                n // 2 * int(math.log2(n))
                + retained // 2 * int(math.log2(retained))
            )
        )
        read = 3 * batch * n * dimension * element_bytes
        write = 3 * batch * retained * dimension * element_bytes
        return {
            "fma": metadata["fma_count"] == fma,
            "read": metadata["selected_read_bytes"] == read,
            "write": metadata["selected_write_bytes"] == write,
        }
    if family == "qkv_bsmm":
        block = int(operator["block_size"])
        density_numerator = 2 * int(math.log2(block))
        activation = batch * n * dimension
        output = 3 * activation
        weights = 3 * dimension * dimension * density_numerator // block
        return {
            "fma": metadata["fma_count"]
            == output * dimension * density_numerator // block,
            "read": metadata["selected_read_bytes"]
            == (activation + weights) * element_bytes,
            "write": metadata["selected_write_bytes"]
            == output * element_bytes,
        }
    window, query = int(operator["window"]), int(operator["query_tile"])
    tensor = batch * n * dimension * element_bytes
    return {
        "fma": metadata["fma_count"] == 2 * batch * n * window * dimension,
        "lower_read": metadata["lower_bound_read_bytes"] == 3 * tensor,
        "selected_read": metadata["selected_read_bytes"]
        == batch * (n // query) * (query + 2 * window) * dimension * element_bytes,
        "write": metadata["selected_write_bytes"] == tensor,
    }


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / item["path"], item)
        for name, item in config["frozen_inputs"].items()
    }
    contracts_path = PROJECT_ROOT / config["frozen_inputs"]["contracts"]["path"]
    contracts = json.loads(contracts_path.read_text(encoding="utf-8"))
    snapshot_reproduces = build_snapshot() == contracts
    output_root = PROJECT_ROOT / config["output_root"]
    compile_path = output_root / "full-mesh-memory-residency-compile-manifest.json"
    run_path = output_root / "full-mesh-memory-residency-run-manifest.json"
    compiled = json.loads(compile_path.read_text(encoding="utf-8"))
    run = json.loads(run_path.read_text(encoding="utf-8"))
    records = debug_records(run)
    path_checks: dict[str, dict[str, bool]] = {}
    path_results: dict[str, dict[str, Any]] = {}
    selected_ois = []
    lower_ois = []
    selected_bytes = []
    tile_counts = []
    for key, contract in contracts["paths"].items():
        output = compiled["outputs"][key]
        memory_config, expected_metadata = compile_residency_path(
            key=key, contract=contract, config=config
        )
        path = PROJECT_ROOT / output["artifact"]["path"]
        stored_config = json.loads(path.read_text(encoding="utf-8"))
        metadata = output["metadata"]
        record = records[key]
        summary = record["summary"]
        memory = summary["memory"]
        formulas = formula_checks(metadata)
        input_tiles = metadata["input_bytes_by_tile"]
        output_tiles = metadata["output_bytes_by_tile"]
        expected_dma_cycles = sum(
            (value + int(config["hardware"]["dma_bytes_per_cycle"]) - 1)
            // int(config["hardware"]["dma_bytes_per_cycle"])
            for value in [*input_tiles, *output_tiles]
        )
        packing = (
            sum(input_tiles) == metadata["selected_read_bytes"]
            and sum(output_tiles) == metadata["selected_write_bytes"]
            and all(
                value > 0
                and value <= int(config["hardware"]["half_bytes"])
                and value % int(config["hardware"]["tile_alignment_bytes"]) == 0
                for value in [*input_tiles, *output_tiles]
            )
        )
        execution_bytes = (
            memory["offchip_read_bytes"] == metadata["selected_read_bytes"]
            and memory["offchip_write_bytes"] == metadata["selected_write_bytes"]
        )
        completion = (
            memory["tile_count"]
            == memory["released_tiles"]
            == memory["drained_tiles"]
            == metadata["tile_count"]
            and memory["idle"]
            and memory["buffer_owners"] == ["dma", "dma"]
            and memory["ownership_violations"] == 0
            and summary["controller_transfers"] == 2 * metadata["tile_count"]
        )
        roofline_null = all(
            value is None for value in metadata["roofline"].values()
        )
        path_checks[key] = {
            "canonical": stored_config == memory_config
            and path.read_text() == canonical_json(memory_config)
            and sha256_file(path) == output["artifact"]["sha256"]
            and metadata == expected_metadata,
            "formula": all(formulas.values()),
            "h102": metadata["fma_count"] == contract["full_fu_counts"]["fma"]
            and metadata["lower_bound_read_bytes"]
            == contract["full_load_bytes"]
            and metadata["selected_write_bytes"]
            == contract["full_store_bytes"],
            "packing": packing,
            "execution_bytes": execution_bytes,
            "dma_cycles": memory["dma_data_cycles"] == expected_dma_cycles
            and memory["dma_setup_cycles"] == 0
            and summary["end_to_end_cycles"] == expected_dma_cycles,
            "completion": completion,
            "oi": math.isfinite(metadata["selected_oi_flop_per_byte"])
            and metadata["selected_oi_flop_per_byte"] > 0
            and math.isfinite(metadata["lower_bound_oi_flop_per_byte"])
            and metadata["lower_bound_oi_flop_per_byte"] > 0,
            "swa_bound": family_not_swa_or_lower(metadata),
            "roofline_null": roofline_null,
        }
        path_results[key] = {
            "family": metadata["family"],
            "tile_count": metadata["tile_count"],
            "selected_read_bytes": metadata["selected_read_bytes"],
            "selected_write_bytes": metadata["selected_write_bytes"],
            "selected_offchip_bytes": metadata["selected_offchip_bytes"],
            "lower_bound_offchip_bytes": metadata["lower_bound_offchip_bytes"],
            "fma_count": metadata["fma_count"],
            "effective_flops": metadata["effective_flops"],
            "selected_oi_flop_per_byte": metadata[
                "selected_oi_flop_per_byte"
            ],
            "lower_bound_oi_flop_per_byte": metadata[
                "lower_bound_oi_flop_per_byte"
            ],
            "dma_data_cycles": memory["dma_data_cycles"],
            "checks": path_checks[key],
        }
        selected_ois.append(metadata["selected_oi_flop_per_byte"])
        lower_ois.append(metadata["lower_bound_oi_flop_per_byte"])
        selected_bytes.append(metadata["selected_offchip_bytes"])
        tile_counts.append(metadata["tile_count"])

    family_counts = Counter(item["family"] for item in path_results.values())
    formula_by_family = {
        family: all(
            path_checks[key]["formula"]
            for key, item in path_results.items()
            if item["family"] == family
        )
        for family in ("fft", "qkv_bsmm", "swa")
    }
    mode_counts = Counter(item["mode"] for item in run["records"])
    execution_checks = {
        "count": len(run["records"])
        == int(config["execution"]["required_executions"]),
        "mode_counts": dict(mode_counts)
        == {"debug": 96, "optimized": 96, "asan": 48, "ubsan": 48},
        "runs": all(item["pass"] for item in run["records"]),
        "replays": all(run["replay_checks"].values()),
        "builds": all(run["cross_build_checks"].values()),
        "manifest": all(run["checks"].values()),
    }
    h106_result = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h106"]["path"]).read_text()
    )
    h106_manifest = qualify(
        PROJECT_ROOT / h106_result["run_manifest"]["path"],
        h106_result["run_manifest"],
    )
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    source_text = "\n".join(
        (
            PROJECT_ROOT / config["source_layout"][name]
        ).read_text(encoding="utf-8", errors="replace")
        for name in ("compiler_core", "compiler", "runner")
    )
    target_free = (
        config["execution"]["paper_performance_targets_consumed"] is False
        and compiled["paper_performance_targets_consumed"] is False
        and run["paper_performance_targets_consumed"] is False
        and "fig25_roofline_utilization" not in source_text
        and "heatmap" not in source_text
        and all(
            all(value is None for value in item["roofline"].values())
            for item in (output["metadata"] for output in compiled["outputs"].values())
        )
    )
    all_paths = all(all(check.values()) for check in path_checks.values())
    swa_strict = all(
        item["selected_oi_flop_per_byte"]
        < item["lower_bound_oi_flop_per_byte"]
        for item in path_results.values()
        if item["family"] == "swa"
    )
    acceptance_gates = [
        len(path_results) == 48
        and dict(family_counts) == {"fft": 8, "qkv_bsmm": 24, "swa": 16},
        snapshot_reproduces
        and all(
            check["h102"] for check in path_checks.values()
        ),
        formula_by_family["fft"],
        formula_by_family["qkv_bsmm"],
        formula_by_family["swa"],
        all(check["packing"] for check in path_checks.values()),
        all(check["execution_bytes"] for check in path_checks.values()),
        all(check["dma_cycles"] for check in path_checks.values()),
        all(check["completion"] for check in path_checks.values()),
        all(execution_checks.values()),
        all(check["oi"] for check in path_checks.values()) and swa_strict,
        target_free and h106_manifest["pass"],
    ]
    integrity_checks = {
        "frozen_inputs": all(item["pass"] for item in frozen.values()),
        "snapshot_reproduces": snapshot_reproduces,
        "compile_manifest": qualify(compile_path)["pass"],
        "run_manifest": qualify(run_path)["pass"],
        "paths": all_paths,
        "execution": all(execution_checks.values()),
        "source_files": all(item["pass"] for item in source_files.values()),
        "h106_regression": h106_manifest["pass"],
        "target_free": target_free,
        "acceptance": all(acceptance_gates) and len(acceptance_gates) == 12,
    }
    integrity = all(integrity_checks.values())
    family_ranges = {}
    for family in ("fft", "qkv_bsmm", "swa"):
        members = [
            item for item in path_results.values() if item["family"] == family
        ]
        family_ranges[family] = {
            "paths": len(members),
            "selected_oi_min": min(
                item["selected_oi_flop_per_byte"] for item in members
            ),
            "selected_oi_max": max(
                item["selected_oi_flop_per_byte"] for item in members
            ),
            "lower_bound_oi_min": min(
                item["lower_bound_oi_flop_per_byte"] for item in members
            ),
            "lower_bound_oi_max": max(
                item["lower_bound_oi_flop_per_byte"] for item in members
            ),
            "offchip_bytes_min": min(
                item["selected_offchip_bytes"] for item in members
            ),
            "offchip_bytes_max": max(
                item["selected_offchip_bytes"] for item in members
            ),
        }
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
        "paper_performance_targets_consumed": False,
        "paper_reproduction_claim": "none_target_free_oi_contract_only",
        "frozen_inputs": frozen,
        "h106_run_manifest": h106_manifest,
        "compile_manifest": qualify(compile_path),
        "run_manifest": qualify(run_path),
        "path_checks": path_checks,
        "path_results": path_results,
        "formula_checks_by_family": formula_by_family,
        "execution_checks": execution_checks,
        "family_ranges": family_ranges,
        "source_files": source_files,
        "summary": {
            "paths": len(path_results),
            "family_counts": dict(family_counts),
            "executions": len(run["records"]),
            "sanitizer_executions": sum(
                item["mode"] in {"asan", "ubsan"} for item in run["records"]
            ),
            "acceptance_gates_passed": sum(acceptance_gates),
            "acceptance_gates_total": len(acceptance_gates),
            "selected_oi_min": min(selected_ois),
            "selected_oi_max": max(selected_ois),
            "lower_bound_oi_min": min(lower_ois),
            "lower_bound_oi_max": max(lower_ois),
            "offchip_bytes_min": min(selected_bytes),
            "offchip_bytes_max": max(selected_bytes),
            "tile_count_min": min(tile_counts),
            "tile_count_max": max(tile_counts),
            "roofline_utilization_available": False,
            "full_paper_rows_reproduced": 0,
            "full_paper_rows_total": 18,
        },
        "integrity_checks": integrity_checks,
    }


def family_not_swa_or_lower(metadata: dict[str, Any]) -> bool:
    return metadata["family"] != "swa" or (
        metadata["selected_oi_flop_per_byte"]
        < metadata["lower_bound_oi_flop_per_byte"]
    )


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
        existing = json.loads(output.read_text())
        keys = (
            "hypothesis_status",
            "audit_integrity",
            "path_results",
            "family_ranges",
            "summary",
            "integrity_checks",
        )
        matches = all(existing.get(key) == report.get(key) for key in keys)
        print(json.dumps({"existing_matches": matches, **report}, indent=2))
        return 0 if matches else 1
    if args.preflight_only:
        print(json.dumps(report, indent=2))
        return 0 if report["audit_integrity"] and not output.exists() else 1
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["hypothesis_status"], **report["summary"]}, indent=2))
    return 0 if report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
