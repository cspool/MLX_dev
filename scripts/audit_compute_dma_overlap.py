#!/usr/bin/env python3
"""Audit H108 compute/DMA overlap and bandwidth sensitivity."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from mlxsim.compute_dma_overlap import compose_point

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/compute_dma_overlap_v1.yaml"


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


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / item["path"], item)
        for name, item in config["frozen_inputs"].items()
    }
    h102 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h102"]["path"]).read_text()
    )
    h107 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h107"]["path"]).read_text()
    )
    output_root = PROJECT_ROOT / config["output_root"]
    manifest_path = output_root / "compute-dma-overlap-run-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    replays = [
        (
            item,
            json.loads((PROJECT_ROOT / item["path"]).read_text()),
        )
        for item in manifest["replays"]
    ]
    replay_checks = {
        "count": len(replays) == int(config["execution"]["deterministic_replays"]),
        "hashes": all(
            sha256_file(PROJECT_ROOT / item["path"]) == item["sha256"]
            for item, _ in replays
        ),
        "byte_identical": len({item["sha256"] for item, _ in replays}) == 1,
        "point_count": all(
            len(payload["points"])
            == int(config["execution"]["required_paths"])
            * int(config["execution"]["required_bandwidth_points"])
            for _, payload in replays
        ),
    }
    points = replays[0][1]["points"]
    point_checks = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for point in points:
        key = point["key"]
        bandwidth = int(point["bandwidth_bytes_per_cycle"])
        h102_cycles = h102["full_estimates"][key]["cycles"]
        expected = compose_point(
            key=key,
            h102_cycles=int(h102_cycles),
            path=h107["path_results"][key],
            bandwidth=bandwidth,
            peak_effective_ops_per_cycle=float(
                config["hardware"]["peak_effective_ops_per_cycle"]
            ),
        )
        schedule = point["schedule"]
        throughput = point["throughput_effective_ops_per_cycle"]
        utilization = point["roofline_utilization_sensitivity"]
        point_id = f"{key}@{bandwidth}"
        point_checks[point_id] = {
            "reproduces": point == expected,
            "inputs": schedule["compute_cycles"] == int(h102_cycles)
            and point["effective_flops"]
            == h107["path_results"][key]["effective_flops"]
            and point["operational_intensity"]
            == h107["path_results"][key]["selected_oi_flop_per_byte"]
            and point["tile_count"] == h107["path_results"][key]["tile_count"],
            "compute_partition": sum(point["compute_cycles_by_tile"])
            == int(h102_cycles)
            and min(point["compute_cycles_by_tile"]) > 0,
            "dma": schedule["dma_cycles"]
            == sum(
                math.ceil(value / bandwidth)
                for value in [
                    *balanced_bytes(h107["path_results"][key], "read"),
                    *balanced_bytes(h107["path_results"][key], "write"),
                ]
            ),
            "events": all(schedule["checks"].values())
            and schedule["fill_count"]
            == schedule["compute_count"]
            == schedule["drain_count"]
            == point["tile_count"],
            "bounds": schedule["ideal_cycles"]
            <= schedule["pipeline_cycles"]
            <= schedule["serial_cycles"],
            "roofline": point["roofline_denominator_ops_per_cycle"]
            == min(
                float(config["hardware"]["peak_effective_ops_per_cycle"]),
                point["operational_intensity"] * bandwidth,
            ),
            "throughput": throughput["serial"]
            <= throughput["pipeline"]
            <= throughput["ideal"],
            "utilization": utilization["serial"]
            <= utilization["pipeline"]
            <= utilization["ideal"]
            and all(
                math.isfinite(value) and 0 < value <= 1
                for value in utilization.values()
            ),
            "null_reproduction": point[
                "selected_mlx_bandwidth_bytes_per_cycle"
            ]
            is None
            and point["figure25_reproduction"] is None,
            "classification": point["bandwidth_classification"]
            == (
                "historical_dpu_sensitivity"
                if bandwidth
                == int(config["hardware"]["historical_dpu_anchor_bytes_per_cycle"])
                else "power_of_two_sensitivity"
            ),
        }
        grouped[key].append(point)
    monotonic_checks = {}
    for key, values in grouped.items():
        values.sort(key=lambda item: item["bandwidth_bytes_per_cycle"])
        monotonic_checks[key] = {
            "bandwidth_grid": [
                item["bandwidth_bytes_per_cycle"] for item in values
            ]
            == config["hardware"]["bandwidth_sweep_bytes_per_cycle"],
            "dma_non_increasing": all(
                values[index]["schedule"]["dma_cycles"]
                >= values[index + 1]["schedule"]["dma_cycles"]
                for index in range(len(values) - 1)
            ),
            "pipeline_non_increasing": all(
                values[index]["schedule"]["pipeline_cycles"]
                >= values[index + 1]["schedule"]["pipeline_cycles"]
                for index in range(len(values) - 1)
            ),
            "oi_invariant": len(
                {item["operational_intensity"] for item in values}
            )
            == 1,
            "flops_invariant": len({item["effective_flops"] for item in values})
            == 1,
        }
    family_counts = Counter(
        h107["path_results"][key]["family"] for key in grouped
    )
    h107_manifest = qualify(
        PROJECT_ROOT / h107["run_manifest"]["path"], h107["run_manifest"]
    )
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    source_text = "\n".join(
        (PROJECT_ROOT / config["source_layout"][name]).read_text()
        for name in ("model", "runner")
    )
    target_free = (
        all(
            payload["paper_performance_targets_consumed"] is False
            and payload["selected_mlx_bandwidth_bytes_per_cycle"] is None
            and payload["figure25_reproductions"] is None
            for _, payload in replays
        )
        and manifest["paper_performance_targets_consumed"] is False
        and config["hardware"]["mlx_bandwidth_bytes_per_cycle"] is None
        and "fig25_roofline_utilization" not in source_text
        and "heatmap" not in source_text
    )
    all_points = all(all(check.values()) for check in point_checks.values())
    all_monotonic = all(
        all(check.values()) for check in monotonic_checks.values()
    )
    acceptance_gates = [
        len(grouped) == 48
        and dict(family_counts) == {"fft": 8, "qkv_bsmm": 24, "swa": 16},
        all(check["compute_partition"] for check in point_checks.values()),
        all(check["dma"] for check in point_checks.values()),
        all(check["events"] for check in point_checks.values()),
        all(check["bounds"] for check in point_checks.values()),
        all_monotonic,
        all(
            check["inputs"] for check in point_checks.values()
        )
        and all(check["oi_invariant"] for check in monotonic_checks.values())
        and all(check["flops_invariant"] for check in monotonic_checks.values()),
        all(check["roofline"] for check in point_checks.values()),
        all(
            check["throughput"] and check["utilization"]
            for check in point_checks.values()
        ),
        all(check["classification"] for check in point_checks.values()),
        all(replay_checks.values()) and all_points,
        target_free and h107_manifest["pass"],
    ]
    integrity_checks = {
        "frozen_inputs": all(item["pass"] for item in frozen.values()),
        "manifest": qualify(manifest_path)["pass"]
        and all(manifest["checks"].values()),
        "replays": all(replay_checks.values()),
        "points": all_points,
        "monotonic": all_monotonic,
        "h107_regression": h107_manifest["pass"],
        "source_files": all(item["pass"] for item in source_files.values()),
        "target_free": target_free,
        "acceptance": all(acceptance_gates) and len(acceptance_gates) == 12,
    }
    integrity = all(integrity_checks.values())
    anchor = int(config["hardware"]["historical_dpu_anchor_bytes_per_cycle"])
    anchor_points = [
        point for point in points if point["bandwidth_bytes_per_cycle"] == anchor
    ]
    family_anchor_ranges = {}
    for family in ("fft", "qkv_bsmm", "swa"):
        members = [point for point in anchor_points if point["family"] == family]
        family_anchor_ranges[family] = {
            "paths": len(members),
            "pipeline_utilization_min": min(
                point["roofline_utilization_sensitivity"]["pipeline"]
                for point in members
            ),
            "pipeline_utilization_max": max(
                point["roofline_utilization_sensitivity"]["pipeline"]
                for point in members
            ),
            "pipeline_cycles_min": min(
                point["schedule"]["pipeline_cycles"] for point in members
            ),
            "pipeline_cycles_max": max(
                point["schedule"]["pipeline_cycles"] for point in members
            ),
        }
    pipeline_utilizations = [
        point["roofline_utilization_sensitivity"]["pipeline"] for point in points
    ]
    overlap_fractions = [
        point["schedule"]["overlap_cycles"]
        / min(
            point["schedule"]["compute_cycles"],
            point["schedule"]["dma_cycles"],
        )
        for point in points
    ]
    overlay_header_path = (
        PROJECT_ROOT
        / "third_party/dsa-framework/dsa-gem5/src/cpu/minor/ssim/mlx_overlay.hh"
    )
    overlay_source_path = (
        PROJECT_ROOT
        / "third_party/dsa-framework/dsa-gem5/src/cpu/minor/ssim/mlx_overlay.cc"
    )
    fu_source_path = PROJECT_ROOT / "src/mlxsim/fig21_timed_paths.py"
    overlay_header = overlay_header_path.read_text()
    overlay_source = overlay_source_path.read_text()
    fu_source = fu_source_path.read_text()
    anchor_qkv = [
        point
        for point in anchor_points
        if point["family"] == "qkv_bsmm"
    ]
    parent_cycle_semantics = {
        "fma_declared_latency": 4,
        "fma_declared_initiation_interval": 1,
        "single_inflight_state_per_block": "bool inflight{false}" in overlay_header,
        "candidate_rejects_inflight_block": (
            "state.complete || state.inflight" in overlay_source
        ),
        "next_iteration_waits_for_completion": (
            "state.inflight = false" in overlay_source
            and "state.ready_cycle = cycle_ + timing.latency" in overlay_source
        ),
        "fma_configuration_verified": (
            '"fma": {"class": "fma", "latency": 4, "initiation_interval": 1}'
            in fu_source
        ),
        "qkv_64Bpc_pipeline_utilization_min": min(
            point["roofline_utilization_sensitivity"]["pipeline"]
            for point in anchor_qkv
        ),
        "qkv_64Bpc_pipeline_utilization_max": max(
            point["roofline_utilization_sensitivity"]["pipeline"]
            for point in anchor_qkv
        ),
        "diagnosis": (
            "trip iterations cannot overlap inside one tagged block, so a "
            "latency-4 II-1 FMA behaves as II-4 for H102 long-trip blocks"
        ),
        "valid_for_figure25_throughput": False,
        "h108_role": "diagnostic_bandwidth_envelope_pending_iteration_pipeline_fix",
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
        "paper_reproduction_claim": "none_bandwidth_sensitivity_only",
        "selected_mlx_bandwidth_bytes_per_cycle": None,
        "figure25_reproductions": None,
        "frozen_inputs": frozen,
        "h107_run_manifest": h107_manifest,
        "run_manifest": qualify(manifest_path),
        "replay_checks": replay_checks,
        "point_checks": point_checks,
        "monotonic_checks": monotonic_checks,
        "family_anchor_ranges": family_anchor_ranges,
        "parent_cycle_semantics": parent_cycle_semantics,
        "diagnostic_source_files": {
            "overlay_header": qualify(overlay_header_path),
            "overlay_source": qualify(overlay_source_path),
            "functional_units": qualify(fu_source_path),
        },
        "source_files": source_files,
        "summary": {
            "paths": len(grouped),
            "bandwidth_points": len(
                config["hardware"]["bandwidth_sweep_bytes_per_cycle"]
            ),
            "points": len(points),
            "records": len(points) * len(replays),
            "acceptance_gates_passed": sum(acceptance_gates),
            "acceptance_gates_total": len(acceptance_gates),
            "pipeline_utilization_sensitivity_min": min(pipeline_utilizations),
            "pipeline_utilization_sensitivity_max": max(pipeline_utilizations),
            "overlap_fraction_min": min(overlap_fractions),
            "overlap_fraction_max": max(overlap_fractions),
            "selected_mlx_bandwidth_available": False,
            "figure25_reproduction_available": False,
            "parent_cycle_semantics_valid_for_roofline": False,
            "full_paper_rows_reproduced": 0,
            "full_paper_rows_total": 18,
        },
        "integrity_checks": integrity_checks,
    }


def balanced_bytes(path: dict[str, Any], kind: str) -> list[int]:
    total = int(
        path["selected_read_bytes"]
        if kind == "read"
        else path["selected_write_bytes"]
    )
    count = int(path["tile_count"])
    units, remainder = divmod(total // 32, count)
    return [(units + (index < remainder)) * 32 for index in range(count)]


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
            "family_anchor_ranges",
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
