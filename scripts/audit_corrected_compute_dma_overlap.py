#!/usr/bin/env python3
"""Audit H111 corrected-cycle compute/DMA bandwidth sensitivity."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from mlxsim.corrected_compute_dma_overlap import compose_corrected_point

try:
    from scripts.audit_compute_dma_overlap import (
        balanced_bytes,
        git_commit,
        qualify,
    )
except ModuleNotFoundError:
    from audit_compute_dma_overlap import (
        balanced_bytes,
        git_commit,
        qualify,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/simulators/corrected_compute_dma_overlap_v1.yaml"
)


def nested_field_names(value: Any) -> set[str]:
    names: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            names.add(str(key))
            names.update(nested_field_names(item))
    elif isinstance(value, list):
        for item in value:
            names.update(nested_field_names(item))
    return names


def point_id(point: dict[str, Any]) -> str:
    return f'{point["key"]}@{int(point["bandwidth_bytes_per_cycle"])}'


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / item["path"], item)
        for name, item in config["frozen_inputs"].items()
    }
    h108 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h108"]["path"]).read_text()
    )
    h107 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h107"]["path"]).read_text()
    )
    h110 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h110"]["path"]).read_text()
    )
    parent_checks = {
        "h108_supported": h108["hypothesis_status"] == "supported"
        and h108["audit_integrity"] is True,
        "h108_cycle_diagnosis": h108["parent_cycle_semantics"][
            "valid_for_figure25_throughput"
        ]
        is False,
        "h107_supported": h107["hypothesis_status"] == "supported"
        and h107["audit_integrity"] is True,
        "h107_roofline_null": h107["summary"][
            "roofline_utilization_available"
        ]
        is False,
        "h110_rejected_with_integrity": h110["hypothesis_status"]
        == "rejected"
        and h110["audit_integrity"] is True,
        "h110_cycles_qualified": h110["summary"][
            "all_cycle_holdouts_pass"
        ]
        is True
        and h110["summary"]["cycle_holdouts_passed"] == 96
        and h110["summary"]["cycle_holdouts_total"] == 96,
        "h110_residence_rejected": h110["summary"][
            "all_residence_holdouts_pass"
        ]
        is False,
    }

    h108_manifest_path = PROJECT_ROOT / h108["run_manifest"]["path"]
    h108_manifest_file = qualify(h108_manifest_path, h108["run_manifest"])
    h108_manifest = json.loads(h108_manifest_path.read_text())
    h108_replay_file = qualify(
        PROJECT_ROOT / h108_manifest["replays"][0]["path"],
        h108_manifest["replays"][0],
    )
    h108_points = json.loads(
        (PROJECT_ROOT / h108_manifest["replays"][0]["path"]).read_text()
    )["points"]
    old_points = {point_id(point): point for point in h108_points}

    output_root = PROJECT_ROOT / config["output_root"]
    manifest_path = (
        output_root / "corrected-compute-dma-overlap-run-manifest.json"
    )
    manifest_file = qualify(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    replays = [
        (
            item,
            json.loads((PROJECT_ROOT / item["path"]).read_text()),
        )
        for item in manifest["replays"]
    ]
    replay_checks = {
        "count": len(replays)
        == int(config["execution"]["deterministic_replays"]),
        "files": all(
            qualify(PROJECT_ROOT / item["path"], item)["pass"]
            for item, _ in replays
        ),
        "byte_identical": len({item["sha256"] for item, _ in replays}) == 1,
        "point_count": all(
            len(payload["points"])
            == int(config["execution"]["required_points"])
            for _, payload in replays
        ),
        "payload_contract": all(
            payload["paper_performance_targets_consumed"] is False
            and payload["selected_mlx_bandwidth_bytes_per_cycle"] is None
            and payload["paper_reproduction_claim"] is None
            for _, payload in replays
        ),
    }
    points = replays[0][1]["points"]
    hardware = config["hardware"]
    forbidden = set(config["execution"]["forbidden_point_fields"])
    point_checks: dict[str, dict[str, bool]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    speedups: dict[str, float] = {}
    for point in points:
        key = point["key"]
        bandwidth = int(point["bandwidth_bytes_per_cycle"])
        identifier = point_id(point)
        estimate = h110["full_estimates"][key]
        cycles_float = float(estimate["cycles"])
        cycles = int(cycles_float)
        path = h107["path_results"][key]
        expected = compose_corrected_point(
            key=key,
            corrected_cycles=cycles,
            h110_issue_utilization=float(estimate["fma_issue_utilization"]),
            path=path,
            bandwidth=bandwidth,
            physical_pes=int(hardware["physical_pes"]),
            simd_width=int(hardware["simd_width"]),
            effective_ops_per_fma=int(hardware["effective_ops_per_fma"]),
            exact_peak_effective_ops_per_cycle=float(
                hardware["exact_peak_effective_ops_per_cycle"]
            ),
            nominal_peak_effective_ops_per_cycle=float(
                hardware["nominal_table_iv_peak_effective_ops_per_cycle"]
            ),
            nominal_peak_relative_difference_limit=float(
                hardware["nominal_peak_relative_difference_limit"]
            ),
        )
        schedule = point["schedule"]
        corrected = point["corrected_compute"]
        throughput = point["throughput_effective_ops_per_cycle"]
        utilization = point["roofline_utilization_sensitivity"]
        recomputed_issue = int(path["fma_count"]) / (
            cycles
            * int(hardware["physical_pes"])
            * int(hardware["simd_width"])
        )
        old = old_points[identifier]
        old_pipeline_cycles = int(old["schedule"]["pipeline_cycles"])
        speedups[identifier] = old_pipeline_cycles / int(
            schedule["pipeline_cycles"]
        )
        point_checks[identifier] = {
            "reproduces": point == expected,
            "input_identity": corrected["cycles"] == cycles
            and point["effective_flops"] == path["effective_flops"]
            and point["operational_intensity"]
            == path["selected_oi_flop_per_byte"]
            and point["tile_count"] == path["tile_count"],
            "cycle_integral": cycles_float == cycles and cycles > 0,
            "effective_fma_work": int(path["effective_flops"])
            == int(path["fma_count"])
            * int(hardware["effective_ops_per_fma"]),
            "issue_reconstruction": math.isclose(
                recomputed_issue,
                float(estimate["fma_issue_utilization"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            and math.isclose(
                corrected["direct_fma_issue_utilization"],
                recomputed_issue,
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            "compute_partition": sum(point["compute_cycles_by_tile"])
            == cycles
            and min(point["compute_cycles_by_tile"]) > 0,
            "dma": schedule["dma_cycles"]
            == sum(
                math.ceil(value / bandwidth)
                for value in [
                    *balanced_bytes(path, "read"),
                    *balanced_bytes(path, "write"),
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
            "peak": point["peak_contract"][
                "exact_peak_effective_ops_per_cycle"
            ]
            == int(hardware["physical_pes"])
            * int(hardware["simd_width"])
            * int(hardware["effective_ops_per_fma"])
            and point["peak_contract"][
                "nominal_peak_relative_difference"
            ]
            <= float(hardware["nominal_peak_relative_difference_limit"]),
            "roofline": point["roofline_denominator_ops_per_cycle"]
            == min(
                float(hardware["exact_peak_effective_ops_per_cycle"]),
                point["operational_intensity"] * bandwidth,
            ),
            "throughput": throughput["serial"]
            <= throughput["pipeline"]
            <= throughput["ideal"]
            and throughput["pipeline"]
            <= corrected["direct_effective_ops_per_cycle"],
            "utilization": utilization["serial"]
            <= utilization["pipeline"]
            <= utilization["ideal"]
            and all(
                math.isfinite(value) and 0 < value <= 1
                for value in utilization.values()
            ),
            "non_slowdown": schedule["pipeline_cycles"]
            <= old_pipeline_cycles,
            "strict_speedup": schedule["pipeline_cycles"]
            < old_pipeline_cycles,
            "classification": point["bandwidth_classification"]
            == (
                "historical_dpu_sensitivity"
                if bandwidth
                == int(
                    hardware["historical_dpu_anchor_bytes_per_cycle"]
                )
                else "power_of_two_sensitivity"
            ),
            "fields_excluded": forbidden.isdisjoint(
                nested_field_names(point)
            ),
            "null_claim": point["selected_mlx_bandwidth_bytes_per_cycle"]
            is None
            and point["paper_reproduction_claim"] is None,
        }
        grouped[key].append(point)

    monotonic_checks: dict[str, dict[str, bool]] = {}
    for key, values in grouped.items():
        values.sort(key=lambda item: item["bandwidth_bytes_per_cycle"])
        monotonic_checks[key] = {
            "grid": [
                item["bandwidth_bytes_per_cycle"] for item in values
            ]
            == hardware["bandwidth_sweep_bytes_per_cycle"],
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
            "cycles_invariant": len(
                {
                    item["corrected_compute"]["cycles"]
                    for item in values
                }
            )
            == 1,
            "oi_invariant": len(
                {item["operational_intensity"] for item in values}
            )
            == 1,
            "flops_invariant": len(
                {item["effective_flops"] for item in values}
            )
            == 1,
            "issue_invariant": len(
                {
                    item["corrected_compute"][
                        "direct_fma_issue_utilization"
                    ]
                    for item in values
                }
            )
            == 1,
        }

    family_counts = Counter(
        h107["path_results"][key]["family"] for key in grouped
    )
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    executable_source = "\n".join(
        (PROJECT_ROOT / config["source_layout"][name]).read_text()
        for name in ("corrected_model", "runner")
    )
    all_fields_excluded = all(
        check["fields_excluded"] for check in point_checks.values()
    ) and all(field not in executable_source for field in forbidden)
    target_free = (
        all(replay_checks.values())
        and manifest["paper_performance_targets_consumed"] is False
        and manifest["selected_mlx_bandwidth_bytes_per_cycle"] is None
        and manifest["paper_reproduction_claim"] is None
        and hardware["mlx_bandwidth_bytes_per_cycle"] is None
        and "fig25_roofline_utilization" not in executable_source.lower()
        and "heatmap" not in executable_source.lower()
    )
    all_points_reproduce = all(
        check["reproduces"] for check in point_checks.values()
    )
    all_monotonic = all(
        all(check.values()) for check in monotonic_checks.values()
    )
    counts = {
        "paths": len(grouped) == int(config["execution"]["required_paths"]),
        "families": dict(family_counts)
        == config["execution"]["family_counts"],
        "points": len(points) == int(config["execution"]["required_points"]),
        "records": len(points) * len(replays)
        == int(config["execution"]["required_records"]),
        "old_points": len(old_points)
        == int(config["execution"]["required_points"]),
        "unique": len({point_id(point) for point in points}) == len(points),
    }
    strict_family = {
        family: any(
            check["strict_speedup"]
            for identifier, check in point_checks.items()
            if h107["path_results"][identifier.rsplit("@", 1)[0]]["family"]
            == family
        )
        for family in ("fft", "qkv_bsmm", "swa")
    }
    h107_manifest = qualify(
        PROJECT_ROOT / h107["run_manifest"]["path"], h107["run_manifest"]
    )
    acceptance_gates = [
        all(frozen_item["pass"] for frozen_item in frozen.values())
        and all(parent_checks.values()),
        all(counts.values()),
        all(
            check["cycle_integral"]
            and check["effective_fma_work"]
            and check["issue_reconstruction"]
            for check in point_checks.values()
        ),
        all(
            check["compute_partition"] and check["dma"]
            for check in point_checks.values()
        ),
        all(check["events"] for check in point_checks.values()),
        all(check["bounds"] for check in point_checks.values()),
        all_monotonic,
        all(
            check["peak"] and check["roofline"]
            for check in point_checks.values()
        ),
        all(
            check["throughput"] and check["utilization"]
            for check in point_checks.values()
        ),
        all(check["non_slowdown"] for check in point_checks.values())
        and all(strict_family.values()),
        all(replay_checks.values())
        and all_points_reproduce
        and h108_manifest_file["pass"]
        and h108_replay_file["pass"],
        target_free
        and all_fields_excluded
        and all(
            check["classification"] and check["null_claim"]
            for check in point_checks.values()
        ),
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "h108_evidence": h108_manifest_file["pass"]
        and h108_replay_file["pass"],
        "h107_evidence": h107_manifest["pass"],
        "manifest": manifest_file["pass"]
        and all(manifest["checks"].values()),
        "replays": all(replay_checks.values()),
        "recomputation": all_points_reproduce,
        "counts": all(counts.values()),
        "source_files": all(item["pass"] for item in source_files.values()),
        "target_free": target_free and all_fields_excluded,
        "acceptance_evaluated": len(acceptance_gates) == 12
        and all(isinstance(value, bool) for value in acceptance_gates),
    }
    integrity = all(integrity_checks.values())
    supported = integrity and all(acceptance_gates)

    family_ranges = {}
    anchor = int(hardware["historical_dpu_anchor_bytes_per_cycle"])
    for family in ("fft", "qkv_bsmm", "swa"):
        family_points = [
            point
            for point in points
            if point["family"] == family
        ]
        anchor_points = [
            point
            for point in family_points
            if int(point["bandwidth_bytes_per_cycle"]) == anchor
        ]
        family_speedups = [
            speedups[point_id(point)] for point in family_points
        ]
        family_ranges[family] = {
            "paths": int(family_counts[family]),
            "points": len(family_points),
            "direct_issue_utilization_min": min(
                point["corrected_compute"][
                    "direct_fma_issue_utilization"
                ]
                for point in family_points
            ),
            "direct_issue_utilization_max": max(
                point["corrected_compute"][
                    "direct_fma_issue_utilization"
                ]
                for point in family_points
            ),
            "anchor_pipeline_utilization_min": min(
                point["roofline_utilization_sensitivity"]["pipeline"]
                for point in anchor_points
            ),
            "anchor_pipeline_utilization_max": max(
                point["roofline_utilization_sensitivity"]["pipeline"]
                for point in anchor_points
            ),
            "matched_h108_speedup_min": min(family_speedups),
            "matched_h108_speedup_max": max(family_speedups),
            "strictly_faster_points": sum(
                speedup > 1.0 for speedup in family_speedups
            ),
            "compute_limited_points": sum(
                point["schedule"]["compute_cycles"]
                >= point["schedule"]["dma_cycles"]
                for point in family_points
            ),
            "dma_limited_points": sum(
                point["schedule"]["compute_cycles"]
                < point["schedule"]["dma_cycles"]
                for point in family_points
            ),
        }
    pipeline_utilizations = [
        point["roofline_utilization_sensitivity"]["pipeline"]
        for point in points
    ]
    summary = {
        "paths": len(grouped),
        "points": len(points),
        "records": len(points) * len(replays),
        "acceptance_gates_passed": sum(acceptance_gates),
        "acceptance_gates_total": len(acceptance_gates),
        "pipeline_utilization_sensitivity_min": min(pipeline_utilizations),
        "pipeline_utilization_sensitivity_max": max(pipeline_utilizations),
        "matched_h108_speedup_min": min(speedups.values()),
        "matched_h108_speedup_max": max(speedups.values()),
        "corrected_points_not_slower": sum(
            speedup >= 1.0 for speedup in speedups.values()
        ),
        "corrected_points_strictly_faster": sum(
            speedup > 1.0 for speedup in speedups.values()
        ),
        "exact_peak_effective_ops_per_cycle": float(
            hardware["exact_peak_effective_ops_per_cycle"]
        ),
        "nominal_peak_relative_difference": abs(
            float(hardware["exact_peak_effective_ops_per_cycle"])
            - float(
                hardware["nominal_table_iv_peak_effective_ops_per_cycle"]
            )
        )
        / float(hardware["nominal_table_iv_peak_effective_ops_per_cycle"]),
        "selected_mlx_bandwidth_available": False,
        "paper_reproduction_available": False,
        "residence_estimates_consumed": False,
        "full_paper_rows_reproduced": 0,
        "full_paper_rows_total": 18,
    }
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "classification": config["classification"],
        "validation_eligible": config["validation_eligible"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_commit": git_commit(),
        "hypothesis_status": "supported" if supported else "rejected",
        "audit_integrity": integrity,
        "paper_performance_targets_consumed": False,
        "paper_reproduction_claim": (
            "none_target_free_corrected_bandwidth_sensitivity_only"
        ),
        "selected_mlx_bandwidth_bytes_per_cycle": None,
        "residence_estimates_consumed": False,
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "h108_run_manifest": h108_manifest_file,
        "h108_replay": h108_replay_file,
        "h107_run_manifest": h107_manifest,
        "run_manifest": manifest_file,
        "replay_checks": replay_checks,
        "counts": counts,
        "point_checks": point_checks,
        "monotonic_checks": monotonic_checks,
        "strict_family_speedup": strict_family,
        "matched_h108_speedups": speedups,
        "family_ranges": family_ranges,
        "acceptance_gates": acceptance_gates,
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
        existing = json.loads(output.read_text())
        keys = (
            "hypothesis_status",
            "audit_integrity",
            "parent_checks",
            "family_ranges",
            "acceptance_gates",
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
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"status": report["hypothesis_status"], **report["summary"]},
            indent=2,
        )
    )
    return 0 if report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
