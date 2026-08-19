#!/usr/bin/env python3
"""Audit H182 target-free RTX4090 traces for Figures 19/20/23."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify
from scripts.run_fig19_20_23_rtx4090_trace import case_specs

DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig19_20_23_rtx4090_trace_v1.yaml"


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    parents = {
        name: json.loads((PROJECT_ROOT / spec["path"]).read_text())
        for name, spec in config["frozen_inputs"].items()
    }
    parent_checks = {
        name: parent["hypothesis_status"] == spec["required_status"]
        and parent["audit_integrity"] is spec["required_integrity"]
        for name, parent in parents.items()
        for spec in [config["frozen_inputs"][name]]
    }
    manifest_path = PROJECT_ROOT / config["manifest_path"]
    manifest_file = qualify(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    expected = case_specs(config)
    expected_keys = {
        f"fig{item['figure']}-N{item['sequence_length']}-{item['component']}"
        for item in expected
    }
    records = manifest["cases"]
    actual_keys = {record["key"] for record in records}
    counts = {
        str(figure): sum(record["figure"] == figure for record in records)
        for figure in (19, 20, 23)
    }
    count_checks = {
        "figure19": counts["19"]
        == int(config["acceptance"]["required_figure19_cases"]),
        "figure20": counts["20"]
        == int(config["acceptance"]["required_figure20_cases"]),
        "figure23": counts["23"]
        == int(config["acceptance"]["required_figure23_cases"]),
        "total": len(records) == int(config["acceptance"]["required_total_cases"]),
        "exact_keys": actual_keys == expected_keys,
    }
    sample_checks = {
        record["key"]: len(record["timing_samples_ms"])
        == int(config["timing"]["timed_iterations"])
        + (
            int(config["timing"]["small_case_extra_iterations"])
            if record["sequence_length"]
            <= int(config["timing"]["small_case_max_sequence"])
            else 0
        )
        and all(
            math.isfinite(float(value)) and float(value) > 0
            for value in record["timing_samples_ms"]
        )
        for record in records
    }
    timing_checks = {
        record["key"]: all(
            math.isfinite(float(record["timing"][field]))
            and float(record["timing"][field]) > 0
            for field in (
                "minimum_ms",
                "p25_ms",
                "median_ms",
                "p75_ms",
                "maximum_ms",
                "mean_ms",
            )
        )
        and record["timing"]["minimum_ms"]
        <= record["timing"]["p25_ms"]
        <= record["timing"]["median_ms"]
        <= record["timing"]["p75_ms"]
        <= record["timing"]["maximum_ms"]
        for record in records
    }
    output_checks = {
        record["key"]: record["output_finite"] is True
        and math.isfinite(float(record["sampled_checksum"]))
        and int(record["output_elements"]) > 0
        for record in records
    }
    precision_checks = {
        record["key"]: (
            record["metadata"]["dtype"] == "float16"
            if record["figure"] == 20
            and record["component"].startswith(("dense_tcu", "dense_flash"))
            else record["metadata"]["dtype"] == "float32"
        )
        for record in records
    }
    gpu_checks = {
        "before_name": manifest["gpu_before"]["name"] == config["gpu"]["expected_name"],
        "before_uuid": manifest["gpu_before"]["uuid"] == config["gpu"]["expected_uuid"],
        "before_capability": manifest["gpu_before"]["compute_cap"]
        == config["gpu"]["expected_compute_capability"],
        "after_identity": manifest["gpu_after"]["name"] == config["gpu"]["expected_name"]
        and manifest["gpu_after"]["uuid"] == config["gpu"]["expected_uuid"]
        and manifest["gpu_after"]["compute_cap"]
        == config["gpu"]["expected_compute_capability"],
        "torch_identity": manifest["torch"]["device_name"] == config["gpu"]["expected_name"]
        and manifest["torch"]["compute_capability"]
        == config["gpu"]["expected_compute_capability"],
        "tf32_disabled": manifest["torch"]["allow_tf32"] is False,
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    runner_text = (PROJECT_ROOT / config["source_layout"]["runner"]).read_text()
    target_free_checks = {
        "manifest": manifest["paper_performance_targets_consumed"] is False,
        "runner": "artifacts/targets" not in runner_text
        and "target_speedup" not in runner_text
        and "paper_targets.yaml" not in runner_text,
        "config": all(
            "artifacts/targets" not in spec["path"]
            for spec in config["frozen_inputs"].values()
        )
        and config["acceptance"]["paper_targets_consumed"] is False,
    }
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        manifest_file["pass"] and manifest["experiment_id"] == config["experiment_id"],
        all(gpu_checks.values()),
        all(count_checks.values()),
        all(sample_checks.values()),
        all(timing_checks.values()),
        all(output_checks.values()),
        all(precision_checks.values()),
        all(item["pass"] for item in source_files.values()),
        all(target_free_checks.values()) and all(manifest["checks"].values()),
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": len(parent_checks) == 4,
        "manifest": manifest_file["pass"],
        "counts": len(count_checks) == 5,
        "samples": len(sample_checks) == 38,
        "timing": len(timing_checks) == 38,
        "outputs": len(output_checks) == 38,
        "precision": len(precision_checks) == 38,
        "gpu": len(gpu_checks) == 6,
        "source": all(item["pass"] for item in source_files.values()),
        "target_free": all(target_free_checks.values()),
        "acceptance_evaluated": len(acceptance_gates) == 10
        and all(isinstance(value, bool) for value in acceptance_gates),
    }
    integrity = all(integrity_checks.values())
    supported = integrity and all(acceptance_gates)
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
        "paper_reproduction_claim": "none_target_free_RTX4090_trace_only",
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "generated_input": manifest_file,
        "gpu_checks": gpu_checks,
        "count_checks": count_checks,
        "sample_checks": sample_checks,
        "timing_checks": timing_checks,
        "output_checks": output_checks,
        "precision_checks": precision_checks,
        "target_free_checks": target_free_checks,
        "cases": records,
        "source_files": source_files,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "gpu": manifest["gpu_before"]["name"],
            "gpu_uuid": manifest["gpu_before"]["uuid"],
            "figure19_cases": counts["19"],
            "figure20_cases": counts["20"],
            "figure23_cases": counts["23"],
            "total_cases": len(records),
            "total_timing_samples": sum(
                len(record["timing_samples_ms"]) for record in records
            ),
            "all_outputs_finite": all(output_checks.values()),
            "paper_targets_consumed": False,
            "acceptance_gates_passed": sum(acceptance_gates),
            "acceptance_gates_total": len(acceptance_gates),
        },
        "integrity_checks": integrity_checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    report = build_audit(config)
    output = PROJECT_ROOT / config["result_path"]
    if args.verify_existing:
        existing = json.loads(output.read_text())
        keys = (
            "hypothesis_status",
            "audit_integrity",
            "gpu_checks",
            "count_checks",
            "sample_checks",
            "timing_checks",
            "output_checks",
            "precision_checks",
            "target_free_checks",
            "cases",
            "acceptance_gates",
            "summary",
            "integrity_checks",
        )
        matches = all(
            json.dumps(existing.get(key), sort_keys=True)
            == json.dumps(report.get(key), sort_keys=True)
            for key in keys
        )
        print(json.dumps({"existing_matches": matches, **report}, indent=2))
        return 0 if matches else 1
    if args.preflight_only:
        print(json.dumps(report, indent=2))
        return 0 if report["audit_integrity"] and not output.exists() else 1
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["hypothesis_status"], **report["summary"]}, indent=2))
    return 0 if report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
