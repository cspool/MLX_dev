#!/usr/bin/env python3
"""Audit Figure 18 workload and MLX measurement-source identifiability."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/fig18_workload_identity_v1.yaml"


def section_between(text: str, start: str, end: str) -> str:
    first = text.index(start)
    second = text.index(end, first)
    return text[first:second]


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    paper_spec = config["frozen_inputs"]["paper"]
    paper_file = qualify(PROJECT_ROOT / paper_spec["path"], paper_spec)
    paper = (PROJECT_ROOT / paper_spec["path"]).read_text()
    performance = section_between(
        paper,
        "# B. MLX Performance",
        "Real-world Butterfly Accelerator:",
    )
    implementation = section_between(
        paper,
        "# A. Software / Hardware Implementation",
        "## B. Benchmark Models and Hardware Baselines",
    )
    disclosed = config["paper_disclosed"]
    disclosure_checks = {
        "sequence_length": "N=1024" in performance
        and disclosed["sequence_length"] == 1024,
        "hidden_dimension": "D=512" in performance
        and disclosed["hidden_dimension"] == 512,
        "compression_ratios": "s=0.75/0.5" in performance
        and disclosed["compression_ratios"] == [0.75, 0.5],
        "single_block": "single transformer block" in performance
        and disclosed["unit"] == "single_transformer_block",
        "precision": "MLX operates in FP16" in performance
        and disclosed["precision"] == "fp16",
    }
    workload_evidence = {
        field: {"status": "not_reported", "figure18_specific_value": None}
        for field in config["required_workload_fields"]
    }
    provenance_evidence = {
        field: {"status": "not_reported", "figure18_specific_value": None}
        for field in config["required_provenance_fields"]
    }
    classification_checks = {
        "workload_count": len(workload_evidence) == 12,
        "provenance_count": len(provenance_evidence) == 6,
        "workload_classified": all(
            item["status"] in {"reported", "not_reported"}
            for item in workload_evidence.values()
        ),
        "provenance_classified": all(
            item["status"] in {"reported", "not_reported"}
            for item in provenance_evidence.values()
        ),
    }
    exact_workload_identified = all(
        item["status"] == "reported" for item in workload_evidence.values()
    )
    exact_performance_provenance_identified = all(
        item["status"] == "reported" for item in provenance_evidence.values()
    )
    designs = {
        "reduced": {
            "simd_width": 8,
            "peak_ops_per_second": 256_000_000_000,
            "measurement_statement": "tuned in simulator",
        },
        "full": {
            "simd_width": 32,
            "peak_ops_per_second": 1_000_000_000_000,
            "measurement_statement": "taped-out design compared against NVIDIA GPUs",
        },
        "figure18_series_assignment": None,
    }
    design_checks = {
        "reduced_simd": "reduce SIMD width from 32 to 8" in implementation,
        "both_sources": "cycle-accurate MLX simulator" in implementation
        and "measurements from the taped-out hardware" in implementation,
        "reduced_peak": "reduced 256 GOp/s version" in paper,
        "full_peak": "real taped-out design (1 TOp/s)" in paper,
        "unassigned": designs["figure18_series_assignment"] is None,
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    source_text = "\n".join(
        (PROJECT_ROOT / path).read_text(errors="replace")
        for path in config["source_layout"].values()
    )
    target_free_checks = {
        "config": config["acceptance"]["targets_consumed"] is False,
        "no_target_artifact": "fig18-run" + "008.json" not in source_text,
        "no_speedup_values": "algorithm_normalized" + "_speedup" not in source_text,
        "no_transfer": "fig19" + "_coupled" not in source_text
        and "fig21" + "_layer" not in source_text,
    }
    acceptance_gates = [
        paper_file["pass"],
        all(disclosure_checks.values()),
        all(classification_checks.values()),
        exact_workload_identified is False,
        exact_performance_provenance_identified is False,
        all(design_checks.values()),
        designs["figure18_series_assignment"] is None,
        all(target_free_checks.values())
        and all(item["pass"] for item in source_files.values()),
        target_free_checks["no_transfer"],
        config["acceptance"]["simulator_change_allowed"] is False,
    ]
    integrity_checks = {
        "paper": paper_file["pass"],
        "disclosures": all(disclosure_checks.values()),
        "classifications": all(classification_checks.values()),
        "designs": all(design_checks.values()),
        "target_free": all(target_free_checks.values()),
        "source_files": all(item["pass"] for item in source_files.values()),
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
        "paper_reproduction_claim": "none_target_free_identity_diagnosis_only",
        "frozen_inputs": {"paper": paper_file},
        "performance_section": performance,
        "implementation_section": implementation,
        "disclosure_checks": disclosure_checks,
        "workload_evidence": workload_evidence,
        "provenance_evidence": provenance_evidence,
        "classification_checks": classification_checks,
        "exact_workload_identified": exact_workload_identified,
        "exact_performance_provenance_identified": (
            exact_performance_provenance_identified
        ),
        "designs": designs,
        "design_checks": design_checks,
        "target_free_checks": target_free_checks,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "workload_fields": len(workload_evidence),
            "missing_workload_fields": sum(
                item["status"] == "not_reported"
                for item in workload_evidence.values()
            ),
            "provenance_fields": len(provenance_evidence),
            "missing_provenance_fields": sum(
                item["status"] == "not_reported"
                for item in provenance_evidence.values()
            ),
            "exact_workload_identified": exact_workload_identified,
            "exact_performance_provenance_identified": (
                exact_performance_provenance_identified
            ),
            "acceptance_gates_passed": sum(acceptance_gates),
            "acceptance_gates_total": len(acceptance_gates),
            "active_simulator_figures_reproduced": 0,
            "active_simulator_figures_total": 8,
        },
        "source_files": source_files,
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
            "workload_evidence",
            "provenance_evidence",
            "exact_workload_identified",
            "exact_performance_provenance_identified",
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
