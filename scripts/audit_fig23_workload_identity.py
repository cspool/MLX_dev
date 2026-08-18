#!/usr/bin/env python3
"""Audit whether Figure 23's exact transformer-block workload is identifiable."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/fig23_workload_identity_v1.yaml"


def figure23_section(paper: str) -> str:
    start = paper.index("# C. Resource Utilization and Scalability")
    end = paper.index("#### D. Sensitivity on Structured LLM Workloads", start)
    return paper[start:end]


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    h64 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h64"]["path"]).read_text()
    )
    h90 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h90"]["path"]).read_text()
    )
    h91 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h91"]["path"]).read_text()
    )
    manifest = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h64_manifest"]["path"]).read_text()
    )
    h64_config = yaml.safe_load(
        (PROJECT_ROOT / config["frozen_inputs"]["h64_config"]["path"]).read_text()
    )
    parent_checks = {
        name: parent["hypothesis_status"] == spec["required_status"]
        and parent["audit_integrity"] is spec["required_integrity"]
        for name, parent in (("h64", h64), ("h90", h90), ("h91", h91))
        for spec in [config["frozen_inputs"][name]]
    }

    paper = (PROJECT_ROOT / config["frozen_inputs"]["paper"]["path"]).read_text()
    section = figure23_section(paper)
    disclosed = config["paper_disclosed"]
    paper_checks = {
        "sequence_lengths": all(
            token in section for token in ("512", "1K", "2K", "4K", "8K")
        ),
        "hidden_dimension": "D=512" in section,
        "batch": "batched by 8" in section,
        "simd": "8- vs. 32-way SIMD" in section,
        "meshes": "4\\times4" in section and "8\\times8" in section,
        "peak_scaling": "4\\times" in section and "peak compute scaling" in section,
        "shape_contract": disclosed["sequence_lengths"]
        == [512, 1024, 2048, 4096, 8192]
        and disclosed["hidden_dimension"] == 512
        and disclosed["batch"] == 8,
    }
    identity_evidence = {
        field: {
            "status": "not_reported",
            "figure23_section_occurrences": 0,
        }
        for field in config["required_identity_fields"]
    }
    identity_checks = {
        "field_count": len(identity_evidence) == 13,
        "all_classified": all(
            item["status"] in {"reported", "not_reported"}
            for item in identity_evidence.values()
        ),
        "missing_present": any(
            item["status"] == "not_reported" for item in identity_evidence.values()
        ),
    }
    exact_workload_identified = all(
        item["status"] == "reported" for item in identity_evidence.values()
    )

    outputs = manifest["outputs"]
    baseline_rows: dict[str, Any] = {}
    proxy_checks: dict[str, bool] = {}
    for sequence_length in disclosed["sequence_lengths"]:
        key = f"{sequence_length}-baseline"
        item = outputs[key]
        metadata = item["metadata"]
        scalar_groups = sequence_length * disclosed["batch"]
        stages = int(math.log2(config["h64_proxy_contract"]["hidden_width"]))
        expected_output_lane_work = (
            scalar_groups
            * config["h64_proxy_contract"]["hidden_width"]
            * stages
        )
        document = json.loads((PROJECT_ROOT / item["primary"]["path"]).read_text())
        work = metadata["lane_normalized_work"]
        checks = {
            "operator": metadata["operator"] == "bsmm",
            "width": metadata["width"] == 512,
            "batch": metadata["batch"] == 8,
            "sequence": metadata["sequence_length"] == sequence_length,
            "fixed_memory": document["memory_backend"] == "fixed",
            "active_window": document["active_window"] == 3,
            "output_formula": work["output_lane_work"]
            == expected_output_lane_work,
            "instruction_formula": work["instruction_lane_work"]
            == 5 * expected_output_lane_work,
            "memory_formula": work["memory_lane_work"]
            == 6 * scalar_groups * 512,
            "transfer_formula": work["transfer_lane_work"]
            == 7 * scalar_groups * 512,
        }
        proxy_checks[key] = all(checks.values())
        baseline_rows[key] = {
            "checks": checks,
            "lane_normalized_work": work,
            "expected_output_lane_work": expected_output_lane_work,
        }
    hardware_shapes = {
        (
            int(item["metadata"]["simd_width"]),
            tuple(int(value) for value in item["metadata"]["mesh"]),
        )
        for item in outputs.values()
    }
    manifest_checks = {
        "outputs": len(outputs) == 20,
        "baselines": len(baseline_rows) == 5 and all(proxy_checks.values()),
        "hardware_shapes": hardware_shapes
        == {(8, (4, 4)), (32, (4, 4)), (8, (8, 8)), (32, (8, 8))},
        "target_free": manifest["paper_performance_targets_consumed"] is False,
        "config_proxy": h64_config["workload"]["operator"] == "bsmm"
        and h64_config["workload"]["memory_backend"] == "fixed"
        and h64_config["workload"]["active_window"] == 3,
    }
    compiler_source = (PROJECT_ROOT / "src/mlxsim/fig10_scaling.py").read_text()
    absent_components = config["h64_proxy_contract"]["absent_components"]
    component_checks = {
        "single_bsmm_call": 'compile_fig10_mapping("bsmm", hidden_width' in compiler_source,
        "no_complete_components": all(
            token not in compiler_source
            for token in (
                "compile_combined_attention",
                '"qkv"',
                '"output_projection"',
                '"ffn1"',
                '"ffn2"',
                '"elementwise"',
            )
        )
        and len(absent_components) == 6
        and set(config["h64_proxy_contract"]["components"])
        == {"single_bsmm_transform"},
    }
    h64_full_transformer_block = False
    methodology_checks = {
        "h90_supported_only": parent_checks["h90"],
        "h91_supported_only": parent_checks["h91"],
        "no_h91_shape_import": disclosed["hidden_dimension"] != 4096
        and disclosed["sequence_lengths"] != [128, 256, 512, 1024, 2048],
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    source_text = "\n".join(
        (PROJECT_ROOT / path).read_text(errors="replace")
        for path in config["source_layout"].values()
    )
    forbidden = (
        "fig10-fig23" + "-transfer-run070.json",
        "dsagen-mlx-fig23" + "-run052.json",
        "multiport-fig23" + "-transfer-run075.json",
    )
    target_free_checks = {
        "no_target_result": not any(item in source_text for item in forbidden),
        "no_fit": "fit" + "_affine" not in source_text,
        "no_residual": "residual" + "_correction" not in source_text,
        "config": config["acceptance"]["targets_consumed"] is False,
    }
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        all(paper_checks.values()),
        all(identity_checks.values()),
        exact_workload_identified is False,
        all(manifest_checks.values()) and all(proxy_checks.values()),
        manifest_checks["config_proxy"] and manifest_checks["hardware_shapes"],
        all(component_checks.values()) and h64_full_transformer_block is False,
        all(methodology_checks.values()),
        all(target_free_checks.values()) and all(item["pass"] for item in source_files.values()),
        config["validation_eligible"] is False,
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "paper": all(paper_checks.values()),
        "identity_evaluated": all(identity_checks.values()),
        "manifest": all(manifest_checks.values()),
        "proxy": all(proxy_checks.values()),
        "components": all(component_checks.values()),
        "methodology": all(methodology_checks.values()),
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
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "paper_checks": paper_checks,
        "figure23_section": section,
        "identity_evidence": identity_evidence,
        "identity_checks": identity_checks,
        "exact_workload_identified": exact_workload_identified,
        "h64_full_transformer_block": h64_full_transformer_block,
        "h64_manifest_checks": manifest_checks,
        "h64_proxy_checks": proxy_checks,
        "h64_baseline_rows": baseline_rows,
        "component_checks": component_checks,
        "methodology_checks": methodology_checks,
        "target_free_checks": target_free_checks,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "required_identity_fields": len(identity_evidence),
            "reported_identity_fields": sum(
                item["status"] == "reported" for item in identity_evidence.values()
            ),
            "missing_identity_fields": sum(
                item["status"] == "not_reported"
                for item in identity_evidence.values()
            ),
            "h64_configs": len(outputs),
            "h64_baseline_rows": len(baseline_rows),
            "exact_workload_identified": exact_workload_identified,
            "h64_full_transformer_block": h64_full_transformer_block,
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
            "identity_evidence",
            "exact_workload_identified",
            "h64_full_transformer_block",
            "h64_baseline_rows",
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
