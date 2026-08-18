#!/usr/bin/env python3
"""Audit target-free dense-Xavier component coverage for Figure 21."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/fig21_xavier_coverage_v1.yaml"


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    evidence = {
        name: json.loads((PROJECT_ROOT / spec["path"]).read_text())
        for name, spec in config["frozen_inputs"].items()
        if name not in {"paper", "h56_bsmm_ptx"}
    }
    parent_checks = {
        name: parent["hypothesis_status"] == spec["required_status"]
        and parent["audit_integrity"] is spec["required_integrity"]
        for name, parent in evidence.items()
        for spec in [config["frozen_inputs"][name]]
    }
    paper = (PROJECT_ROOT / config["frozen_inputs"]["paper"]["path"]).read_text()
    paper_checks = {
        "figure21": "Fig. 21 presents an end-to-end comparison" in paper,
        "dense_xavier": "a dense model on Xavier" in paper,
        "tensor_cores": "dense kernels often use Tensor Cores" in paper,
        "all_operators": "All inference operators" in paper,
        "rmsnorm": "RMSNorm" in paper,
        "positional_embedding": "positional embeddings" in paper,
    }
    h95 = evidence["h95_mlx"]
    h96 = evidence["h96_closure"]
    h56 = evidence["h56_xavier"]
    h77 = evidence["h77_sparse_projection"]
    h135 = evidence["h135_structured_attention"]
    sequences = config["figure21_contract"]["sequence_lengths"]
    mlx_checks = {
        "rows": len(h95["rows"]) == int(config["acceptance"]["expected_mlx_rows"]),
        "sequences": [row["sequence_length"] for row in h95["rows"]] == sequences,
        "positive": all(
            row["mlx_total_cycles"] > 0 and row["mlx_latency_seconds"] > 0 for row in h95["rows"]
        ),
        "xavier_null": all(
            row["xavier_total_cycles"] is None and row["speedup_over_xavier"] is None
            for row in h95["rows"]
        ),
    }
    speedup_rows = [row for row in h96["rows"] if row["series"] == "speedup_over_xavier"]
    closure_checks = {
        "five_missing": len(speedup_rows) == 5
        and all(row["status"] == "execution_incomplete" for row in speedup_rows),
        "reason": all(
            row["reason"] == "matched_dense_xavier_tensor_cycles_unavailable"
            for row in speedup_rows
        ),
    }
    derived_config_record = h56["config_derivation"]["derived"]
    derived_config_path = PROJECT_ROOT / derived_config_record["path"]
    derived_config_text = derived_config_path.read_text()
    derived_config_sha = hashlib.sha256(derived_config_path.read_bytes()).hexdigest()
    ptx = (PROJECT_ROOT / config["frozen_inputs"]["h56_bsmm_ptx"]["path"]).read_text()
    tensor_config_checks = {
        "parent_limitation": "dense Tensor Core kernels are not yet represented"
        in h56["proxy_limitations"],
        "derived_qualified": derived_config_record["pass"] is True
        and derived_config_sha == derived_config_record["sha256"],
        "tensor_available": "-gpgpu_tensor_core_avail 1" in derived_config_text,
        "four_units": "-gpgpu_num_tensor_core_units 4" in derived_config_text,
        "ptx_no_wmma": "wmma" not in ptx.lower(),
        "ptx_no_mma_sync": "mma.sync" not in ptx.lower(),
    }
    sparse_projection_checks = {
        "two_shapes": h77["coverage"]["sequence_lengths"] == [256, 8192],
        "sparse_families": h77["coverage"]["kernels"] == ["qkv", "ffn1", "ffn2"],
        "attention_excluded": h77["coverage"]["excluded"]["attention"]
        == "requires_fft_and_compressed_attention_anchors",
        "cuda_fma_model": "slope_cycles_per_fma" in h77["xavier_model"]
        and all("sparse_cuda_speedup" in estimate for estimate in h77["estimates"].values()),
        "not_dense_tensor": "tensor" not in json.dumps(h77["coverage"]).lower(),
    }
    structured_attention_checks = {
        "two_shapes": set(h135["compositions"]) == {"N256", "N8192"},
        "structured_components": all(
            set(item["xavier_components"]) == {"fftcmp", "qk", "softmax", "sv"}
            for item in h135["compositions"].values()
        ),
        "transparent_proxy": all(
            item["xavier_mapping_claim"] == "transparent_proxy_not_author_cuda"
            for item in h135["compositions"].values()
        ),
        "not_dense_full_attention": all(
            "fftcmp" in item["xavier_components"] for item in h135["compositions"].values()
        ),
    }
    existing_elementwise = any(
        token in json.dumps(parent).lower()
        for parent in (h56, h77, h135)
        for token in ("rmsnorm_kernel", "rope_kernel", "residual_kernel", "activation_kernel")
    )
    family_rows = []
    component_rows = []
    for sequence in sequences:
        for family, components in config["figure21_contract"]["required_xavier_families"].items():
            family_rows.append(
                {
                    "sequence_length": sequence,
                    "family": family,
                    "components": components,
                    "qualified": False,
                    "reason": "matched_dense_xavier_execution_missing",
                }
            )
            component_rows.extend(
                {
                    "sequence_length": sequence,
                    "family": family,
                    "component": component,
                    "qualified": False,
                    "required_execution": (
                        "wmma_tensorcore"
                        if family == "dense_tensor_projection"
                        or (family == "dense_attention" and component in {"qk", "sv"})
                        else "cuda_core"
                    ),
                }
                for component in components
            )
    coverage_checks = {
        "family_rows": len(family_rows)
        == int(config["acceptance"]["expected_required_family_rows"]),
        "component_rows": len(component_rows)
        == int(config["acceptance"]["expected_required_component_rows"]),
        "qualified_zero": sum(row["qualified"] for row in family_rows)
        == int(config["acceptance"]["expected_qualified_xavier_rows"]),
        "all_shapes": {row["sequence_length"] for row in family_rows} == set(sequences),
        "all_families": {row["family"] for row in family_rows}
        == set(config["figure21_contract"]["required_xavier_families"]),
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path) for name, path in config["source_layout"].items()
    }
    source_text = "\n".join(
        (PROJECT_ROOT / path).read_text(errors="replace")
        for path in config["source_layout"].values()
    )
    forbidden = (
        "fig21-target" + "s-run094.json",
        "speedup_over_xavier" + '": 4',
        "sparse" + "_as_dense_substitute",
    )
    target_free_check = (
        not any(token in source_text for token in forbidden)
        and config["acceptance"]["targets_consumed"] is False
    )
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        all(paper_checks.values()),
        all(mlx_checks.values()),
        all(closure_checks.values()),
        all(tensor_config_checks.values()),
        all(sparse_projection_checks.values()),
        all(structured_attention_checks.values()),
        existing_elementwise is False,
        all(coverage_checks.values()),
        target_free_check
        and all(item["pass"] for item in source_files.values())
        and sum(row["qualified"] for row in family_rows) == 0,
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "paper": all(paper_checks.values()),
        "mlx": all(mlx_checks.values()),
        "closure": all(closure_checks.values()),
        "tensor_config": all(tensor_config_checks.values()),
        "sparse_projection": all(sparse_projection_checks.values()),
        "structured_attention": all(structured_attention_checks.values()),
        "coverage": all(coverage_checks.values()),
        "source": target_free_check and all(item["pass"] for item in source_files.values()),
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
        "paper_reproduction_claim": "none_target_free_dense_xavier_gap_only",
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "paper_checks": paper_checks,
        "mlx_checks": mlx_checks,
        "closure_checks": closure_checks,
        "tensor_config_checks": tensor_config_checks,
        "sparse_projection_checks": sparse_projection_checks,
        "structured_attention_checks": structured_attention_checks,
        "existing_elementwise_execution": existing_elementwise,
        "family_rows": family_rows,
        "component_rows": component_rows,
        "coverage_checks": coverage_checks,
        "source_files": source_files,
        "target_free_check": target_free_check,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "mlx_complete_rows": len(h95["rows"]),
            "missing_speedup_rows": len(speedup_rows),
            "required_xavier_family_rows": len(family_rows),
            "required_xavier_component_rows": len(component_rows),
            "qualified_xavier_family_rows": sum(row["qualified"] for row in family_rows),
            "h56_tensor_units_enabled": tensor_config_checks["tensor_available"]
            and tensor_config_checks["four_units"],
            "h56_executed_tensor_instructions": False,
            "figure21_dense_xavier_complete": False,
            "active_simulator_figures_reproduced": 3,
            "active_simulator_figures_total": 8,
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
            "family_rows",
            "component_rows",
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
