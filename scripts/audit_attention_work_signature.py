#!/usr/bin/env python3
"""Build and audit H79's target-free compressed-attention work signature."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from mlxsim.attention_signature import attention_work_signature

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/attention_work_signature_v1.yaml"


def qualify(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    exists = path.is_file()
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if exists else None
    checks = {"is_file": exists, "sha256": digest == expected["sha256"]}
    return {
        "path": str(path.relative_to(PROJECT_ROOT)) if exists else str(path),
        "bytes": path.stat().st_size if exists else None,
        "sha256": digest,
        "checks": checks,
        "pass": all(checks.values()),
    }


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={PROJECT_ROOT}", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _parent_check(report: dict[str, Any], spec: dict[str, Any]) -> bool:
    return (
        "required_status" not in spec
        or report.get("hypothesis_status") == spec["required_status"]
    ) and (
        "required_integrity" not in spec
        or report.get("audit_integrity") is spec["required_integrity"]
    )


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    files = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    json_parents = {}
    parent_checks = {}
    for name in ("logical_work", "h57_result", "fu_anchor", "pe_contract"):
        spec = config["frozen_inputs"][name]
        report = json.loads(
            (PROJECT_ROOT / spec["path"]).read_text(encoding="utf-8")
        )
        json_parents[name] = report
        parent_checks[name] = _parent_check(report, spec)

    shape = config["shape"]
    signatures = {}
    reconciliation = {}
    for n in shape["sequence_lengths"]:
        signature = attention_work_signature(
            sequence_length=int(n),
            hidden_dimension=int(shape["hidden_dimension"]),
            batch=int(shape["batch"]),
            projections=int(shape["fft_projections"]),
            compression_ratio=float(shape["compression_ratio"]),
            fft_template={
                key: int(value) for key, value in config["fft_template"].items()
            },
        )
        logical = json_parents["logical_work"]["logical_profiles"][
            f"attention-N{n}"
        ]
        fft_parent, attention_parent = logical["components"]
        fft = signature["fft_compression"]
        attention = signature["compressed_attention"]
        checks = {
            "fft_analytical_operations": fft["analytical_operations"]
            == fft_parent["operations"],
            "fft_analytical_stages": fft[
                "analytical_stage_count_excluding_shuffle"
            ]
            == fft_parent["stages"],
            "fft_shuffle_stage_explicit": fft["tagged_stage_count"]
            == fft_parent["stages"] + 1,
            "attention_analytical_operations": attention[
                "analytical_operations_excluding_fdiv"
            ]
            == attention_parent["operations"],
            "attention_stages": attention["tagged_stage_count"]
            == attention_parent["stages"],
            "fdiv_matches_output": attention["fu_instruction_instances"]["fdiv"]
            == attention_parent["output_elements"],
            "combined_analytical_operations": signature[
                "analytical_operations_excluding_fdiv"
            ]
            == logical["operations"],
        }
        signature["logical_parent"] = {
            "operations": logical["operations"],
            "offchip_bytes": logical["offchip_bytes"],
            "output_elements": logical["output_elements"],
            "component_count": logical["component_count"],
        }
        signatures[f"N{n}"] = signature
        reconciliation[f"N{n}"] = checks

    h57_manifest = json.loads(
        (
            PROJECT_ROOT
            / config["frozen_inputs"]["h57_manifest"]["path"]
        ).read_text(encoding="utf-8")
    )
    mlx_outputs = {item["name"]: item for item in h57_manifest["mlx_outputs"]}
    h57_gaps = {}
    for n, case in ((256, "short"), (8192, "long")):
        proxy = mlx_outputs[f"fft--{case}"]
        signature = signatures[f"N{n}"]
        proxy_ops = {
            "alu_add" if operation == "add" else operation
            for operation in proxy["metadata"]["operation_counts"]
        }
        attention_ops = set(
            signature["compressed_attention"]["fu_instruction_instances"]
        )
        h57_gaps[f"N{n}"] = {
            "proxy_key": f"fft--{case}",
            "proxy_stage_count": proxy["metadata"]["stage_count"],
            "required_fft_stage_count": signature["fft_compression"][
                "tagged_stage_count"
            ],
            "proxy_operation_classes": sorted(proxy_ops),
            "missing_attention_operation_classes": sorted(attention_ops - proxy_ops),
            "single_proxy_for_two_components": json_parents["logical_work"][
                "comparisons"
            ][f"attention-N{n}"]["proxy_key"]
            == f"fft--{case}",
            "stage_count_mismatch": proxy["metadata"]["stage_count"]
            != signature["fft_compression"]["tagged_stage_count"],
            "attention_fu_mix_missing": {"fmax", "fexp", "fdiv"}.issubset(
                attention_ops - proxy_ops
            ),
        }

    anchors = json_parents["fu_anchor"]["measurements"]
    anchor_classes = {
        "fft": sorted(
            anchors["fft_cmp--Llama2_512"]["summary"][
                "productive_pe_cycles_by_fu_class"
            ]
        ),
        "swa": sorted(
            anchors["swa_w128_q32--Llama2_512"]["summary"][
                "productive_pe_cycles_by_fu_class"
            ]
        ),
    }
    paper_text = (
        PROJECT_ROOT / config["frozen_inputs"]["paper"]["path"]
    ).read_text(encoding="utf-8")
    template_text = (
        PROJECT_ROOT / config["frozen_inputs"]["execution_template"]["path"]
    ).read_text(encoding="utf-8")
    source_checks = {
        "paper_spatial_pe": "Spatial PE: Enabling Layer-Folded Execution"
        in paper_text,
        "paper_tagged_block": "tagged block" in paper_text,
        "paper_fdiv": "FDIV" in paper_text,
        "template_fft_mix": '("fma",) * 4 + ("add",) * 6' in template_text,
        "template_attention_mix": '("fexp", "add")' in template_text
        and '("fdiv",)' in template_text,
    }
    h57_gap_checks = {
        key: all(
            (
                item["single_proxy_for_two_components"],
                item["stage_count_mismatch"],
                item["attention_fu_mix_missing"],
                item["proxy_stage_count"]
                == int(config["expected_h57_fft_stage_count"]),
            )
        )
        for key, item in h57_gaps.items()
    }
    integrity_checks = {
        "frozen_files": all(item["pass"] for item in files.values()),
        "parents": all(parent_checks.values()),
        "reconciliation": all(
            all(checks.values()) for checks in reconciliation.values()
        ),
        "h57_gap": all(h57_gap_checks.values()),
        "source_contract": all(source_checks.values()),
        "fu_anchors": anchor_classes["fft"] == ["alu", "fma", "shuffle"]
        and anchor_classes["swa"]
        == ["alu", "fma", "reduce", "transcendental"],
        "no_targets": True,
    }
    integrity = all(integrity_checks.values())
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
        "frozen_inputs": files,
        "parent_checks": parent_checks,
        "signatures": signatures,
        "reconciliation": reconciliation,
        "h57_gaps": h57_gaps,
        "h57_gap_checks": h57_gap_checks,
        "available_physical_anchor_classes": anchor_classes,
        "source_checks": source_checks,
        "integrity_checks": integrity_checks,
        "paper_performance_targets_consumed": False,
        "conclusion": (
            "H57 single FFT proxy cannot represent matched FFT compression plus "
            "compressed attention"
        ),
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
        existing = json.loads(output.read_text(encoding="utf-8"))
        keys = (
            "hypothesis_status",
            "audit_integrity",
            "signatures",
            "reconciliation",
            "h57_gaps",
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
            {
                "hypothesis_status": report["hypothesis_status"],
                "audit_integrity": report["audit_integrity"],
                "h57_gap_checks": report["h57_gap_checks"],
            },
            indent=2,
        )
    )
    return 0 if report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
