#!/usr/bin/env python3
"""Audit H91 Figure 21 batch-8 one-layer contracts and u=1 graphs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from mlxsim.dsagen_overlay import canonical_json
from mlxsim.fig21_layer_contract import build_shape_contract

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig21_layer_contract_v1.yaml"


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


def qualify_unfrozen(path: Path) -> dict[str, Any]:
    exists = path.is_file()
    return {
        "path": str(path.relative_to(PROJECT_ROOT)) if exists else str(path),
        "bytes": path.stat().st_size if exists else None,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if exists else None,
        "pass": exists,
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


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    files = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    parents = {
        name: json.loads((PROJECT_ROOT / spec["path"]).read_text(encoding="utf-8"))
        for name, spec in config["frozen_inputs"].items()
        if name in {"identity", "attention", "full_block"}
    }
    parent_checks = {
        name: report["hypothesis_status"] == config["frozen_inputs"][name]["required_status"]
        and report["audit_integrity"]
        is config["frozen_inputs"][name]["required_integrity"]
        for name, report in parents.items()
    }
    output_root = PROJECT_ROOT / config["output_root"]
    manifest_path = output_root / "fig21-layer-contract-manifest.json"
    manifest_file = qualify_unfrozen(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shape = config["shape"]
    replay_checks = {}
    component_checks = {}
    artifacts = {}
    for n_value in shape["sequence_lengths"]:
        n = int(n_value)
        key = f"N{n}"
        document, contract = build_shape_contract(
            sequence_length=n,
            batch=int(shape["batch"]),
            hidden_dimension=int(shape["hidden_dimension"]),
            ffn_dimension=int(shape["ffn_dimension"]),
            simd_width=int(shape["simd_width"]),
            vector_bytes=int(shape["vector_bytes"]),
            active_window=int(shape["active_window"]),
            logical_profile=parents["identity"]["logical_profiles"][key],
        )
        artifact_spec = manifest["outputs"][key]
        path = PROJECT_ROOT / artifact_spec["path"]
        artifact = qualify(path, artifact_spec)
        artifacts[key] = artifact
        replay_checks[key] = artifact["pass"] and path.read_text(
            encoding="utf-8"
        ) == canonical_json(document) and manifest["contracts"][key] == contract
        checks = {
            "contract": all(contract["checks"].values()),
            "structured_components": set(contract["structured_components"])
            == {"qkv", "attention", "output", "ffn1", "ffn2"},
            "dense_components": set(contract["dense_components"])
            == {"qkv", "attention", "output", "ffn1", "ffn2"},
            "main_operations": all(
                contract[mode + "_components"][component]["operations"]
                == parents["identity"]["logical_profiles"][key][mode][component][
                    "operations"
                ]
                for mode in ("structured", "dense")
                for component in ("qkv", "attention", "output", "ffn1", "ffn2")
            ),
            "elementwise": contract["elementwise"]["evidence"]
            == config["elementwise_contract"]["evidence"]
            and all(
                value > 0
                for value in contract["elementwise"][
                    "fu_instruction_instances"
                ].values()
            ),
            "full_scale": contract["full_scale"]
            == int(shape["batch"]) * (n // 2) ** 2 // 128,
            "attention_footprint": contract["unit_attention_metadata"][
                "max_active_instruction_footprint_per_pe"
            ]
            <= 32,
        }
        component_checks[key] = all(checks.values())
    source_files = {
        name: qualify_unfrozen(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    implementation_text = "\n".join(
        (PROJECT_ROOT / path).read_text(encoding="utf-8")
        for name, path in config["source_layout"].items()
        if name != "auditor"
    ).lower()
    summary = {
        "shape_count": len(manifest["contracts"]),
        "all_contract_checks_pass": all(
            all(contract["checks"].values())
            for contract in manifest["contracts"].values()
        ),
        "all_replays_match": all(replay_checks.values()),
        "matched_one_layer_contract_available": True,
        "thirty_two_layer_timing_executed": False,
    }
    integrity_checks = {
        "frozen_files": all(item["pass"] for item in files.values()),
        "parents": all(parent_checks.values()),
        "manifest": manifest_file["pass"]
        and manifest["paper_performance_targets_consumed"] is False,
        "five_shapes": set(manifest["contracts"])
        == {f"N{int(n)}" for n in shape["sequence_lengths"]},
        "replays": all(replay_checks.values()),
        "components": all(component_checks.values()),
        "source_files": all(item["pass"] for item in source_files.values()),
        "targets_absent": "paper_targets" not in implementation_text,
        "summary": summary["all_contract_checks_pass"]
        and summary["all_replays_match"],
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
        "manifest": manifest_file,
        "artifacts": artifacts,
        "contracts": manifest["contracts"],
        "replay_checks": replay_checks,
        "component_checks": component_checks,
        "source_files": source_files,
        "summary": summary,
        "integrity_checks": integrity_checks,
        "paper_performance_targets_consumed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
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
            "contracts",
            "replay_checks",
            "component_checks",
            "summary",
            "integrity_checks",
        )
        matches = all(existing.get(key) == report.get(key) for key in keys)
        print(json.dumps({"existing_matches": matches, **report}, indent=2))
        return 0 if matches else 1
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0 if report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
