#!/usr/bin/env python3
"""Audit H83 combined SIMD32 Attention with four DSAGEN SRAM ports."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from mlxsim.dsagen_combined_attention import compile_combined_attention
from mlxsim.dsagen_overlay import canonical_json
from mlxsim.repeat_folding import fit_affine, relative_error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/combined_attention_memory_v1.yaml"


def qualify(path: Path, expected: dict[str, Any] | None = None) -> dict[str, Any]:
    exists = path.is_file()
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if exists else None
    checks = {"is_file": exists}
    if expected and "sha256" in expected:
        checks["sha256"] = digest == expected["sha256"]
    try:
        display = str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        display = str(path)
    return {
        "path": display,
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
        report.get("hypothesis_status") == spec["required_status"]
        and report.get("audit_integrity") is spec["required_integrity"]
    )


def _expected_fu(signature: dict[str, Any], shape_name: str) -> dict[str, int]:
    shape = signature["signatures"][shape_name]
    fft = shape["fft_compression"]["fu_instruction_instances"]
    attention = shape["compressed_attention"]["fu_instruction_instances"]
    return {
        "fma": int(fft["fma"] + attention["fma"]),
        "add": int(fft["alu_add"] + attention["alu_add"]),
        "shuffle": int(fft["shuffle"]),
        "fmax": int(attention["fmax"]),
        "fexp": int(attention["fexp"]),
        "fdiv": int(attention["fdiv"]),
    }


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    files = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    parents = {}
    parent_checks = {}
    for name in (
        "signature",
        "fft_estimator",
        "grouped_attention",
        "spad",
        "column_ports",
        "pe_contract",
    ):
        spec = config["frozen_inputs"][name]
        report = json.loads(
            (PROJECT_ROOT / spec["path"]).read_text(encoding="utf-8")
        )
        parents[name] = report
        parent_checks[name] = _parent_check(report, spec)

    output_root = PROJECT_ROOT / config["output_root"]
    compile_path = output_root / "combined-attention-compile-manifest.json"
    run_path = output_root / "combined-attention-run-manifest.json"
    compile_file = qualify(compile_path)
    run_file = qualify(run_path)
    compiler = json.loads(compile_path.read_text(encoding="utf-8"))
    run = json.loads(run_path.read_text(encoding="utf-8"))

    compile_checks = {}
    structure_checks = {}
    execution_checks = {}
    measurements = {}
    for key, item in compiler["outputs"].items():
        shape_name, scale_text = key.split("-u")
        scale = int(scale_text)
        shape = config["shapes"][shape_name]
        document, metadata = compile_combined_attention(
            name=key,
            sequence_length=int(shape["sequence_length"]),
            retained_length=int(shape["retained_length"]),
            hidden_dimension=int(config["hardware"]["hidden_dimension"]),
            forward_stages=int(shape["fft_forward_stages"]),
            inverse_stages=int(shape["fft_inverse_stages"]),
            fft_scale=scale * int(shape["fft_scale_per_u"]),
            attention_scale=scale * int(shape["attention_scale_per_u"]),
            vector_bytes=int(config["hardware"]["vector_bytes"]),
            active_window=int(config["hardware"]["active_window"]),
        )
        artifact_path = PROJECT_ROOT / item["artifact"]["path"]
        artifact = qualify(artifact_path, item["artifact"])
        compile_checks[key] = (
            artifact["pass"]
            and artifact_path.read_text(encoding="utf-8") == canonical_json(document)
            and item["metadata"] == metadata
        )
        inverse_first = [
            block
            for block in document["blocks"]
            if f"_fft_s{int(shape['fft_forward_stages']) + 1}_" in block["id"]
        ]
        qk_blocks = [block for block in document["blocks"] if "_qk_l" in block["id"]]
        sv_blocks = [block for block in document["blocks"] if "_sv_l" in block["id"]]
        qk_periods = {
            int(value)
            for block in qk_blocks
            for value in block["wait_event_periods"].values()
        }
        structure_checks[key] = (
            document["active_window"] == int(config["hardware"]["active_window"])
            and document["memory_backend"] == "dsagen_spad"
            and metadata["simd_width"] == 32
            and metadata["vector_bytes"] == 64
            and metadata["max_active_instruction_footprint_per_pe"]
            <= int(config["hardware"]["instruction_entries_per_pe"])
            and len(inverse_first) == 12
            and all(
                list(block["wait_event_multiplicities"].values()) == [2]
                for block in inverse_first
            )
            and qk_periods == {int(shape["sequence_length"])}
            and all(
                int(shape["sequence_length"])
                in set(block["wait_event_periods"].values())
                and int(config["hardware"]["hidden_dimension"])
                in set(block["wait_event_periods"].values())
                for block in sv_blocks
            )
        )

        first = run["runs"][key]["first"]
        second = run["runs"][key]["second"]
        first_summary = qualify(
            PROJECT_ROOT / first["summary_path"],
            {"sha256": first["summary_sha256"]},
        )
        second_summary = qualify(
            PROJECT_ROOT / second["summary_path"],
            {"sha256": second["summary_sha256"]},
        )
        first_adapter = qualify(
            PROJECT_ROOT / first["adapter_path"],
            {"sha256": first["adapter_sha256"]},
        )
        second_adapter = qualify(
            PROJECT_ROOT / second["adapter_path"],
            {"sha256": second["adapter_sha256"]},
        )
        summary = first["summary"]
        adapter = first["adapter"]
        per_port_requests = sum(port["requests"] for port in adapter["per_port"])
        per_port_responses = sum(port["responses"] for port in adapter["per_port"])
        expected_instructions = sum(metadata["pipeline_counts"].values())
        checks = {
            "files": first_summary["pass"]
            and second_summary["pass"]
            and first_adapter["pass"]
            and second_adapter["pass"],
            "replay": first["summary_sha256"] == second["summary_sha256"]
            and first["adapter_sha256"] == second["adapter_sha256"],
            "done": summary["done"] is True,
            "paper_static": summary["pe_dependency_model"] == "paper_static",
            "memory_backend": summary["memory_backend"] == "adapter",
            "instructions": summary["instructions_issued"]
            == summary["instructions_completed"]
            == expected_instructions,
            "pipelines": summary["issued_by_pipeline"] == metadata["pipeline_counts"],
            "events": summary["boundary_events_emitted"]
            == metadata["dynamic_event_count"],
            "memory": summary["external_memory_requests"]
            == summary["external_memory_completions"]
            == metadata["memory_requests"]
            == adapter["requests"]
            == adapter["responses"]
            == per_port_requests
            == per_port_responses,
            "ports": adapter["ports"] == int(config["hardware"]["spad_ports"]),
            "axis": adapter["axis"] == config["hardware"]["spad_axis"],
        }
        execution_checks[key] = all(checks.values())
        measurements[key] = {
            "cycles": int(summary["cycles"]),
            "summary": summary,
            "adapter": adapter,
            "metadata": metadata,
            "checks": checks,
        }

    scales = [
        int(value) for value in [*config["fit_scales"], *config["holdout_scales"]]
    ]
    base_scale = min(scales)
    work_checks = {}
    for shape_name in config["shapes"]:
        base = measurements[f"{shape_name}-u{base_scale}"]
        for scale in scales:
            item = measurements[f"{shape_name}-u{scale}"]
            scalar_fields = (
                "instructions_issued",
                "instructions_completed",
                "boundary_events_emitted",
                "route_hops",
                "skip_hops",
                "unit_hops",
                "external_memory_requests",
                "external_memory_completions",
            )
            summary_linear = all(
                item["summary"][field] * base_scale
                == base["summary"][field] * scale
                for field in scalar_fields
            )
            metadata_linear = all(
                item["metadata"][field] * base_scale
                == base["metadata"][field] * scale
                for field in (
                    "dynamic_event_count",
                    "memory_requests",
                    "offchip_bytes",
                    "boundary_xfers",
                    "boundary_bytes",
                )
            )
            operation_linear = all(
                item["metadata"]["operation_counts"][operation] * base_scale
                == base["metadata"]["operation_counts"][operation] * scale
                for operation in base["metadata"]["operation_counts"]
            )
            work_checks[f"{shape_name}-u{scale}"] = (
                summary_linear and metadata_linear and operation_linear
            )

    full_conservation = {}
    models = {}
    errors = []
    fit_scales = [int(value) for value in config["fit_scales"]]
    holdout_scales = [int(value) for value in config["holdout_scales"]]
    limit = float(config["cycle_relative_error_limit"])
    for shape_name, shape in config["shapes"].items():
        base = measurements[f"{shape_name}-u{base_scale}"]["metadata"]
        full_scale = int(shape["full_scale"])
        expected_fu = _expected_fu(parents["signature"], shape_name)
        derived_fu = {
            operation: base["operation_counts"][operation]
            * full_scale
            * 32
            // base_scale
            for operation in expected_fu
        }
        fu_checks = {
            operation: derived_fu[operation] == expected_fu[operation]
            for operation in expected_fu
        }
        derived_offchip = base["offchip_bytes"] * full_scale // base_scale
        derived_boundary = base["boundary_bytes"] * full_scale // base_scale
        byte_checks = {
            "offchip": derived_offchip == int(shape["full_offchip_bytes"]),
            "boundary": derived_boundary == int(shape["boundary_bytes"]),
        }
        full_conservation[shape_name] = {
            "full_scale": full_scale,
            "expected_fu": expected_fu,
            "derived_fu": derived_fu,
            "fu_checks": fu_checks,
            "expected_offchip_bytes": int(shape["full_offchip_bytes"]),
            "derived_offchip_bytes": derived_offchip,
            "expected_boundary_bytes": int(shape["boundary_bytes"]),
            "derived_boundary_bytes": derived_boundary,
            "byte_checks": byte_checks,
            "pass": all(fu_checks.values()) and all(byte_checks.values()),
        }
        model = fit_affine(
            fit_scales[0],
            measurements[f"{shape_name}-u{fit_scales[0]}"]["cycles"],
            fit_scales[1],
            measurements[f"{shape_name}-u{fit_scales[1]}"]["cycles"],
        )
        holdouts = []
        for scale in holdout_scales:
            actual_cycles = measurements[f"{shape_name}-u{scale}"]["cycles"]
            predicted = model.predict(scale)
            error = relative_error(predicted, actual_cycles)
            errors.append(error)
            holdouts.append(
                {
                    "scale": scale,
                    "actual_cycles": actual_cycles,
                    "predicted_cycles": predicted,
                    "relative_error": error,
                    "pass_5pct": error <= limit,
                }
            )
        models[shape_name] = {
            "intercept": model.intercept,
            "slope_cycles_per_scale": model.slope,
            "holdouts": holdouts,
            "full_scale": full_scale,
            "full_work_predicted_cycles": model.predict(full_scale),
        }

    numerical = {
        "passing_holdouts": sum(error <= limit for error in errors),
        "total_holdouts": len(errors),
        "mape": sum(errors) / len(errors),
        "max_error": max(errors),
        "all_holdouts_pass": all(error <= limit for error in errors),
    }
    legacy = qualify(
        PROJECT_ROOT / run["legacy"]["summary_path"],
        {"sha256": run["legacy"]["summary_sha256"]},
    )
    legacy_expected = config["frozen_inputs"]["legacy_summary"]["sha256"]
    legacy_check = legacy["pass"] and run["legacy"]["summary_sha256"] == legacy_expected
    patch_path = PROJECT_ROOT / config["source_layout"]["tracked_patch"]
    reverse = subprocess.run(
        ["patch", "--dry-run", "-R", "-p1", "-i", str(patch_path)],
        cwd=PROJECT_ROOT / "third_party/dsa-framework/dsa-gem5",
        capture_output=True,
        text=True,
        check=False,
    )
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    header_text = (
        PROJECT_ROOT / config["source_layout"]["overlay_header"]
    ).read_text(encoding="utf-8")
    source_text = (
        PROJECT_ROOT / config["source_layout"]["overlay_source"]
    ).read_text(encoding="utf-8")
    source_checks = {
        "period_map": "wait_event_periods" in header_text,
        "multiplicity_map": "wait_event_multiplicities" in header_text,
        "readiness_formula": "(state.iteration / period + 1) * multiplicity - 1"
        in source_text,
        "named_event_validation": "requires a named wait event" in source_text,
        "patch_reverse": reverse.returncode == 0,
    }
    implementation_text = "\n".join(
        (PROJECT_ROOT / path).read_text(encoding="utf-8")
        for name, path in config["source_layout"].items()
        if name in {"compiler_core", "compiler", "runner"}
    )
    integrity_checks = {
        "frozen_files": all(item["pass"] for item in files.values()),
        "parents": all(parent_checks.values()),
        "compile_manifest": compile_file["pass"] and len(compiler["outputs"]) == 8,
        "compiler_replay": all(compile_checks.values()),
        "group_structure": all(structure_checks.values()),
        "run_manifest": run_file["pass"],
        "run_replays": all(run["checks"].values()),
        "executions": all(execution_checks.values()),
        "legacy_exact": legacy_check,
        "work_linear": all(work_checks.values()),
        "full_work_conserved": all(
            item["pass"] for item in full_conservation.values()
        ),
        "source_files": all(item["pass"] for item in source_files.values()),
        "source_semantics": all(source_checks.values()),
        "targets_absent": "paper_targets" not in implementation_text,
        "targets_consumed": compiler["paper_performance_targets_consumed"] is False
        and run["paper_performance_targets_consumed"] is False,
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
        "hypothesis_status": (
            "supported" if integrity and numerical["all_holdouts_pass"] else "rejected"
        ),
        "audit_integrity": integrity,
        "frozen_inputs": files,
        "parent_checks": parent_checks,
        "compile_manifest": compile_file,
        "run_manifest": run_file,
        "compile_checks": compile_checks,
        "structure_checks": structure_checks,
        "execution_checks": execution_checks,
        "legacy_regression": {"artifact": legacy, "exact": legacy_check},
        "work_linearity_checks": work_checks,
        "full_work_conservation": full_conservation,
        "measurements": measurements,
        "models": models,
        "numerical": numerical,
        "source_files": source_files,
        "source_checks": source_checks,
        "integrity_checks": integrity_checks,
        "paper_performance_targets_consumed": False,
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
            "legacy_regression",
            "full_work_conservation",
            "models",
            "numerical",
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
                "numerical": report["numerical"],
                "full_work_cycles": {
                    key: item["full_work_predicted_cycles"]
                    for key, item in report["models"].items()
                },
            },
            indent=2,
        )
    )
    return 0 if report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
