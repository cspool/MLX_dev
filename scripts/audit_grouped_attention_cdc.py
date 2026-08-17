#!/usr/bin/env python3
"""Audit H82 grouped-event compressed-attention execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from mlxsim.dsagen_grouped_attention import compile_grouped_attention
from mlxsim.dsagen_overlay import canonical_json
from mlxsim.repeat_folding import fit_affine, relative_error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/grouped_attention_cdc_v1.yaml"


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


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    files = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    parents = {}
    parent_checks = {}
    for name in ("signature", "fft_estimator", "pe_contract"):
        spec = config["frozen_inputs"][name]
        report = json.loads(
            (PROJECT_ROOT / spec["path"]).read_text(encoding="utf-8")
        )
        parents[name] = report
        parent_checks[name] = _parent_check(report, spec)

    output_root = PROJECT_ROOT / config["output_root"]
    compile_path = output_root / "grouped-attention-compile-manifest.json"
    run_path = output_root / "grouped-attention-run-manifest.json"
    compile_file = qualify(compile_path)
    run_file = qualify(run_path)
    compiler = json.loads(compile_path.read_text(encoding="utf-8"))
    run = json.loads(run_path.read_text(encoding="utf-8"))

    compile_checks = {}
    structure_checks = {}
    execution_checks = {}
    measurements = {}
    for key, item in compiler["outputs"].items():
        shape_name, scale_text = key.split("-q")
        scale = int(scale_text)
        shape = config["shapes"][shape_name]
        expected_document, expected_metadata = compile_grouped_attention(
            name=key,
            retained_length=int(shape["retained_length"]),
            hidden_dimension=int(config["hardware"]["hidden_dimension"]),
            scale=scale,
        )
        artifact_path = PROJECT_ROOT / item["artifact"]["path"]
        artifact = qualify(artifact_path, item["artifact"])
        compile_checks[key] = (
            artifact["pass"]
            and artifact_path.read_text(encoding="utf-8")
            == canonical_json(expected_document)
            and item["metadata"] == expected_metadata
        )
        qk_blocks = [
            block for block in expected_document["blocks"] if "_qk_l" in block["id"]
        ]
        sv_blocks = [
            block for block in expected_document["blocks"] if "_sv_l" in block["id"]
        ]
        div_blocks = [
            block for block in expected_document["blocks"] if "_div_l" in block["id"]
        ]
        retained = int(shape["retained_length"])
        hidden = int(config["hardware"]["hidden_dimension"])
        structure_checks[key] = (
            len(qk_blocks) == len(sv_blocks) == len(div_blocks) == 4
            and all(block["trip_count"] == scale * hidden for block in qk_blocks)
            and all(
                block["instructions"][0]["emit_event_period"] == hidden
                for block in qk_blocks
            )
            and all(block["trip_count"] == scale * hidden for block in sv_blocks)
            and all(block["wait_event_period"] == hidden for block in sv_blocks)
            and all(
                block["instructions"][0]["emit_event_period"] == retained
                for block in sv_blocks
            )
            and all(
                block["trip_count"] == scale * hidden // retained
                for block in div_blocks
            )
        )

        first = run["runs"][key]["first"]
        second = run["runs"][key]["second"]
        first_file = qualify(
            PROJECT_ROOT / first["path"], {"sha256": first["sha256"]}
        )
        second_file = qualify(
            PROJECT_ROOT / second["path"], {"sha256": second["sha256"]}
        )
        summary = first["summary"]
        expected_instructions = sum(expected_metadata["pipeline_counts"].values())
        checks = {
            "files": first_file["pass"] and second_file["pass"],
            "replay": first["sha256"] == second["sha256"],
            "done": summary["done"] is True,
            "paper_static": summary["pe_dependency_model"] == "paper_static",
            "fixed_memory": summary["memory_backend"] == "fixed",
            "instructions": summary["instructions_issued"]
            == summary["instructions_completed"]
            == expected_instructions,
            "pipelines": summary["issued_by_pipeline"]
            == expected_metadata["pipeline_counts"],
            "events": summary["boundary_events_emitted"]
            == expected_metadata["dynamic_event_count"],
            "no_routes": summary["route_hops"] == 0,
        }
        execution_checks[key] = all(checks.values())
        measurements[key] = {
            "cycles": int(summary["cycles"]),
            "summary": summary,
            "metadata": expected_metadata,
            "checks": checks,
        }

    scales = [
        int(value) for value in [*config["fit_scales"], *config["holdout_scales"]]
    ]
    work_checks = {}
    for shape_name in config["shapes"]:
        base = measurements[f"{shape_name}-q1"]
        for scale in scales:
            item = measurements[f"{shape_name}-q{scale}"]
            work_checks[f"{shape_name}-q{scale}"] = (
                item["summary"]["instructions_issued"]
                == base["summary"]["instructions_issued"] * scale
                and item["summary"]["boundary_events_emitted"]
                == base["summary"]["boundary_events_emitted"] * scale
                and all(
                    item["metadata"]["operation_counts"][operation]
                    == base["metadata"]["operation_counts"][operation] * scale
                    for operation in base["metadata"]["operation_counts"]
                )
            )

    full_conservation = {}
    models = {}
    errors = []
    fit_scales = [int(value) for value in config["fit_scales"]]
    holdout_scales = [int(value) for value in config["holdout_scales"]]
    limit = float(config["cycle_relative_error_limit"])
    for shape_name, shape in config["shapes"].items():
        base = measurements[f"{shape_name}-q1"]["metadata"]
        full_scale = int(shape["full_scale"])
        expected = parents["signature"]["signatures"][shape_name][
            "compressed_attention"
        ]["fu_instruction_instances"]
        actual = {
            "fma": base["operation_counts"]["fma"] * full_scale * 8,
            "fmax": base["operation_counts"]["fmax"] * full_scale * 8,
            "fexp": base["operation_counts"]["fexp"] * full_scale * 8,
            "alu_add": base["operation_counts"]["add"] * full_scale * 8,
            "fdiv": base["operation_counts"]["fdiv"] * full_scale * 8,
        }
        checks = {name: actual[name] == expected[name] for name in expected}
        full_conservation[shape_name] = {
            "full_scale": full_scale,
            "expected": expected,
            "derived": actual,
            "checks": checks,
            "pass": all(checks.values()),
        }
        model = fit_affine(
            fit_scales[0],
            measurements[f"{shape_name}-q{fit_scales[0]}"]["cycles"],
            fit_scales[1],
            measurements[f"{shape_name}-q{fit_scales[1]}"]["cycles"],
        )
        holdouts = []
        for scale in holdout_scales:
            actual_cycles = measurements[f"{shape_name}-q{scale}"]["cycles"]
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
        PROJECT_ROOT / run["legacy"]["path"],
        {"sha256": run["legacy"]["sha256"]},
    )
    legacy_expected = config["frozen_inputs"]["legacy_summary"]["sha256"]
    legacy_check = legacy["pass"] and run["legacy"]["sha256"] == legacy_expected
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
        "fields": "emit_event_period{1}" in header_text
        and "wait_event_period{1}" in header_text,
        "emit_group": "(state.iteration + 1) % instruction.emit_event_period"
        in source_text,
        "wait_group": "state.iteration / block.wait_event_period" in source_text,
        "validation": "emit-event period must divide block trip count" in source_text,
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
        "compile_manifest": compile_file["pass"]
        and len(compiler["outputs"]) == 8,
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
