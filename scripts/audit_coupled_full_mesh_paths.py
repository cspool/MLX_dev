#!/usr/bin/env python3
"""Audit H114 coupled full-mesh q folding and full reconstruction."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from mlxsim.coupled_full_mesh_paths import (
    compile_coupled_path,
    contract_full_scale,
)
from mlxsim.dsagen_overlay import canonical_json
from mlxsim.repeat_folding import fit_affine, relative_error

try:
    from scripts.audit_compute_dma_overlap import git_commit, qualify
except ModuleNotFoundError:
    from audit_compute_dma_overlap import git_commit, qualify

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/coupled_full_mesh_paths_v1.yaml"


def partition_identity(
    coupled: dict[str, Any], baseline: dict[str, Any], tile_count: int
) -> bool:
    if (
        coupled["functional_units"] != baseline["functional_units"]
        or coupled["routing"] != baseline["routing"]
        or coupled["pipelines"] != baseline["pipelines"]
        or coupled["pe_dependency_model"] != baseline["pe_dependency_model"]
        or coupled["active_window"] != baseline["active_window"]
        or coupled["dpu"] != baseline["dpu"]
        or len(coupled["blocks"]) != len(baseline["blocks"]) * tile_count
    ):
        return False
    base_tags = sorted({int(block["tag"]) for block in baseline["blocks"]})
    minimum_tag = base_tags[0]
    tag_span = base_tags[-1] - minimum_tag + 1
    block_count = len(baseline["blocks"])
    ignored_block = {
        "id",
        "tag",
        "trip_count",
        "instance_base",
        "predecessors",
        "wait_events",
        "wait_event_periods",
        "wait_event_multiplicities",
        "instructions",
    }
    ignored_instruction = {
        "id",
        "emit_event",
        "emit_event_period",
        "memory_address",
        "memory_address_sequence",
        "memory_external",
    }
    for source_index, source in enumerate(baseline["blocks"]):
        members = [
            coupled["blocks"][tile * block_count + source_index]
            for tile in range(tile_count)
        ]
        if sum(int(member["trip_count"]) for member in members) != int(
            source["trip_count"]
        ):
            return False
        for tile, member in enumerate(members):
            if (
                member["id"] != f"tile{tile}__{source['id']}"
                or int(member["tag"])
                != tile * tag_span + int(source["tag"]) - minimum_tag + 1
                or {key: value for key, value in member.items() if key not in ignored_block}
                != {key: value for key, value in source.items() if key not in ignored_block}
                or len(member["instructions"]) != len(source["instructions"])
                or len(member.get("wait_events", []))
                != len(source.get("wait_events", []))
                or any(
                    not left.endswith(f"__{right}")
                    for left, right in zip(
                        member.get("wait_events", []), source.get("wait_events", [])
                    )
                )
            ):
                return False
            for left, right in zip(member["instructions"], source["instructions"]):
                if {
                    key: value
                    for key, value in left.items()
                    if key not in ignored_instruction
                } != {
                    key: value
                    for key, value in right.items()
                    if key not in ignored_instruction
                }:
                    return False
                if right.get("emit_event") and (
                    not left["emit_event"].endswith(f"__{right['emit_event']}")
                    or int(left["emit_event_period"])
                    != (
                        int(member["trip_count"])
                        if tile_count > 1
                        else int(right.get("emit_event_period", 1))
                    )
                ):
                    return False
    return True


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trace_patch_audit(config: dict[str, Any]) -> dict[str, Any]:
    patch = PROJECT_ROOT / config["source_layout"]["trace_patch"]
    descendant_patch = (
        PROJECT_ROOT
        / "patches/dsagen/dsa-gem5-historical-multiport-spad-v1.patch"
    )
    current_header = PROJECT_ROOT / config["source_layout"]["adapter_header"]
    current_source = PROJECT_ROOT / config["source_layout"]["adapter_source"]
    report = {
        "patch": qualify(patch),
        "reverse_check": False,
        "baseline_header": False,
        "baseline_source": False,
        "forward_check": False,
        "round_trip_exact": False,
    }
    if not patch.is_file():
        report["pass"] = False
        return report
    with tempfile.TemporaryDirectory(prefix="mlx-h114-patch-") as temporary:
        root = Path(temporary)
        target = root / "simulator_ext/dsagen"
        target.mkdir(parents=True)
        header = target / "historical_dpu_memory.hh"
        source = target / "historical_dpu_memory.cc"
        shutil.copy2(current_header, header)
        shutil.copy2(current_source, source)
        descendant_reverse = subprocess.run(
            [
                "git",
                "apply",
                "--unidiff-zero",
                "-R",
                "--check",
                str(descendant_patch),
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if descendant_reverse.returncode != 0:
            report["pass"] = False
            return report
        subprocess.run(
            [
                "git",
                "apply",
                "--unidiff-zero",
                "-R",
                str(descendant_patch),
            ],
            cwd=root,
            check=True,
        )
        reverse = subprocess.run(
            ["git", "apply", "-R", "--check", str(patch)],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        report["reverse_check"] = reverse.returncode == 0
        if reverse.returncode == 0:
            subprocess.run(["git", "apply", "-R", str(patch)], cwd=root, check=True)
            report["baseline_header"] = sha256(header) == config["trace_control"][
                "adapter_header_before_sha256"
            ]
            report["baseline_source"] = sha256(source) == config["trace_control"][
                "adapter_source_before_sha256"
            ]
            forward = subprocess.run(
                ["git", "apply", "--check", str(patch)],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            report["forward_check"] = forward.returncode == 0
            if forward.returncode == 0:
                subprocess.run(["git", "apply", str(patch)], cwd=root, check=True)
                subprocess.run(
                    [
                        "git",
                        "apply",
                        "--unidiff-zero",
                        "--check",
                        str(descendant_patch),
                    ],
                    cwd=root,
                    check=True,
                )
                subprocess.run(
                    [
                        "git",
                        "apply",
                        "--unidiff-zero",
                        str(descendant_patch),
                    ],
                    cwd=root,
                    check=True,
                )
                report["round_trip_exact"] = (
                    header.read_bytes() == current_header.read_bytes()
                    and source.read_bytes() == current_source.read_bytes()
                )
    report["pass"] = all(
        report[key]
        for key in (
            "reverse_check",
            "baseline_header",
            "baseline_source",
            "forward_check",
            "round_trip_exact",
        )
    )
    return report


def capacity_patch_audit(config: dict[str, Any]) -> dict[str, Any]:
    patch = PROJECT_ROOT / config["source_layout"]["capacity_patch"]
    active_patch = PROJECT_ROOT / config["source_layout"]["active_scan_patch"]
    resident_patch = (
        PROJECT_ROOT
        / "patches/dsagen/dsa-gem5-active-window-instruction-capacity-v1.patch"
    )
    current_header = PROJECT_ROOT / config["source_layout"]["overlay_header"]
    current_source = PROJECT_ROOT / config["source_layout"]["overlay_source"]
    report = {
        "patch": qualify(patch),
        "active_patch": qualify(active_patch),
        "resident_patch": qualify(resident_patch),
        "resident_reverse_check": False,
        "active_reverse_check": False,
        "reverse_check": False,
        "baseline_header": False,
        "baseline_source": False,
        "forward_check": False,
        "active_forward_check": False,
        "resident_forward_check": False,
        "round_trip_exact": False,
    }
    if not patch.is_file():
        report["pass"] = False
        return report
    with tempfile.TemporaryDirectory(prefix="mlx-h114-capacity-") as temporary:
        root = Path(temporary)
        target = root / "src/cpu/minor/ssim"
        target.mkdir(parents=True)
        header = target / "mlx_overlay.hh"
        source = target / "mlx_overlay.cc"
        shutil.copy2(current_header, header)
        shutil.copy2(current_source, source)
        resident_reverse = subprocess.run(
            ["git", "apply", "-R", "--check", str(resident_patch)],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        report["resident_reverse_check"] = resident_reverse.returncode == 0
        if resident_reverse.returncode != 0:
            report["pass"] = False
            return report
        subprocess.run(
            ["git", "apply", "-R", str(resident_patch)],
            cwd=root,
            check=True,
        )
        active_reverse = subprocess.run(
            [
                "git",
                "apply",
                "--unidiff-zero",
                "-R",
                "--check",
                str(active_patch),
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        report["active_reverse_check"] = active_reverse.returncode == 0
        if active_reverse.returncode != 0:
            report["pass"] = False
            return report
        subprocess.run(
            ["git", "apply", "--unidiff-zero", "-R", str(active_patch)],
            cwd=root,
            check=True,
        )
        reverse = subprocess.run(
            ["git", "apply", "-R", "--check", str(patch)],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        report["reverse_check"] = reverse.returncode == 0
        if reverse.returncode == 0:
            subprocess.run(["git", "apply", "-R", str(patch)], cwd=root, check=True)
            report["baseline_header"] = sha256(header) == config[
                "capacity_control"
            ]["overlay_header_before_sha256"]
            report["baseline_source"] = sha256(source) == config[
                "capacity_control"
            ]["overlay_source_before_sha256"]
            forward = subprocess.run(
                ["git", "apply", "--check", str(patch)],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            report["forward_check"] = forward.returncode == 0
            if forward.returncode == 0:
                subprocess.run(["git", "apply", str(patch)], cwd=root, check=True)
                active_forward = subprocess.run(
                    [
                        "git",
                        "apply",
                        "--unidiff-zero",
                        "--check",
                        str(active_patch),
                    ],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                report["active_forward_check"] = active_forward.returncode == 0
                if active_forward.returncode == 0:
                    subprocess.run(
                        ["git", "apply", "--unidiff-zero", str(active_patch)],
                        cwd=root,
                        check=True,
                    )
                    resident_forward = subprocess.run(
                        ["git", "apply", "--check", str(resident_patch)],
                        cwd=root,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    report["resident_forward_check"] = (
                        resident_forward.returncode == 0
                    )
                    if resident_forward.returncode == 0:
                        subprocess.run(
                            ["git", "apply", str(resident_patch)],
                            cwd=root,
                            check=True,
                        )
                report["round_trip_exact"] = (
                    header.read_bytes() == current_header.read_bytes()
                    and source.read_bytes() == current_source.read_bytes()
                )
    report["pass"] = all(
        report[key]
        for key in (
            "reverse_check",
            "resident_reverse_check",
            "active_reverse_check",
            "baseline_header",
            "baseline_source",
            "forward_check",
            "active_forward_check",
            "resident_forward_check",
            "round_trip_exact",
        )
    )
    return report


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    h113 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h113"]["path"]).read_text()
    )
    h110 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h110"]["path"]).read_text()
    )
    h107 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h107"]["path"]).read_text()
    )
    parent_checks = {
        "h113": h113["hypothesis_status"] == "supported"
        and h113["audit_integrity"] is True,
        "h107": h107["hypothesis_status"] == "supported"
        and h107["audit_integrity"] is True,
        "h110_partial": h110["hypothesis_status"] == "rejected"
        and h110["audit_integrity"] is True
        and h110["summary"]["all_cycle_holdouts_pass"] is True
        and h110["summary"]["cycle_holdouts_passed"] == 96,
    }
    h110_compile_path = PROJECT_ROOT / h110["compile_manifest"]["path"]
    h110_run_path = PROJECT_ROOT / h110["run_manifest"]["path"]
    h110_compile_file = qualify(h110_compile_path, h110["compile_manifest"])
    h110_run_file = qualify(h110_run_path, h110["run_manifest"])
    h110_compile = json.loads(h110_compile_path.read_text())

    output_root = PROJECT_ROOT / config["output_root"]
    compile_path = output_root / "coupled-full-mesh-compile-manifest.json"
    run_path = output_root / "coupled-full-mesh-run-manifest.json"
    compile_file = qualify(compile_path)
    run_file = qualify(run_path)
    compiled = json.loads(compile_path.read_text())
    run = json.loads(run_path.read_text())

    compile_checks = {}
    semantic_checks = {}
    for run_key, item in compiled["outputs"].items():
        path_key, scale_text = run_key.rsplit("-q", 1)
        document, memory, metadata, h110_document = compile_coupled_path(
            run_key=run_key,
            contract=h110_compile["path_contracts"][path_key],
            path=h107["path_results"][path_key],
            scale=int(scale_text),
            config=config,
        )
        overlay_path = PROJECT_ROOT / item["overlay"]["path"]
        memory_path = PROJECT_ROOT / item["memory"]["path"]
        compile_checks[run_key] = (
            qualify(overlay_path, item["overlay"])["pass"]
            and qualify(memory_path, item["memory"])["pass"]
            and overlay_path.read_text() == canonical_json(document)
            and memory_path.read_text() == canonical_json(memory)
            and item["metadata"] == metadata
            and all(metadata["checks"].values())
        )
        semantic_checks[run_key] = partition_identity(
            document, h110_document, int(metadata["tile_count"])
        )

    records = {
        (item["run_key"], item["mode"], int(item["replay"])): item
        for item in run["records"]
    }
    record_checks = {}
    execution_checks = {}
    measurements = {}
    for run_key, item in compiled["outputs"].items():
        metadata = item["metadata"]
        optimized = records[(run_key, "optimized", 1)]
        summary = optimized["summary"]
        overlay, memory = summary["overlay"], summary["memory"]
        matching = [record for key, record in records.items() if key[0] == run_key]
        record_checks[run_key] = all(
            record["pass"]
            and record["returncode"] == 0
            and record["stderr"] == ""
            and qualify(
                PROJECT_ROOT / record["summary_path"],
                {"sha256": record["summary_sha256"]},
            )["pass"]
            for record in matching
        )
        checks = {
            "done": overlay["done"] is True and memory["idle"] is True,
            "mode": overlay["pe_dependency_model"] == "dpu_pipelined"
            and overlay["memory_backend"] == "dpu_memory",
            "contexts": overlay["iteration_contexts_per_block"]
            == int(config["hardware"]["iteration_contexts_per_block"])
            and overlay["max_inflight_iterations_per_block"]
            <= int(config["hardware"]["iteration_contexts_per_block"]),
            "instructions": overlay["instructions_issued"]
            == overlay["instructions_completed"]
            == sum(metadata["pipeline_counts"].values()),
            "pipelines": overlay["issued_by_pipeline"]
            == metadata["pipeline_counts"],
            "requests": overlay["external_memory_requests"]
            == overlay["external_memory_completions"]
            == metadata["memory_requests"]
            == memory["requests"]
            == memory["responses"],
            "adapter_split": memory["read_requests"]
            == metadata["pipeline_counts"]["load"]
            and memory["write_requests"] == metadata["pipeline_counts"]["store"],
            "bytes": memory["offchip_read_bytes"]
            == metadata["scaled_read_bytes"]
            and memory["offchip_write_bytes"] == metadata["scaled_write_bytes"],
            "tiles": memory["tile_count"]
            == memory["released_tiles"]
            == memory["drained_tiles"]
            == metadata["tile_count"],
            "ownership": memory["ownership_wait_checks"] > 0
            and memory["ownership_violations"] == 0
            and overlay["external_memory_wait_cycles"] > 0,
            "trace_off": metadata["memory_record_events"] is False,
            "target_free": summary["paper_performance_targets_consumed"] is False,
        }
        execution_checks[run_key] = checks
        h110_cycles = int(h110["measurements"][run_key]["cycles"])
        measurements[run_key] = {
            "cycles": int(summary["end_to_end_cycles"]),
            "overlay_cycles": int(summary["overlay_cycles"]),
            "h110_cycles": h110_cycles,
            "cycle_slowdown_vs_h110": int(summary["end_to_end_cycles"])
            / h110_cycles,
            "scalar_fma": int(metadata["scalar_fma"]),
            "fma_issue_utilization": int(metadata["scalar_fma"])
            / (
                int(summary["end_to_end_cycles"])
                * int(config["hardware"]["physical_pes"])
                * int(config["hardware"]["simd_width"])
            ),
            "tile_count": int(metadata["tile_count"]),
            "ownership_wait_checks": int(memory["ownership_wait_checks"]),
            "memory_wait_cycles": int(overlay["external_memory_wait_cycles"]),
        }

    fit_scales = [int(value) for value in config["scales"]["fit"]]
    holdout_scales = [int(value) for value in config["scales"]["holdout"]]
    limit = float(config["scales"]["cycle_relative_error_limit"])
    models = {}
    holdout_errors = []
    full_estimates = {}
    for path_key, contract in h110_compile["path_contracts"].items():
        model = fit_affine(
            fit_scales[0],
            measurements[f"{path_key}-q{fit_scales[0]}"]["cycles"],
            fit_scales[1],
            measurements[f"{path_key}-q{fit_scales[1]}"]["cycles"],
        )
        holdouts = []
        for scale in holdout_scales:
            actual = measurements[f"{path_key}-q{scale}"]["cycles"]
            predicted = model.predict(scale)
            error = relative_error(predicted, actual)
            holdout_errors.append(error)
            holdouts.append(
                {
                    "scale": scale,
                    "actual_cycles": actual,
                    "predicted_cycles": predicted,
                    "relative_error": error,
                    "pass_5pct": error <= limit,
                }
            )
        eligible = all(item["pass_5pct"] for item in holdouts)
        full_scale = contract_full_scale(contract)
        full_cycles = model.predict(full_scale) if eligible else None
        path = h107["path_results"][path_key]
        models[path_key] = {
            "family": path["family"],
            "intercept": model.intercept,
            "slope": model.slope,
            "holdouts": holdouts,
            "eligible": eligible,
            "full_scale": full_scale,
        }
        full_estimates[path_key] = {
            "cycles": full_cycles,
            "fma_issue_utilization": (
                int(path["fma_count"])
                / (
                    full_cycles
                    * int(config["hardware"]["physical_pes"])
                    * int(config["hardware"]["simd_width"])
                )
                if full_cycles is not None
                else None
            ),
            "offchip_bytes": int(path["selected_offchip_bytes"]),
            "tile_count": int(path["tile_count"]),
            "eligible": eligible,
        }

    reconstruction_checks = {
        path_key: full_estimates[path_key]["offchip_bytes"]
        == int(path["selected_offchip_bytes"])
        and full_estimates[path_key]["tile_count"] == int(path["tile_count"])
        and int(path["fma_count"])
        == int(h110_compile["path_contracts"][path_key]["actual"]["fu"]["fma"])
        for path_key, path in h107["path_results"].items()
    }
    family_counts = Counter(
        h107["path_results"][key]["family"] for key in models
    )
    patches = {
        "trace": trace_patch_audit(config),
        "capacity": capacity_patch_audit(config),
    }
    regression_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["regressions"].items()
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    executable_source = "\n".join(
        (PROJECT_ROOT / config["source_layout"][name]).read_text().lower()
        for name in ("scenario_core", "compiler", "runner")
    )
    target_free = (
        "fig25_roofline_utilization" not in executable_source
        and "paper_targets" not in executable_source
        and "residual_scale" not in executable_source
        and "family_correction" not in executable_source
        and compiled["paper_performance_targets_consumed"] is False
        and run["paper_performance_targets_consumed"] is False
    )
    counts = {
        "paths": len(models) == int(config["execution"]["required_paths"]),
        "configs": len(compiled["outputs"])
        == int(config["execution"]["required_configs"]),
        "records": len(records) == int(config["execution"]["required_executions"]),
        "holdouts": len(holdout_errors)
        == int(config["execution"]["required_cycle_holdouts"]),
        "families": dict(family_counts) == {"fft": 8, "qkv_bsmm": 24, "swa": 16},
    }
    acceptance_gates = [
        all(item["pass"] for item in frozen.values())
        and all(parent_checks.values())
        and h110_compile_file["pass"]
        and h110_run_file["pass"],
        all(compile_checks.values()) and all(semantic_checks.values()),
        all(
            item["metadata"]["checks"]["fma_scale"]
            and item["metadata"]["checks"]["read_scale"]
            and item["metadata"]["checks"]["write_scale"]
            and item["metadata"]["checks"]["oi"]
            for item in compiled["outputs"].values()
        ),
        min(item["metadata"]["tile_count"] for item in compiled["outputs"].values())
        == 1
        and max(
            item["metadata"]["tile_count"] for item in compiled["outputs"].values()
        )
        == 24
        and all(
            item["metadata"]["checks"]["capacity"]
            and item["metadata"]["checks"]["alignment"]
            and item["metadata"]["checks"]["store_divisibility"]
            for item in compiled["outputs"].values()
        ),
        all(
            check["done"]
            and check["instructions"]
            and check["pipelines"]
            and check["requests"]
            and check["adapter_split"]
            and check["bytes"]
            and check["tiles"]
            for check in execution_checks.values()
        ),
        all(
            check["mode"] and check["contexts"] and check["ownership"]
            for check in execution_checks.values()
        ),
        all(check["trace_off"] for check in execution_checks.values())
        and all(item["pass"] for item in run["regressions"].values()),
        all(value["cycle_slowdown_vs_h110"] >= 1 for value in measurements.values())
        and all(run["replay_checks"].values()),
        all(error <= limit for error in holdout_errors),
        all(reconstruction_checks.values())
        and all(item["eligible"] for item in full_estimates.values()),
        all(run["sanitizer_checks"].values())
        and all(item["pass"] for item in patches.values())
        and all(item["pass"] for item in regression_files.values()),
        target_free and all(check["target_free"] for check in execution_checks.values()),
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "parent_manifests": h110_compile_file["pass"] and h110_run_file["pass"],
        "compile_manifest": compile_file["pass"],
        "run_manifest": run_file["pass"] and all(run["checks"].values()),
        "compile": all(compile_checks.values()),
        "semantics": all(semantic_checks.values()),
        "records": all(record_checks.values()),
        "execution_evaluated": all(
            all(check.values()) for check in execution_checks.values()
        ),
        "counts": all(counts.values()),
        "reconstruction": all(reconstruction_checks.values()),
        "patches": all(item["pass"] for item in patches.values()),
        "regressions": all(item["pass"] for item in regression_files.values())
        and all(item["pass"] for item in run["regressions"].values()),
        "source_files": all(item["pass"] for item in source_files.values()),
        "target_free": target_free,
        "acceptance_evaluated": len(acceptance_gates) == 12
        and all(isinstance(value, bool) for value in acceptance_gates),
    }
    integrity = all(integrity_checks.values())
    supported = integrity and all(acceptance_gates)
    passing_holdouts = sum(error <= limit for error in holdout_errors)
    eligible_paths = sum(item["eligible"] for item in full_estimates.values())
    slowdowns = [item["cycle_slowdown_vs_h110"] for item in measurements.values()]
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
        "paper_reproduction_claim": "none_target_free_coupled_full_paths_only",
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "compile_manifest": compile_file,
        "run_manifest": run_file,
        "compile_checks": compile_checks,
        "semantic_checks": semantic_checks,
        "record_checks": record_checks,
        "execution_checks": execution_checks,
        "measurements": measurements,
        "models": models,
        "full_estimates": full_estimates,
        "reconstruction_checks": reconstruction_checks,
        "counts": counts,
        "patch_checks": patches,
        "regression_files": regression_files,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "paths": len(models),
            "configs": len(compiled["outputs"]),
            "executions": len(records),
            "sanitizer_executions": sum(
                key[1] in {"asan", "ubsan"} for key in records
            ),
            "cycle_holdouts_passed": passing_holdouts,
            "cycle_holdouts_total": len(holdout_errors),
            "cycle_mape": sum(holdout_errors) / len(holdout_errors),
            "cycle_max_error": max(holdout_errors),
            "eligible_full_paths": eligible_paths,
            "coupled_slowdown_min": min(slowdowns),
            "coupled_slowdown_max": max(slowdowns),
            "tile_count_min": min(
                item["metadata"]["tile_count"] for item in compiled["outputs"].values()
            ),
            "tile_count_max": max(
                item["metadata"]["tile_count"] for item in compiled["outputs"].values()
            ),
            "acceptance_gates_passed": sum(acceptance_gates),
            "acceptance_gates_total": len(acceptance_gates),
            "full_paper_rows_reproduced": 0,
            "full_paper_rows_total": 18,
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
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report = build_audit(config)
    output = PROJECT_ROOT / config["result_path"]
    if args.verify_existing:
        existing = json.loads(output.read_text())
        keys = (
            "hypothesis_status",
            "audit_integrity",
            "models",
            "full_estimates",
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
