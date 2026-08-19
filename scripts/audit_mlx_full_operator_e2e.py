#!/usr/bin/env python3
"""Audit H175's full-operator MLX end-to-end numerical execution."""

from __future__ import annotations

import argparse
import copy
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_attention_functional import numpy_reference as attention_reference
from scripts.audit_bsmm_functional import numpy_golden as bsmm_golden
from scripts.audit_complete_block_functional import (
    boundary_addresses,
    fft_compress,
    without_functional,
)
from scripts.audit_elementwise_functional import numpy_reference as elementwise_reference
from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify
from scripts.audit_swa_functional import numpy_reference as swa_reference
from scripts.compile_bsmm_functional import input_address as bsmm_input_address
from scripts.compile_mlx_full_operator_e2e import (
    build_document,
    normalized_address,
    raw_address,
)

DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/simulators/mlx_full_operator_e2e_functional_v1.yaml"
)


def load_h161_config(config: dict[str, Any]) -> dict[str, Any]:
    h171_config = yaml.safe_load(
        (PROJECT_ROOT / config["frozen_inputs"]["h171_config"]["path"]).read_text()
    )
    h170_config = yaml.safe_load(
        (
            PROJECT_ROOT
            / h171_config["frozen_inputs"]["h170_config"]["path"]
        ).read_text()
    )
    return yaml.safe_load(
        (
            PROJECT_ROOT
            / h170_config["frozen_inputs"]["h161_config"]["path"]
        ).read_text()
    )


def preprocess_reference(
    config: dict[str, Any], h161_config: dict[str, Any]
) -> tuple[list[list[float]], list[list[float]]]:
    inputs = h161_config["components"][0]
    bsmm_config = yaml.safe_load((PROJECT_ROOT / inputs["config"]).read_text())
    vectors = bsmm_config["operator_contract"]["inputs"]
    normalized: list[list[float]] = []
    rotated: list[list[float]] = []
    width = int(config["preprocess"]["width"])
    for batch, vector_values in enumerate(vectors):
        vector = [float(value) for value in vector_values]
        inverse = 1.0 / math.sqrt(
            sum(value * value for value in vector) / width
            + float(config["preprocess"]["epsilon"])
        )
        norm = [value * inverse for value in vector]
        rope: list[float] = []
        for pair in range(width // 2):
            first, second = norm[2 * pair], norm[2 * pair + 1]
            angle = (
                float(config["preprocess"]["rope_angle_scale"])
                * (batch + 1)
                * (pair + 1)
            )
            cosine, sine = math.cos(angle), math.sin(angle)
            rope.extend(
                (
                    first * cosine - second * sine,
                    first * sine + second * cosine,
                )
            )
        normalized.append(norm)
        rotated.append(rope)
    return normalized, rotated


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    parents = {
        name: json.loads((PROJECT_ROOT / spec["path"]).read_text())
        for name, spec in config["frozen_inputs"].items()
        if "required_status" in spec
    }
    parent_checks = {
        name: parent["hypothesis_status"]
        == config["frozen_inputs"][name]["required_status"]
        and parent["audit_integrity"]
        is config["frozen_inputs"][name]["required_integrity"]
        for name, parent in parents.items()
    }
    output_root = PROJECT_ROOT / config["output_root"]
    compile_path = output_root / "mlx-full-operator-compile-manifest.json"
    run_path = output_root / "mlx-full-operator-run-manifest.json"
    compile_file = qualify(compile_path)
    run_file = qualify(run_path)
    compiler = json.loads(compile_path.read_text())
    run = json.loads(run_path.read_text())
    rebuilt = {
        mode: build_document(config, enabled=mode == "enabled")
        for mode in config["execution"]["functional_modes"]
    }
    compile_checks: dict[str, bool] = {}
    for mode, item in compiler["outputs"].items():
        path = PROJECT_ROOT / item["artifact"]["path"]
        compile_checks[mode] = (
            qualify(path, item["artifact"])["pass"]
            and json.loads(path.read_text()) == rebuilt[mode]
            and item["deterministic"] is True
            and item["schedule_counts"]
            == rebuilt[mode]["metadata"]["schedule_counts"]
        )
    enabled = rebuilt["enabled"]
    disabled = rebuilt["disabled"]
    enabled_mode = copy.deepcopy(enabled)
    enabled_mode["functional_execution"]["enabled"] = False
    enabled_mode["metadata"]["functional_enabled"] = False
    mode_checks = {
        "only_functional_mode": enabled_mode == disabled,
        "active_window": enabled["active_window"]
        == int(config["composition"]["active_window"]),
        "mesh": [
            enabled["routing"]["mesh_width"],
            enabled["routing"]["mesh_height"],
        ]
        == config["composition"]["mesh"],
    }
    tags = sorted({int(block["tag"]) for block in enabled["blocks"]})
    component_names = [item["name"] for item in enabled["metadata"]["components"]]
    graph_checks = {
        "tags": tags == list(range(1, int(config["composition"]["expected_tags"]) + 1)),
        "blocks": len(enabled["blocks"])
        == int(config["composition"]["expected_blocks"]),
        "components": component_names
        == ["rmsnorm_rope", "bsmm", "fft_cmp", "attention", "swa", "elementwise"],
        "operators": all(
            name in enabled["metadata"]["schedule_counts"]["operations"]
            for name in ("frsqrt", "shuffle", "mul", "fma", "fmax", "fexp", "fdiv")
        ),
        "predecessor": all(
            2 in block["predecessors"]
            for block in enabled["blocks"]
            if int(block["tag"]) == 3
        ),
    }
    memory_addresses = {
        int(item["address"]) for item in enabled["functional_execution"]["memory"]
    }
    raw_addresses = {
        raw_address(config, batch, index)
        for batch in range(int(config["preprocess"]["batches"]))
        for index in range(int(config["preprocess"]["width"]))
    }
    original_addresses = {
        bsmm_input_address(batch, index)
        for batch in range(int(config["preprocess"]["batches"]))
        for index in range(int(config["preprocess"]["width"]))
    }
    store_addresses = {
        int(instruction["memory_address"])
        for block in enabled["blocks"]
        if int(block["tag"]) == 2
        for instruction in block["instructions"]
        if instruction["pipeline"] == "store"
    }
    memory_checks = {
        "raw_seeded": raw_addresses <= memory_addresses,
        "original_unseeded": not (original_addresses & memory_addresses),
        "rope_stores": store_addresses == original_addresses,
    }
    counts = enabled["metadata"]["schedule_counts"]
    static_checks = {
        "operations": counts["functional_operations"]
        == int(config["composition"]["expected_operations"]),
        "memory": counts["memory_requests"]
        == int(config["composition"]["expected_memory_requests"])
        and counts["memory_bytes"] == int(config["composition"]["expected_memory_bytes"]),
        "events": counts["boundary_events"]
        == int(config["composition"]["expected_boundary_events"]),
        "routes": counts["route_hops"]
        == int(config["composition"]["expected_route_hops"]),
        "added_operations": counts["functional_operations"]
        - int(parents["h171"]["summary"]["complete_functional_operations"])
        == int(config["preprocess"]["added_functional_operations"]),
        "added_memory": counts["memory_requests"]
        - int(parents["h171"]["complete_checks"]["memory"] is True) * 162
        == int(config["preprocess"]["added_memory_requests"]),
    }
    run_checks = {
        "experiment": run["experiment_id"] == config["experiment_id"],
        "target_free": run["paper_performance_targets_consumed"] is False,
        "modes": set(run["records"]) == {"enabled", "disabled"},
        "count": sum(len(builds) for builds in run["records"].values())
        == int(config["execution"]["expected_runs"]),
        "checks": all(run["checks"].values()),
    }
    execution_checks: dict[str, bool] = {}
    for mode, builds in run["records"].items():
        execution_checks[mode] = (
            set(builds) == set(config["execution"]["builds"])
            and all(item["pass"] and item["stderr_bytes"] == 0 for item in builds.values())
            and all(
                item["summary"]["done"] is True
                and item["summary"]["instructions_issued"]
                == item["summary"]["instructions_completed"]
                == int(config["composition"]["expected_operations"])
                and item["summary"]["boundary_events_emitted"]
                == int(config["composition"]["expected_boundary_events"])
                and item["summary"]["route_hops"]
                == int(config["composition"]["expected_route_hops"])
                for item in builds.values()
            )
        )
    summary = run["records"]["enabled"]["opt"]["summary"]
    disabled_summary = run["records"]["disabled"]["opt"]["summary"]
    functional = summary["functional"]
    h161_config = load_h161_config(config)
    normalized, rotated = preprocess_reference(config, h161_config)
    reference_config = copy.deepcopy(h161_config)
    bsmm_config_path = PROJECT_ROOT / reference_config["components"][0]["config"]
    bsmm_config = yaml.safe_load(bsmm_config_path.read_text())
    bsmm_config["operator_contract"]["inputs"] = rotated
    temporary_config = copy.deepcopy(reference_config)
    # full_chain_reference loads the component file, so reproduce its steps by
    # temporarily using a generated in-memory-compatible reference below.
    bsmm_values, _, _ = bsmm_golden(bsmm_config)
    component_configs = {
        item["name"]: yaml.safe_load((PROJECT_ROOT / item["config"]).read_text())
        for item in temporary_config["components"]
    }
    fft_values = fft_compress(bsmm_values, component_configs["fft_cmp"])
    attention = attention_reference(
        component_configs["attention"], {"actual_outputs": fft_values}
    )
    attention_values = [value for row in attention["output"] for value in row]
    swa = swa_reference(
        component_configs["swa"], {"actual_outputs": attention_values}
    )
    swa_values = [value for row in swa["output"] for value in row]
    elementwise = elementwise_reference(
        component_configs["elementwise"], {"actual_outputs": swa_values}
    )
    final_values = [value for row in elementwise["output"] for value in row]
    references = {
        "bsmm": bsmm_values,
        "fft_cmp": fft_values,
        "attention": attention_values,
        "swa": swa_values,
        "elementwise": final_values,
    }
    addresses = boundary_addresses()
    actual_normalized = [
        float(functional["memory"][str(normalized_address(config, batch, index))])
        for batch in range(len(normalized))
        for index in range(len(normalized[batch]))
    ]
    actual_rotated = [
        float(functional["memory"][str(bsmm_input_address(batch, index))])
        for batch in range(len(rotated))
        for index in range(len(rotated[batch]))
    ]
    expected_normalized = [value for vector in normalized for value in vector]
    expected_rotated = [value for vector in rotated for value in vector]
    boundary_errors = {
        "rmsnorm": [
            abs(actual - expected)
            for actual, expected in zip(
                actual_normalized, expected_normalized, strict=True
            )
        ],
        "rope": [
            abs(actual - expected)
            for actual, expected in zip(actual_rotated, expected_rotated, strict=True)
        ],
        **{
            name: [
                abs(float(functional["memory"][str(address)]) - expected)
                for address, expected in zip(addresses[name], values, strict=True)
            ]
            for name, values in references.items()
        },
    }
    maximum_error = max(error for values in boundary_errors.values() for error in values)
    numeric_checks = {
        "enabled": functional["enabled"] is True,
        "operations": functional["operations"]
        == int(config["composition"]["expected_operations"]),
        "finite": functional["nan_values"] == 0 and functional["errors"] == 0,
        "boundaries": maximum_error
        <= float(config["execution"]["maximum_absolute_error"]),
        "outputs": len(references["elementwise"])
        == int(config["composition"]["expected_outputs"]),
        "timing_identity": without_functional(summary)
        == without_functional(disabled_summary),
    }
    inventory_checks = {
        "mlx_groups": len(config["composition"]["operator_groups"]) == 7,
        "xavier_groups": parents["xavier_functional"]["summary"][
            "operator_groups"
        ]
        == 11,
        "performance": parents["performance_estimate"]["summary"][
            "paper_informed_estimate_complete"
        ]
        is True,
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    target_free_check = (
        run["paper_performance_targets_consumed"] is False
        and config["execution"]["paper_performance_targets_consumed"] is False
    )
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        all(compile_checks.values()) and all(mode_checks.values()),
        all(memory_checks.values()),
        graph_checks["operators"],
        all(static_checks.values()),
        all(run_checks.values()) and all(execution_checks.values()),
        numeric_checks["boundaries"]
        and numeric_checks["outputs"]
        and numeric_checks["timing_identity"],
        numeric_checks["operations"] and numeric_checks["finite"],
        all(inventory_checks.values()),
        target_free_check and all(item["pass"] for item in source_files.values()),
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "generated": compile_file["pass"] and run_file["pass"],
        "compile": all(compile_checks.values()),
        "mode": all(mode_checks.values()),
        "graph": all(graph_checks.values()),
        "memory": all(memory_checks.values()),
        "static": all(static_checks.values()),
        "run": all(run_checks.values()) and all(execution_checks.values()),
        "numeric": all(numeric_checks.values()),
        "inventory": all(inventory_checks.values()),
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
        "paper_reproduction_claim": "none_full_operator_functional_only",
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "generated_inputs": {"compile": compile_file, "run": run_file},
        "compile_checks": compile_checks,
        "mode_checks": mode_checks,
        "graph_checks": graph_checks,
        "memory_checks": memory_checks,
        "static_checks": static_checks,
        "run_checks": run_checks,
        "execution_checks": execution_checks,
        "boundary_errors": boundary_errors,
        "numeric_checks": numeric_checks,
        "inventory_checks": inventory_checks,
        "source_files": source_files,
        "target_free_check": target_free_check,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "operator_groups": len(config["composition"]["operator_groups"]),
            "tags": len(tags),
            "blocks": len(enabled["blocks"]),
            "functional_operations": functional["operations"],
            "memory_requests": counts["memory_requests"],
            "memory_bytes": counts["memory_bytes"],
            "boundary_events": counts["boundary_events"],
            "route_hops": counts["route_hops"],
            "cycles": summary["cycles"],
            "maximum_absolute_error": maximum_error,
            "outputs": len(references["elementwise"]),
            "enabled_disabled_timing_identical": numeric_checks[
                "timing_identity"
            ],
            "mlx_full_operator_functional_complete": supported,
            "xavier_full_operator_functional_complete": inventory_checks[
                "xavier_groups"
            ],
            "paper_aligned_performance_estimate_complete": inventory_checks[
                "performance"
            ],
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
            "graph_checks",
            "memory_checks",
            "static_checks",
            "boundary_errors",
            "numeric_checks",
            "inventory_checks",
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
