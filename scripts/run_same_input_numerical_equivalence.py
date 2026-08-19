#!/usr/bin/env python3
"""Run H189 same-input golden versus lowered numerical comparisons."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mlxsim.numerical_equivalence import MappingConfig, compare_execution, execute_graph
from mlxsim.workload_lowering import validate_suite

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/same_input_numerical_equivalence_v1.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    spec = yaml.safe_load(
        (PROJECT_ROOT / config["frozen_inputs"]["workload_spec"]["path"]).read_text()
    )
    orders = validate_suite(spec)
    contract = config["test_contract"]
    runs: list[dict[str, Any]] = []
    mapping_outputs: dict[tuple[str, int, str], list[tuple[str, np.ndarray]]] = {}
    for graph_id, graph in spec["graphs"].items():
        graph_contract = contract["graphs"][graph_id]
        for seed in contract["seeds"]:
            for precision, precision_spec in contract["precisions"].items():
                dtype = str(precision_spec["dtype"])
                golden = execute_graph(
                    graph_id=graph_id,
                    graph=graph,
                    order=orders[graph_id],
                    contract=graph_contract,
                    seed=int(seed),
                    dtype=dtype,
                    mapping=None,
                )
                group = (graph_id, int(seed), precision)
                mapping_outputs[group] = []
                for mapping_spec in contract["mappings"]:
                    mapping = MappingConfig(
                        name=str(mapping_spec["name"]),
                        simd_width=int(mapping_spec["simd_width"]),
                        mesh=tuple(int(value) for value in mapping_spec["mesh"]),
                    )
                    lowered = execute_graph(
                        graph_id=graph_id,
                        graph=graph,
                        order=orders[graph_id],
                        contract=graph_contract,
                        seed=int(seed),
                        dtype=dtype,
                        mapping=mapping,
                    )
                    comparison = compare_execution(lowered, golden)
                    absolute_limit = float(precision_spec["absolute_tolerance"])
                    relative_limit = float(precision_spec["relative_tolerance"])
                    boundary_passes = {
                        node_id: values["maximum_absolute_error"] <= absolute_limit
                        or values["maximum_relative_error"] <= relative_limit
                        for node_id, values in comparison["boundaries"].items()
                    }
                    final_pass = (
                        comparison["final_maximum_absolute_error"] <= absolute_limit
                        or comparison["final_maximum_relative_error"] <= relative_limit
                    )
                    run = {
                        "graph_id": graph_id,
                        "seed": int(seed),
                        "precision": precision,
                        "dtype": dtype,
                        "mapping": {
                            "name": mapping.name,
                            "simd_width": mapping.simd_width,
                            "mesh": list(mapping.mesh),
                            "pe_count": mapping.pe_count,
                        },
                        "topological_order": orders[graph_id],
                        "sinks": lowered["sinks"],
                        "comparison": comparison,
                        "boundary_passes": boundary_passes,
                        "final_pass": final_pass,
                        "lowered_final_sha256": lowered["final_sha256"],
                        "golden_final_sha256": golden["final_sha256"],
                        "work": {
                            "operation_counts": lowered["operation_counts"],
                            "tensor_elements": lowered["tensor_elements"],
                            "node_count": len(lowered["events"]),
                        },
                        "pass": all(boundary_passes.values())
                        and final_pass
                        and comparison["event_order_identity"]
                        and comparison["operation_count_identity"]
                        and comparison["tensor_element_identity"],
                    }
                    runs.append(run)
                    mapping_outputs[group].append((mapping.name, lowered["final"].copy()))
    invariance: list[dict[str, Any]] = []
    for (graph_id, seed, precision), outputs in mapping_outputs.items():
        reference_name, reference_output = outputs[0]
        limits = contract["precisions"][precision]
        absolute_limit = float(limits["absolute_tolerance"])
        relative_limit = float(limits["relative_tolerance"])
        for mapping_name, output in outputs[1:]:
            absolute_error = float(np.max(np.abs(output - reference_output)))
            relative_error = float(
                np.max(
                    np.abs(output - reference_output)
                    / np.maximum(np.abs(reference_output), 1.0e-8)
                )
            )
            invariance.append(
                {
                    "graph_id": graph_id,
                    "seed": seed,
                    "precision": precision,
                    "reference_mapping": reference_name,
                    "mapping": mapping_name,
                    "maximum_absolute_error": absolute_error,
                    "maximum_relative_error": relative_error,
                    "within_tolerance": absolute_error <= absolute_limit
                    or relative_error <= relative_limit,
                }
            )
    boundary_count = sum(len(run["comparison"]["boundaries"]) for run in runs)
    maximum_absolute_error = max(
        [
            values["maximum_absolute_error"]
            for run in runs
            for values in run["comparison"]["boundaries"].values()
        ]
        + [run["comparison"]["final_maximum_absolute_error"] for run in runs]
    )
    maximum_relative_error = max(
        [
            values["maximum_relative_error"]
            for run in runs
            for values in run["comparison"]["boundaries"].values()
        ]
        + [run["comparison"]["final_maximum_relative_error"] for run in runs]
    )
    checks = {
        "graphs": len(orders) == int(config["acceptance"]["required_graphs"]),
        "nodes": sum(len(order) for order in orders.values())
        == int(config["acceptance"]["required_nodes"]),
        "runs": len(runs) == int(config["acceptance"]["required_runs"]),
        "boundaries": boundary_count
        == int(config["acceptance"]["required_boundary_comparisons"]),
        "finals": len(runs) == int(config["acceptance"]["required_final_comparisons"]),
        "run_passes": all(run["pass"] for run in runs),
        "invariance": len(invariance)
        == int(config["acceptance"]["required_mapping_invariance_checks"])
        and all(item["within_tolerance"] for item in invariance),
        "finite": math.isfinite(maximum_absolute_error)
        and math.isfinite(maximum_relative_error),
        "target_free": config["acceptance"]["paper_targets_consumed"] is False,
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "paper_performance_targets_consumed": False,
        "topological_orders": orders,
        "runs": runs,
        "mapping_invariance": invariance,
        "summary": {
            "graphs": len(orders),
            "nodes": sum(len(order) for order in orders.values()),
            "runs": len(runs),
            "boundary_comparisons": boundary_count,
            "final_comparisons": len(runs),
            "mapping_invariance_checks": len(invariance),
            "maximum_absolute_error": maximum_absolute_error,
            "maximum_relative_error": maximum_relative_error,
        },
        "checks": checks,
    }
    path = PROJECT_ROOT / config["execution_manifest"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": manifest["summary"], "checks": checks}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
