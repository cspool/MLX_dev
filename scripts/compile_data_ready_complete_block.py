#!/usr/bin/env python3
"""Compile H171 address-ready MLX and coarse-barrier baseline documents."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml

from scripts.compile_complete_block_functional import link_address_maps
from scripts.compile_fft_cmp_functional import schedule_counts
from scripts.compile_single_baseline_complete_block import (
    build_documents as build_h170_documents,
)
from scripts.compile_single_baseline_complete_block import digest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/data_ready_complete_block_v1.yaml"


def instruction_addresses(instruction: dict[str, Any], trip_count: int) -> list[int]:
    sequence = instruction.get("memory_address_sequence")
    if sequence is not None:
        addresses = [int(value) for value in sequence]
        if len(addresses) != trip_count:
            raise ValueError("memory address sequence does not match trip count")
        return addresses
    return [int(instruction["memory_address"])] * trip_count


def event_name(instruction_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_]+", "_", instruction_id)
    return f"h171_data_ready__{normalized}"


def add_data_ready_events(document: dict[str, Any]) -> dict[str, int]:
    components = [item["name"] for item in document["metadata"]["components"]]
    mappings = link_address_maps()
    linked_addresses = {
        int(upstream)
        for component in components
        if component != "bsmm"
        for upstream in mappings[component].values()
    }
    stores: list[tuple[dict[str, Any], dict[str, Any], list[int]]] = []
    for block in document["blocks"]:
        for instruction in block["instructions"]:
            if instruction["pipeline"] != "store":
                continue
            addresses = instruction_addresses(instruction, int(block["trip_count"]))
            if linked_addresses.intersection(addresses):
                stores.append((block, instruction, addresses))
    address_ready: dict[int, tuple[str, int]] = {}
    event_emissions = 0
    for block, instruction, addresses in stores:
        if instruction.get("emit_event"):
            raise ValueError("linked store already has an event")
        event = event_name(instruction["id"])
        instruction["emit_event"] = event
        event_emissions += int(block["trip_count"])
        for occurrence, address in enumerate(addresses, start=1):
            if address not in linked_addresses:
                continue
            if address in address_ready:
                raise ValueError(f"duplicate linked store address: {address}")
            address_ready[address] = (event, occurrence)
    if set(address_ready) != linked_addresses:
        missing = sorted(linked_addresses - set(address_ready))
        raise ValueError(f"linked addresses without store events: {missing}")

    subscriptions = 0
    for block in document["blocks"]:
        trip_count = int(block["trip_count"])
        requirements: dict[str, dict[int, int]] = {}
        for instruction in block["instructions"]:
            if instruction["pipeline"] != "load":
                continue
            for iteration, address in enumerate(
                instruction_addresses(instruction, trip_count)
            ):
                if address not in address_ready:
                    continue
                event, occurrence = address_ready[address]
                requirements.setdefault(event, {})[iteration] = max(
                    occurrence,
                    requirements.setdefault(event, {}).get(iteration, 0),
                )
        for event, by_iteration in requirements.items():
            if set(by_iteration) != set(range(trip_count)):
                raise ValueError(
                    f"event {event} does not cover every iteration of {block['id']}"
                )
            multiplicity = by_iteration[0]
            if any(
                by_iteration[iteration] != (iteration + 1) * multiplicity
                for iteration in range(trip_count)
            ):
                raise ValueError(
                    f"event {event} cannot be encoded for block {block['id']}"
                )
            block.setdefault("wait_events", [])
            if event not in block["wait_events"]:
                block["wait_events"].append(event)
                block["wait_events"].sort()
                subscriptions += 1
            if multiplicity != 1:
                block.setdefault("wait_event_multiplicities", {})[event] = multiplicity
    return {
        "linked_addresses": len(linked_addresses),
        "event_definitions": len(stores),
        "event_emissions": event_emissions,
        "subscriptions": subscriptions,
    }


def remove_coarse_component_predecessors(document: dict[str, Any]) -> int:
    components = document["metadata"]["components"]
    removed = 0
    for previous, current in pairwise(components):
        previous_final = int(previous["tag_range"][1])
        current_first = int(current["tag_range"][0])
        for block in document["blocks"]:
            if int(block["tag"]) != current_first:
                continue
            if previous_final in block["predecessors"]:
                block["predecessors"].remove(previous_final)
                removed += 1
    return removed


def build_documents(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    h170_config = yaml.safe_load(
        (PROJECT_ROOT / config["frozen_inputs"]["h170_config"]["path"]).read_text()
    )
    source_documents = build_h170_documents(h170_config)
    baseline_id = config["baseline"]["id"]
    h170_mlx_id = h170_config["mlx"]["id"]
    mlx_id = config["mlx"]["id"]
    documents: dict[str, dict[str, Any]] = {}
    for prefix in config["prefixes"]:
        for mode in config["execution"]["functional_modes"]:
            baseline_key = f"{prefix}--{baseline_id}--{mode}"
            baseline = copy.deepcopy(source_documents[baseline_key])
            baseline_stats = add_data_ready_events(baseline)
            baseline["metadata"]["experiment_id"] = config["experiment_id"]
            baseline["metadata"]["boundary_mode"] = (
                "coarse_predecessor_plus_redundant_data_ready"
            )
            baseline["metadata"]["data_ready"] = {
                **baseline_stats,
                "coarse_predecessors_removed": 0,
            }
            baseline["metadata"]["schedule_counts"] = schedule_counts(baseline)
            documents[baseline_key] = baseline

            source_mlx_key = f"{prefix}--{h170_mlx_id}--{mode}"
            mlx = copy.deepcopy(source_documents[source_mlx_key])
            mlx_stats = add_data_ready_events(mlx)
            removed = remove_coarse_component_predecessors(mlx)
            mlx["metadata"]["experiment_id"] = config["experiment_id"]
            mlx["metadata"]["architecture"] = mlx_id
            mlx["metadata"]["boundary_mode"] = "address_matched_store_ready"
            mlx["metadata"]["data_ready"] = {
                **mlx_stats,
                "coarse_predecessors_removed": removed,
            }
            mlx["metadata"]["schedule_counts"] = schedule_counts(mlx)
            documents[f"{prefix}--{mlx_id}--{mode}"] = mlx
    return documents


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output_root = PROJECT_ROOT / config["output_root"]
    config_root = output_root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    documents = build_documents(config)
    replay = build_documents(config)
    outputs: dict[str, Any] = {}
    for key, document in documents.items():
        path = config_root / f"{key}.json"
        path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        outputs[key] = {
            "artifact": digest(path),
            "deterministic": document == replay[key],
            "metadata": document["metadata"],
            "schedule_counts": document["metadata"]["schedule_counts"],
            "input_memory_sha256": hashlib.sha256(
                json.dumps(
                    document["functional_execution"]["memory"],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        }
    full_baseline = outputs[f"complete--{config['baseline']['id']}--enabled"]
    full_mlx = outputs[f"complete--{config['mlx']['id']}--enabled"]
    contract = config["boundary_contract"]
    checks = {
        "count": len(outputs) == int(config["execution"]["expected_configs"]),
        "deterministic": all(item["deterministic"] for item in outputs.values()),
        "target_free": all(
            item["metadata"]["paper_performance_targets_consumed"] is False
            for item in outputs.values()
        ),
        "full_event_definitions": full_mlx["metadata"]["data_ready"][
            "event_definitions"
        ]
        == int(contract["store_event_definitions"]),
        "full_event_emissions": full_mlx["metadata"]["data_ready"][
            "event_emissions"
        ]
        == int(contract["store_event_emissions"]),
        "full_boundary_events": full_mlx["schedule_counts"]["boundary_events"]
        == full_baseline["schedule_counts"]["boundary_events"]
        == int(contract["expected_boundary_events"]),
        "removed_predecessors": full_mlx["metadata"]["data_ready"][
            "coarse_predecessors_removed"
        ]
        == int(contract["mlx_removed_coarse_predecessors"])
        and full_baseline["metadata"]["data_ready"][
            "coarse_predecessors_removed"
        ]
        == int(contract["baseline_removed_coarse_predecessors"]),
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "paper_performance_targets_consumed": False,
        "outputs": outputs,
        "checks": checks,
    }
    path = output_root / "data-ready-complete-compile-manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"outputs": len(outputs), "checks": checks}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
