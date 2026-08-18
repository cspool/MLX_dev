#!/usr/bin/env python3
"""Compile H161 one-execution complete Transformer block configs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from scripts.compile_attention_functional import (
    attention_document,
)
from scripts.compile_attention_functional import (
    output_address as attention_output_address,
)
from scripts.compile_attention_functional import (
    q_address as attention_q_address,
)
from scripts.compile_bsmm_functional import (
    bsmm_document,
)
from scripts.compile_bsmm_functional import (
    output_address as bsmm_output_address,
)
from scripts.compile_elementwise_functional import (
    elementwise_document,
)
from scripts.compile_elementwise_functional import (
    input_address as elementwise_input_address,
)
from scripts.compile_elementwise_functional import (
    output_address as elementwise_output_address,
)
from scripts.compile_fft_cmp_functional import (
    fft_cmp_document,
    schedule_counts,
)
from scripts.compile_fft_cmp_functional import (
    input_address as fft_input_address,
)
from scripts.compile_fft_cmp_functional import (
    output_address as fft_output_address,
)
from scripts.compile_swa_functional import (
    output_address as swa_output_address,
)
from scripts.compile_swa_functional import (
    q_address as swa_q_address,
)
from scripts.compile_swa_functional import (
    swa_document,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/complete_block_functional_v1.yaml"

Compiler = Callable[[dict[str, Any]], dict[str, Any]]


def digest(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def component_document(name: str, config: dict[str, Any], enabled: bool) -> dict[str, Any]:
    compilers: dict[str, Callable[..., dict[str, Any]]] = {
        "bsmm": bsmm_document,
        "fft_cmp": fft_cmp_document,
        "attention": attention_document,
        "swa": swa_document,
        "elementwise": elementwise_document,
    }
    return compilers[name](config, enabled=enabled)


def link_address_maps() -> dict[str, dict[int, int]]:
    return {
        "bsmm": {},
        "fft_cmp": {
            fft_input_address(batch, index): bsmm_output_address(batch, index)
            for batch in range(2)
            for index in range(4)
        },
        "attention": {
            attention_q_address(row, dimension): fft_output_address(row, dimension)
            for row in range(2)
            for dimension in range(2)
        },
        "swa": {
            swa_q_address(row, dimension): attention_output_address(row, dimension)
            for row in range(2)
            for dimension in range(2)
        },
        "elementwise": {
            elementwise_input_address(index): swa_output_address(index // 2, index % 2)
            for index in range(8)
        },
    }


def replace_memory_addresses(document: dict[str, Any], replacements: dict[int, int]) -> None:
    document["functional_execution"]["memory"] = [
        item
        for item in document["functional_execution"]["memory"]
        if int(item["address"]) not in replacements
    ]
    for block in document["blocks"]:
        for instruction in block["instructions"]:
            address = int(instruction.get("memory_address", 0))
            if address in replacements:
                instruction["memory_address"] = replacements[address]
            if "memory_address_sequence" in instruction:
                instruction["memory_address_sequence"] = [
                    replacements.get(int(item), int(item))
                    for item in instruction["memory_address_sequence"]
                ]


def translate_document(
    document: dict[str, Any], *, tag_offset: int, x_offset: int
) -> None:
    for seed in document["functional_execution"]["registers"]:
        seed["pe"] = [int(seed["pe"][0]) + x_offset, int(seed["pe"][1])]
        seed["tag"] = int(seed["tag"]) + tag_offset
    for block in document["blocks"]:
        block["pe"] = [int(block["pe"][0]) + x_offset, int(block["pe"][1])]
        block["tag"] = int(block["tag"]) + tag_offset
        block["predecessors"] = [
            int(predecessor) + tag_offset for predecessor in block["predecessors"]
        ]
        for instruction in block["instructions"]:
            if instruction["pipeline"] == "xfer":
                instruction["destination"] = [
                    int(instruction["destination"][0]) + x_offset,
                    int(instruction["destination"][1]),
                ]
                instruction["destination_tag"] = (
                    int(instruction["destination_tag"]) + tag_offset
                )


def complete_block_document(config: dict[str, Any], *, enabled: bool) -> dict[str, Any]:
    replacements = link_address_maps()
    blocks: list[dict[str, Any]] = []
    memory: list[dict[str, Any]] = []
    registers: list[dict[str, Any]] = []
    component_metadata = []
    previous_final_tag: int | None = None
    for specification in config["components"]:
        name = specification["name"]
        component_config = yaml.safe_load(
            (PROJECT_ROOT / specification["config"]).read_text()
        )
        document = copy.deepcopy(component_document(name, component_config, enabled))
        original_counts = schedule_counts(document)
        replace_memory_addresses(document, replacements[name])
        tag_offset = int(specification["tag_offset"])
        x_offset = int(specification["x_offset"])
        translate_document(document, tag_offset=tag_offset, x_offset=x_offset)
        first_tag = tag_offset + 1
        final_tag = max(int(block["tag"]) for block in document["blocks"])
        if previous_final_tag is not None:
            for block in document["blocks"]:
                if int(block["tag"]) == first_tag:
                    block["predecessors"] = sorted(
                        {*block["predecessors"], previous_final_tag}
                    )
        component_metadata.append(
            {
                "name": name,
                "config": specification["config"],
                "tag_range": [first_tag, final_tag],
                "x_offset": x_offset,
                "source_schedule_counts": original_counts,
                "linked_seed_count": len(replacements[name]),
                "linked_address_map": {
                    str(downstream): upstream
                    for downstream, upstream in sorted(replacements[name].items())
                },
            }
        )
        previous_final_tag = final_tag
        blocks.extend(document["blocks"])
        memory.extend(document["functional_execution"]["memory"])
        registers.extend(document["functional_execution"]["registers"])
    addresses = [int(item["address"]) for item in memory]
    if len(addresses) != len(set(addresses)):
        raise ValueError("complete-block functional memory seeds must be unique")
    document = {
        "schema_version": 1,
        "active_window": int(config["composition_contract"]["active_window"]),
        "record_events": True,
        "start_in_roi": True,
        "memory_backend": "fixed",
        "pe_dependency_model": "scoreboard_experimental",
        "functional_execution": {
            "enabled": enabled,
            "strict_memory": True,
            "memory": sorted(memory, key=lambda item: int(item["address"])),
            "registers": sorted(
                registers,
                key=lambda item: (
                    int(item["tag"]),
                    int(item["pe"][1]),
                    int(item["pe"][0]),
                    int(item["iteration"]),
                    int(item["reg"]),
                ),
            ),
        },
        "register_file": {"banks": 16, "read_ports": 3, "write_ports": 4},
        "pipelines": {
            pipeline: {"latency": 1, "initiation_interval": 1}
            for pipeline in ("load", "store", "compute", "xfer")
        },
        "functional_units": {
            "add": {"class": "alu", "latency": 2, "initiation_interval": 1},
            "mul": {"class": "mul", "latency": 3, "initiation_interval": 1},
            "fma": {"class": "fma", "latency": 4, "initiation_interval": 1},
            "fmax": {"class": "reduce", "latency": 2, "initiation_interval": 1},
            "fexp": {
                "class": "transcendental",
                "latency": 8,
                "initiation_interval": 4,
            },
            "fdiv": {
                "class": "transcendental",
                "latency": 8,
                "initiation_interval": 4,
            },
        },
        "routing": {
            "mesh_width": int(config["composition_contract"]["mesh"][0]),
            "mesh_height": int(config["composition_contract"]["mesh"][1]),
            "skip_steps": config["composition_contract"]["skip_steps"],
            "latency_per_hop": 1,
            "link_capacity": 1,
        },
        "blocks": blocks,
        "metadata": {
            "experiment_id": config["experiment_id"],
            "operator_family": "complete_transformer_block",
            "paper_performance_targets_consumed": False,
            "functional_enabled": enabled,
            "components": component_metadata,
            "dynamic_link_count": sum(bool(item) for item in replacements.values()),
            "output_addresses": [elementwise_output_address(index) for index in range(8)],
        },
    }
    document["metadata"]["schedule_counts"] = schedule_counts(document)
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output_root = PROJECT_ROOT / config["output_root"]
    config_root = output_root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, enabled in (("enabled", True), ("disabled", False)):
        document = complete_block_document(config, enabled=enabled)
        replay = complete_block_document(config, enabled=enabled)
        path = config_root / f"complete-block-{name}.json"
        path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        outputs[name] = {
            "artifact": digest(path),
            "deterministic": document == replay,
            "schedule_counts": document["metadata"]["schedule_counts"],
        }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "paper_performance_targets_consumed": False,
        "outputs": outputs,
    }
    path = output_root / "complete-block-functional-compile-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if all(item["deterministic"] for item in outputs.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
