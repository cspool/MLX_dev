#!/usr/bin/env python3
"""Compile paired single-layer-baseline and MLX complete-block prefixes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from scripts.compile_attention_functional import output_address as attention_output_address
from scripts.compile_complete_block_functional import (
    complete_block_document,
)
from scripts.compile_elementwise_functional import output_address as elementwise_output_address
from scripts.compile_fft_cmp_functional import output_address as fft_output_address
from scripts.compile_fft_cmp_functional import schedule_counts
from scripts.compile_swa_functional import output_address as swa_output_address

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/simulators/single_baseline_complete_block_v1.yaml"
)


def digest(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def output_addresses(component: str) -> list[int]:
    if component == "fft_cmp":
        return [
            fft_output_address(row, dimension)
            for row in range(2)
            for dimension in range(2)
        ]
    if component == "attention":
        return [
            attention_output_address(row, dimension)
            for row in range(2)
            for dimension in range(2)
        ]
    if component == "swa":
        return [
            swa_output_address(row, dimension)
            for row in range(4)
            for dimension in range(2)
        ]
    if component == "elementwise":
        return [elementwise_output_address(index) for index in range(8)]
    raise ValueError(f"unsupported final component: {component}")


def truncate_prefix(
    document: dict[str, Any],
    *,
    prefix: str,
    final_component: str,
    required_components: list[str],
    architecture: str,
    active_window: int,
) -> dict[str, Any]:
    result = copy.deepcopy(document)
    components = result["metadata"]["components"]
    component_names = [item["name"] for item in components]
    if required_components != component_names[: len(required_components)]:
        raise ValueError(f"prefix {prefix} is not cumulative")
    matching = [item for item in components if item["name"] == final_component]
    if len(matching) != 1:
        raise ValueError(f"missing final component: {final_component}")
    final_tag = int(matching[0]["tag_range"][1])
    result["blocks"] = [
        block for block in result["blocks"] if int(block["tag"]) <= final_tag
    ]
    result["functional_execution"]["registers"] = [
        seed
        for seed in result["functional_execution"]["registers"]
        if int(seed["tag"]) <= final_tag
    ]
    result["active_window"] = int(active_window)
    result["metadata"]["experiment_id"] = "H170"
    result["metadata"]["prefix"] = prefix
    result["metadata"]["architecture"] = architecture
    result["metadata"]["active_window"] = int(active_window)
    result["metadata"]["components"] = components[: len(required_components)]
    result["metadata"]["dynamic_link_count"] = len(required_components) - 1
    result["metadata"]["output_addresses"] = output_addresses(final_component)
    result["metadata"]["schedule_counts"] = schedule_counts(result)
    return result


def build_documents(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    h161_config = yaml.safe_load(
        (PROJECT_ROOT / config["frozen_inputs"]["h161_config"]["path"]).read_text()
    )
    documents: dict[str, dict[str, Any]] = {}
    for functional_mode, enabled in (("enabled", True), ("disabled", False)):
        source = complete_block_document(h161_config, enabled=enabled)
        for prefix, prefix_spec in config["prefixes"].items():
            for architecture_name, architecture_spec in (
                (config["baseline"]["id"], config["baseline"]),
                (config["mlx"]["id"], config["mlx"]),
            ):
                key = f"{prefix}--{architecture_name}--{functional_mode}"
                documents[key] = truncate_prefix(
                    source,
                    prefix=prefix,
                    final_component=prefix_spec["final_component"],
                    required_components=prefix_spec["required_components"],
                    architecture=architecture_name,
                    active_window=int(architecture_spec["active_window"]),
                )
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
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "paper_performance_targets_consumed": False,
        "outputs": outputs,
        "checks": {
            "count": len(outputs) == int(config["execution"]["expected_configs"]),
            "deterministic": all(item["deterministic"] for item in outputs.values()),
            "target_free": all(
                item["metadata"]["paper_performance_targets_consumed"] is False
                for item in outputs.values()
            ),
        },
    }
    path = output_root / "single-baseline-complete-compile-manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"outputs": len(outputs), "checks": manifest["checks"]}, indent=2))
    return 0 if all(manifest["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
