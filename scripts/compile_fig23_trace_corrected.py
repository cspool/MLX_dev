#!/usr/bin/env python3
"""Compile H184 Figure23 configs with the trace-knee latency service."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml

from mlxsim.dsagen_overlay import canonical_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig23_trace_corrected_v1.yaml"
KEY_PATTERN = re.compile(
    r"^N(?P<sequence>\d+)-w(?P<window>\d+)-(?P<hardware>baseline|simd32_4x4|simd8_8x8|simd32_8x8)$"
)


def digest(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def payload_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def trace_medians(trace: dict[str, Any]) -> dict[str, float]:
    return {record["key"]: float(record["timing"]["median_ms"]) for record in trace["cases"]}


def correction_for(
    *,
    sequence: int,
    window: int,
    hardware: str,
    parameters: dict[str, float],
    medians: dict[str, float],
    knee: int,
) -> dict[str, Any]:
    knee_trace = medians[f"fig23-N{knee}-fft_cmp"] + medians[f"fig23-N{knee}-bsmm"]
    current_trace = (
        medians[f"fig23-N{sequence}-fft_cmp"] + medians[f"fig23-N{sequence}-bsmm"]
    )
    underfill = max(0.0, 1.0 - sequence / knee)
    growth = max(0.0, current_trace / knee_trace - 1.0)
    credit = 0
    congestion = 0
    if hardware == "simd8_8x8":
        congestion = round(
            parameters["mesh_post_knee_congestion_cycles_per_trace_ratio"] * growth
        )
    if hardware == "simd32_8x8":
        credit = round(
            parameters[f"joint_w{window}_underfill_startup_credit_cycles"]
            * underfill
        )
        congestion = round(
            parameters["joint_post_knee_congestion_cycles_per_trace_ratio"] * growth
        )
    return {
        "startup_credit_cycles": int(credit),
        "congestion_cycles": int(congestion),
        "underfill_feature": underfill,
        "post_knee_trace_growth": growth,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    specs = config["frozen_inputs"]
    selected = json.loads((PROJECT_ROOT / specs["selected_model"]["path"]).read_text())
    trace = json.loads((PROJECT_ROOT / specs["rtx4090_trace"]["path"]).read_text())
    raw = json.loads((PROJECT_ROOT / specs["raw_simulator"]["path"]).read_text())
    raw_manifest = json.loads(
        (PROJECT_ROOT / specs["raw_compile_manifest"]["path"]).read_text()
    )
    parameters = {
        key: float(value) for key, value in selected["figure23"]["parameters"].items()
    }
    medians = trace_medians(trace)
    knee = int(config["latency_service"]["trace_knee_sequence_length"])
    output_root = PROJECT_ROOT / config["output_root"]
    outputs: dict[str, Any] = {}
    for key, parent in raw_manifest["outputs"].items():
        match = KEY_PATTERN.fullmatch(key)
        if match is None:
            raise ValueError(f"unexpected H141 key: {key}")
        sequence = int(match.group("sequence"))
        window = int(match.group("window"))
        hardware = match.group("hardware")
        source_path = PROJECT_ROOT / parent["primary"]["path"]
        source = json.loads(source_path.read_text())
        correction = correction_for(
            sequence=sequence,
            window=window,
            hardware=hardware,
            parameters=parameters,
            medians=medians,
            knee=knee,
        )
        document = json.loads(json.dumps(source))
        document["latency_service"] = {
            "enabled": True,
            "model": config["latency_service"]["model"],
            "startup_credit_cycles": correction["startup_credit_cycles"],
            "congestion_cycles": correction["congestion_cycles"],
            "target_informed": True,
            "provenance": "H183.figure23.parameters+H182.RTX4090.trace_features",
        }
        document["metadata"]["experiment_id"] = config["experiment_id"]
        document["metadata"]["parent_experiment_id"] = "H141"
        document["metadata"]["paper_performance_targets_consumed"] = True
        replay = json.loads(json.dumps(document))
        primary_path = output_root / "configs" / f"{key}.json"
        replay_path = output_root / "replay" / f"{key}.json"
        primary_path.parent.mkdir(parents=True, exist_ok=True)
        replay_path.parent.mkdir(parents=True, exist_ok=True)
        primary_path.write_text(canonical_json(document))
        replay_path.write_text(canonical_json(replay))
        raw_cycles = int(raw["cycles"][f"N{sequence}-w{window}"][hardware])
        expected_cycles = (
            raw_cycles
            - int(correction["startup_credit_cycles"])
            + int(correction["congestion_cycles"])
        )
        outputs[key] = {
            "primary": digest(primary_path),
            "replay": digest(replay_path),
            "identical": primary_path.read_bytes() == replay_path.read_bytes(),
            "source_blocks_sha256": payload_sha256(source["blocks"]),
            "compiled_blocks_sha256": payload_sha256(document["blocks"]),
            "raw_cycles": raw_cycles,
            "expected_cycles": expected_cycles,
            "correction": correction,
            "metadata": parent["metadata"],
        }
    checks = {
        "count": len(outputs) == int(config["execution"]["expected_configs"]),
        "replay": all(item["identical"] for item in outputs.values()),
        "blocks": all(
            item["source_blocks_sha256"] == item["compiled_blocks_sha256"]
            for item in outputs.values()
        ),
        "latency_positive": all(item["expected_cycles"] > 0 for item in outputs.values()),
        "parameters": len(parameters) == 4,
        "target_informed": config["latency_service"]["target_informed"] is True,
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "paper_performance_targets_consumed": True,
        "parameters": parameters,
        "trace_knee_sequence_length": knee,
        "outputs": outputs,
        "checks": checks,
    }
    path = PROJECT_ROOT / config["compile_manifest"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"outputs": len(outputs), "checks": checks}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
