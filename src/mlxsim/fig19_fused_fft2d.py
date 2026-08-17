"""Mechanism follow-up: fuse Fig. 19's two FFT axes into one event profile."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from mlxsim.fig19_components import (
    audit_fig19_component_digitization,
    load_component_manifest,
)
from mlxsim.fig19_mlx_transfer import (
    compare_mlx_transfer,
    load_transfer_config,
    mapped_workloads,
    simulate_fig19_mlx,
)
from mlxsim.schema import CalibrationConfig, HardwareConfig, KernelProfile, Workload
from mlxsim.simulator import MLXSimulator
from mlxsim.workloads import compile_workload

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs/analysis/fig19_fused_fft2d_v1.yaml"
H23_RESULT = PROJECT_ROOT / "artifacts/results/fig19-mlx-event-transfer-run027.json"


def load_fusion_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def fuse_fft2d_profile(
    base_config: dict[str, Any], sequence_length: int, fusion: dict[str, Any]
) -> tuple[Workload, KernelProfile, dict[str, Any]]:
    """Apply the frozen store/load-to-NoC transformation to two FFT profiles."""

    attention = mapped_workloads(base_config, sequence_length)["attention"]
    if len(attention) != 2:
        raise ValueError("H24 requires exactly two attention-axis workloads")
    hidden_workload, token_workload = attention
    hidden = compile_workload(hidden_workload)
    token = compile_workload(token_workload)
    hidden_stages = list(hidden.stages)
    token_stages = list(token.stages)

    removed_store_bytes = float(hidden_stages[-1].store_bytes)
    removed_load_bytes = float(token_stages[0].load_bytes)
    if removed_store_bytes <= 0 or removed_load_bytes <= 0:
        raise ValueError("expected terminal store and initial load before FFT2D fusion")
    intermediate_elements = (
        int(base_config["model"]["batch"])
        * int(sequence_length)
        * int(base_config["model"]["hidden_dim"])
    )
    handoff_bytes = intermediate_elements * int(fusion["intermediate_bytes_per_element"])
    hidden_stages[-1] = replace(
        hidden_stages[-1],
        store_bytes=0.0,
        transfer_bytes=float(hidden_stages[-1].transfer_bytes) + handoff_bytes,
    )
    token_stages[0] = replace(token_stages[0], load_bytes=0.0)
    tag_offset = max(stage.tag for stage in hidden_stages) + 1
    token_stages = [replace(stage, tag=stage.tag + tag_offset) for stage in token_stages]

    stages = tuple(hidden_stages + token_stages)
    profile = KernelProfile(
        operations=float(hidden.operations + token.operations),
        offchip_bytes=float(
            hidden.offchip_bytes
            + token.offchip_bytes
            - removed_store_bytes
            - removed_load_bytes
        ),
        output_elements=max(float(hidden.output_elements), float(token.output_elements)),
        stages=stages,
        metadata={
            "kind": "fused_fft2d",
            "hidden_axis_stage_count": len(hidden_stages),
            "token_axis_stage_count": len(token_stages),
            "stage_count": len(stages),
            "handoff_bytes": handoff_bytes,
            "intermediate_format": fusion["intermediate_format"],
            "launch_count": int(fusion["launch_count"]),
        },
    )
    anchor = replace(token_workload, name=f"fig19-fused-fft2d-N{sequence_length}")
    invariants = {
        "operations_before": float(hidden.operations + token.operations),
        "operations_after": float(profile.operations),
        "operations_preserved": profile.operations == hidden.operations + token.operations,
        "removed_store_bytes": removed_store_bytes,
        "removed_load_bytes": removed_load_bytes,
        "handoff_bytes": handoff_bytes,
        "hidden_final_store_bytes_after": float(profile.stages[len(hidden_stages) - 1].store_bytes),
        "hidden_final_transfer_bytes_after": float(
            profile.stages[len(hidden_stages) - 1].transfer_bytes
        ),
        "token_initial_load_bytes_after": float(profile.stages[len(hidden_stages)].load_bytes),
        "stage_count_before": len(hidden.stages) + len(token.stages),
        "stage_count_after": len(profile.stages),
        "tags_strictly_increasing_by_axis": (
            max(stage.tag for stage in hidden_stages)
            < min(stage.tag for stage in token_stages)
        ),
    }
    invariants["pass"] = (
        invariants["operations_preserved"]
        and invariants["hidden_final_store_bytes_after"] == 0.0
        and invariants["hidden_final_transfer_bytes_after"] == handoff_bytes
        and invariants["token_initial_load_bytes_after"] == 0.0
        and invariants["stage_count_before"] == invariants["stage_count_after"]
        and invariants["tags_strictly_increasing_by_axis"]
    )
    return anchor, profile, invariants


def simulate_fused_attention(
    fusion_config: dict[str, Any],
    base_config: dict[str, Any],
    hardware: HardwareConfig,
    calibration: CalibrationConfig,
) -> list[dict[str, Any]]:
    simulator = MLXSimulator(hardware, calibration)
    layers = int(base_config["model"]["num_layers"])
    results: list[dict[str, Any]] = []
    for length in base_config["model"]["sequence_lengths"]:
        anchor, profile, invariants = fuse_fft2d_profile(
            base_config, int(length), fusion_config["fusion"]
        )
        simulation = simulator.simulate_profile(anchor, profile).to_dict()
        results.append(
            {
                "sequence_length": int(length),
                "attention_latency_ms": float(simulation["latency_us"]) * layers / 1000.0,
                "per_layer_simulation": simulation,
                "fusion_invariants": invariants,
            }
        )
    return results


def _compare_attention(
    targets: dict[str, Any],
    fused: list[dict[str, Any]],
    isolated: list[dict[str, Any]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    target_index = {
        int(length): index for index, length in enumerate(targets["sequence_lengths"])
    }
    isolated_by_length = {int(item["sequence_length"]): item for item in isolated}
    points: list[dict[str, Any]] = []
    for item in fused:
        length = int(item["sequence_length"])
        target = float(targets["mlx"]["attention_latency_ms"][target_index[length]])
        actual = float(item["attention_latency_ms"])
        isolated_actual = float(isolated_by_length[length]["attention_latency_ms"])
        error = abs(actual - target) / target
        isolated_error = abs(isolated_actual - target) / target
        points.append(
            {
                "sequence_length": length,
                "target_latency_ms": target,
                "isolated_latency_ms": isolated_actual,
                "fused_latency_ms": actual,
                "latency_change_fraction": actual / isolated_actual - 1.0,
                "isolated_absolute_relative_error": isolated_error,
                "fused_absolute_relative_error": error,
                "absolute_error_change": error - isolated_error,
                "tolerance": tolerance,
                "pass": error <= tolerance,
            }
        )
    errors = [point["fused_absolute_relative_error"] for point in points]
    return {
        "points": points,
        "summary": {
            "point_count": len(points),
            "mape": sum(errors) / len(errors),
            "max_absolute_relative_error": max(errors),
            "all_points_pass": all(point["pass"] for point in points),
            "all_points_improve": all(point["absolute_error_change"] < 0 for point in points),
        },
    }


def _h23_replay_check(isolated: list[dict[str, Any]]) -> dict[str, Any]:
    prior = json.loads(H23_RESULT.read_text(encoding="utf-8"))["simulated"]
    prior_by_length = {int(item["sequence_length"]): item for item in prior}
    points = []
    for item in isolated:
        length = int(item["sequence_length"])
        for key in ("attention_latency_ms", "ffn_latency_ms", "total_latency_ms"):
            error = abs(float(item[key]) - float(prior_by_length[length][key]))
            points.append(
                {
                    "sequence_length": length,
                    "metric": key,
                    "absolute_error": error,
                    "pass": error <= 1e-12,
                }
            )
    return {"points": points, "pass": all(point["pass"] for point in points)}


def project_git_revision() -> str:
    return subprocess.run(
        ["git", "-c", f"safe.directory={PROJECT_ROOT}", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def run_fused_fft2d_audit(config: dict[str, Any]) -> dict[str, Any]:
    base_config = load_transfer_config(PROJECT_ROOT / config["base_transfer_config"])
    hardware = HardwareConfig.from_yaml(PROJECT_ROOT / base_config["hardware"])
    calibration = CalibrationConfig.from_yaml(PROJECT_ROOT / base_config["calibration"])
    manifest = load_component_manifest(PROJECT_ROOT / base_config["targets"])
    digitization = audit_fig19_component_digitization(manifest, verify_source=True)
    if not digitization["summary"]["pass"]:
        raise RuntimeError("Fig. 19 component target integrity failed")

    isolated = simulate_fig19_mlx(base_config, hardware, calibration)
    replay = _h23_replay_check(isolated)
    if not replay["pass"]:
        raise RuntimeError("H23 baseline did not replay exactly")
    fused = simulate_fused_attention(config, base_config, hardware, calibration)
    if not all(item["fusion_invariants"]["pass"] for item in fused):
        raise RuntimeError("fused profile transformation invariant failed")
    attention = _compare_attention(
        digitization["derived_targets"],
        fused,
        isolated,
        tolerance=float(config["decision"]["attention_relative_error_gate"]),
    )

    fused_by_length = {int(item["sequence_length"]): item for item in fused}
    diagnostic_totals = [
        {
            "sequence_length": int(item["sequence_length"]),
            "attention_latency_ms": fused_by_length[int(item["sequence_length"])][
                "attention_latency_ms"
            ],
            "ffn_latency_ms": float(item["ffn_latency_ms"]),
            "total_latency_ms": (
                fused_by_length[int(item["sequence_length"])]["attention_latency_ms"]
                + float(item["ffn_latency_ms"])
            ),
        }
        for item in isolated
    ]
    totals = compare_mlx_transfer(
        digitization["derived_targets"],
        diagnostic_totals,
        component_tolerance=float(config["decision"]["attention_relative_error_gate"]),
        total_tolerance=float(config["decision"]["attention_relative_error_gate"]),
    )
    return {
        "run_id": config["run"]["id"],
        "hypothesis": config["run"]["hypothesis"],
        "protocol": config["run"]["protocol"],
        "classification": config["decision"]["classification"],
        "validation_eligible": bool(config["decision"]["validation_eligible"]),
        "project_git_revision": project_git_revision(),
        "hardware": hardware.to_dict(),
        "calibration": calibration.to_dict(),
        "fusion": dict(config["fusion"]),
        "digitization": digitization,
        "h23_isolated_replay": replay,
        "isolated": isolated,
        "fused_attention": fused,
        "attention_comparison": attention,
        "diagnostic_totals": {
            "simulated": diagnostic_totals,
            "comparison": {
                "points": totals["total_points"],
                "summary": totals["total_summary"],
            },
        },
        "verdict": "supported" if attention["summary"]["all_points_pass"] else "rejected",
    }
