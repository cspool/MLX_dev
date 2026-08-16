from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from .roofline import RooflineCalibration
from .schema import CalibrationConfig, HardwareConfig, Workload
from .simulator import MLXSimulator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET_PATH = PROJECT_ROOT / "artifacts" / "targets" / "paper_targets.yaml"
REDUCED_CONFIG = PROJECT_ROOT / "configs" / "hardware" / "mlx_reduced.yaml"
PAPER_CALIBRATION = PROJECT_ROOT / "configs" / "calibration" / "paper_v1.yaml"
ROOFLINE_CALIBRATION = PROJECT_ROOT / "configs" / "calibration" / "roofline_v1.yaml"


def load_targets(path: str | Path = TARGET_PATH) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        result = yaml.safe_load(handle)
    if not isinstance(result, dict):
        raise TypeError(f"target file must contain a mapping: {path}")
    return result


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        raise ValueError("geometric mean requires positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _audit_series(actual: list[float], target: list[float]) -> dict[str, Any]:
    if len(actual) != len(target):
        raise ValueError("actual and target series lengths differ")
    errors = [abs(a - t) / abs(t) if t else abs(a - t) for a, t in zip(actual, target, strict=True)]
    return {
        "absolute_relative_errors": errors,
        "mean_absolute_percentage_error": sum(errors) / len(errors),
        "maximum_absolute_percentage_error": max(errors),
        "pass_10pct": all(error <= 0.10 for error in errors),
    }


def reproduce_fig22(config_path: str | Path = REDUCED_CONFIG) -> dict[str, Any]:
    hardware = HardwareConfig.from_yaml(config_path)
    calibration = CalibrationConfig.from_yaml(PAPER_CALIBRATION)
    simulator = MLXSimulator(hardware, calibration)
    target = load_targets()["fig22_compute_utilization"]
    sizes = list(target["sizes"])
    bsmm: list[float] = []
    fft: list[float] = []
    raw: dict[str, list[dict[str, Any]]] = {"bsmm": [], "chunk_fft": []}
    for size in sizes:
        bsmm_result = simulator.simulate(
            Workload(
                kernel="bsmm",
                n=size,
                d=512,
                batch=1,
                block_size=32,
                name=f"fig22-bsmm-{size}",
            )
        )
        fft_result = simulator.simulate(
            Workload(
                kernel="fft_cmp",
                n=size,
                d=512,
                batch=1,
                chunk_length=size,
                compression_ratio=0.5,
                name=f"fig22-fft-{size}",
            )
        )
        bsmm.append(bsmm_result.compute_utilization)
        fft.append(fft_result.compute_utilization)
        raw["bsmm"].append(bsmm_result.to_dict())
        raw["chunk_fft"].append(fft_result.to_dict())
    return {
        "figure": 22,
        "hardware": hardware.to_dict(),
        "calibration": calibration.to_dict(),
        "sizes": sizes,
        "actual": {"bsmm": bsmm, "chunk_fft": fft},
        "target": {"bsmm": target["bsmm"], "chunk_fft": target["chunk_fft"]},
        "audit": {
            "bsmm": _audit_series(bsmm, target["bsmm"]),
            "chunk_fft": _audit_series(fft, target["chunk_fft"]),
        },
        "raw": raw,
    }


def reproduce_fig23(config_path: str | Path = REDUCED_CONFIG) -> dict[str, Any]:
    baseline = HardwareConfig.from_yaml(config_path)
    calibration = CalibrationConfig.from_yaml(PAPER_CALIBRATION)
    target = load_targets()["fig23_scalability"]
    sizes = list(target["sequence_lengths"])
    configurations = {
        "baseline": baseline,
        "simd32_4x4": replace(baseline, name="mlx-simd32-4x4", simd_width=32),
        "simd8_8x8": replace(baseline, name="mlx-simd8-8x8", mesh_x=8, mesh_y=8),
        "simd32_8x8": replace(baseline, name="mlx-simd32-8x8", simd_width=32, mesh_x=8, mesh_y=8),
    }
    cycles: dict[str, list[float]] = {name: [] for name in configurations}
    raw: dict[str, list[dict[str, Any]]] = {name: [] for name in configurations}
    for size in sizes:
        workload = Workload(
            kernel="transformer",
            n=size,
            d=512,
            batch=8,
            block_size=32,
            compression_ratio=0.5,
            chunk_length=64,
            window=128,
            query_block=32,
            name=f"fig23-transformer-{size}",
        )
        for name, hardware in configurations.items():
            result = MLXSimulator(hardware, calibration).simulate(workload)
            cycles[name].append(result.cycles)
            raw[name].append(result.to_dict())

    speedups = {
        name: [base / current for base, current in zip(cycles["baseline"], values, strict=True)]
        for name, values in cycles.items()
        if name != "baseline"
    }
    gmeans = {name: geometric_mean(values) for name, values in speedups.items()}
    return {
        "figure": 23,
        "calibration": calibration.to_dict(),
        "sizes": sizes,
        "actual": speedups,
        "target": {name: target[name] for name in ("simd32_4x4", "simd8_8x8", "simd32_8x8")},
        "geometric_means": gmeans,
        "audit": {name: _audit_series(values, target[name]) for name, values in speedups.items()},
        "raw": raw,
    }


def run_h2_ablations(config_path: str | Path = REDUCED_CONFIG) -> dict[str, Any]:
    baseline = HardwareConfig.from_yaml(config_path)
    calibration = CalibrationConfig.from_yaml(PAPER_CALIBRATION)
    workload = Workload(
        kernel="fft_cmp",
        n=512,
        d=64,
        batch=1,
        block_size=32,
        compression_ratio=0.5,
        chunk_length=512,
        window=128,
        query_block=32,
        name="h2-ablation-communication-sensitive-fft512",
    )
    configurations = {
        "baseline": baseline,
        "single_active_tag": replace(baseline, name="mlx-single-tag", active_tags=1),
        "no_skip_links": replace(
            baseline,
            name="mlx-no-skip",
            skip_distance=1,
            max_skip_hops=max(baseline.mesh_x, baseline.mesh_y),
        ),
        "unified_pipeline": replace(
            baseline, name="mlx-unified-pipeline", decoupled_pipelines=False
        ),
    }
    results = {
        name: MLXSimulator(hardware, calibration).simulate(workload).to_dict()
        for name, hardware in configurations.items()
    }
    baseline_cycles = results["baseline"]["cycles"]
    return {
        "experiment": "h2-ablations",
        "calibration": calibration.to_dict(),
        "workload": workload.to_dict(),
        "results": results,
        "cycle_regression": {
            name: result["cycles"] / baseline_cycles for name, result in results.items()
        },
    }


def reproduce_fig25() -> dict[str, Any]:
    calibration = RooflineCalibration.from_yaml(ROOFLINE_CALIBRATION)
    target = load_targets()["fig25_roofline_utilization"]
    cases = [
        ("BERT_512", 512, 1024),
        ("Llama2_1K", 1024, 4096),
        ("InternLM2_4K", 4096, 4096),
        ("BERT_8K", 8192, 1024),
    ]
    operators = list(target["operators"])
    actual: dict[str, list[list[float]]] = {}
    audits: dict[str, dict[str, Any]] = {}
    for system in ("rtx3090", "orin", "mlx"):
        matrix = [
            [calibration.utilization(system, operator, n, d) for _, n, d in cases]
            for operator in operators
        ]
        actual[system] = matrix
        flat_actual = [value for row in matrix for value in row]
        flat_target = [value for row in target["heatmap"][system] for value in row]
        audits[system] = _audit_series(flat_actual, flat_target)
    return {
        "figure": 25,
        "classification": "calibration-replay",
        "validation_eligible": False,
        "fit_degrees_of_freedom": {
            "per_surface": 4,
            "anchors_per_surface": 4,
        },
        "protocol_note": (
            "The four coefficients exactly interpolate the four digitized cells in each "
            "system/operator surface. This exercises the roofline pipeline but cannot validate it."
        ),
        "cases": [name for name, _, _ in cases],
        "operators": operators,
        "calibration": calibration.to_dict(),
        "actual": actual,
        "target": target["heatmap"],
        "audit": audits,
    }


def _fig24_workload(operator: str, n: int, d: int) -> Workload:
    common = {"n": n, "d": d, "batch": 32, "name": f"fig24-{operator}-{n}-{d}"}
    if operator == "fft_cmp":
        return Workload(kernel="fft_cmp", chunk_length=64, compression_ratio=0.5, **common)
    if operator.startswith("qkv_bsmm"):
        block_size = {"qkv_bsmm": 16, "qkv_bsmm_b32": 32, "qkv_bsmm_b64": 64}[operator]
        return Workload(kernel="bsmm", block_size=block_size, projections=3, **common)
    if operator == "swa_w128_q32":
        return Workload(kernel="swa", window=128, query_block=32, **common)
    if operator == "swa_w256_q64":
        return Workload(kernel="swa", window=256, query_block=64, **common)
    raise ValueError(f"unknown Fig. 24 operator: {operator}")


def reproduce_fig24() -> dict[str, Any]:
    full_hardware = HardwareConfig.from_yaml(PROJECT_ROOT / "configs/hardware/mlx_full.yaml")
    event_calibration = CalibrationConfig.from_yaml(PAPER_CALIBRATION)
    roofline_calibration = RooflineCalibration.from_yaml(ROOFLINE_CALIBRATION)
    simulator = MLXSimulator(full_hardware, event_calibration)
    target = load_targets()["fig24_structured_sweep"]
    cases = [
        ("BERT", "BERT_512", 512, 1024),
        ("BERT", "BERT_8K", 8192, 1024),
        ("Llama2", "Llama2_512", 512, 4096),
        ("Llama2", "Llama2_1K", 1024, 4096),
        ("Llama2", "Llama2_4K", 4096, 4096),
        ("InternLM2", "InternLM2_2K", 2048, 4096),
        ("InternLM2", "InternLM2_8K", 8192, 4096),
    ]
    operators = list(target["mlx_over_orin"])
    actual: dict[str, list[float]] = {}
    raw: dict[str, list[dict[str, Any]]] = {}
    audits: dict[str, dict[str, Any]] = {}
    for operator in operators:
        ratios: list[float] = []
        details: list[dict[str, Any]] = []
        for family, case_name, n, d in cases:
            mlx_result = simulator.simulate(_fig24_workload(operator, n, d))
            orin_gops = roofline_calibration.baseline_gops("orin", operator, family, n)
            ratio = mlx_result.achieved_gops / orin_gops
            ratios.append(ratio)
            details.append(
                {
                    "case": case_name,
                    "mlx": mlx_result.to_dict(),
                    "orin_proxy_gops": orin_gops,
                    "ratio": ratio,
                }
            )
        actual[operator] = ratios
        raw[operator] = details
        audits[operator] = _audit_series(ratios, target["mlx_over_orin"][operator])
    return {
        "figure": 24,
        "classification": "calibration-replay-gpu-proxy",
        "validation_eligible": False,
        "fit_degrees_of_freedom": {
            "per_operator_gpu_surface": 7,
            "anchors_per_operator": 7,
        },
        "protocol_note": (
            "The Orin proxy has seven coefficients for each seven-point paper series. "
            "It is a saturated calibration replay and not the held-out test pre-registered in H2."
        ),
        "cases": [case_name for _, case_name, _, _ in cases],
        "actual": actual,
        "target": target["mlx_over_orin"],
        "audit": audits,
        "event_calibration": event_calibration.to_dict(),
        "roofline_calibration": roofline_calibration.to_dict(),
        "raw": raw,
    }


def reproduce(figure: str | int) -> dict[str, Any]:
    value = str(figure).lower()
    if value == "22":
        return reproduce_fig22()
    if value == "23":
        return reproduce_fig23()
    if value == "25":
        return reproduce_fig25()
    if value == "24":
        return reproduce_fig24()
    if value == "all":
        return {
            "fig22": reproduce_fig22(),
            "fig23": reproduce_fig23(),
            "h2_ablations": run_h2_ablations(),
            "fig24": reproduce_fig24(),
            "fig25": reproduce_fig25(),
        }
    if value in {"h2", "h2-ablations"}:
        return run_h2_ablations()
    raise ValueError(
        f"implemented figures are 22, 23, 24, 25, h2-ablations, or all; got {figure!r}"
    )


def write_json(data: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
