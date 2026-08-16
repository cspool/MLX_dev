from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from .gpu import GpuBaselineConfig
from .roofline import RooflineCalibration
from .schema import CalibrationConfig, HardwareConfig, Workload
from .simulator import MLXSimulator
from .workloads import compile_workload

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET_PATH = PROJECT_ROOT / "artifacts" / "targets" / "paper_targets.yaml"
REDUCED_CONFIG = PROJECT_ROOT / "configs" / "hardware" / "mlx_reduced.yaml"
PAPER_CALIBRATION = PROJECT_ROOT / "configs" / "calibration" / "paper_v1.yaml"
ROOFLINE_CALIBRATION = PROJECT_ROOT / "configs" / "calibration" / "roofline_v1.yaml"
XAVIER_CONFIG = PROJECT_ROOT / "configs" / "hardware" / "xavier_paper.yaml"

LLAMA_D = 4096
LLAMA_FFN_D = 11008
LLAMA_LAYERS = 32
LLAMA_MODIFIED_LAYERS = 24
LLAMA_HEADS = 32
LLAMA_HEAD_D = 128
LLAMA_VOCAB = 32000
FP16_BYTES = 2


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


def _llama_kernel_workloads(n: int, *, sparse: bool, batch: int) -> dict[str, list[Workload]]:
    projection_kernel = "bsmm" if sparse else "gemm"
    common = {"n": n, "batch": batch, "block_size": 32}
    if sparse:
        attention = [
            Workload(
                kernel="fft_cmp",
                d=LLAMA_D,
                projections=3,
                compression_ratio=0.5,
                chunk_length=n,
                name=f"llama-fft-cmp-N{n}",
                **common,
            ),
            Workload(
                kernel="attention",
                n=n // 2,
                d=LLAMA_D,
                batch=batch,
                name=f"llama-compressed-attention-N{n}",
            ),
        ]
    else:
        attention = [
            Workload(
                kernel="attention",
                d=LLAMA_D,
                name=f"llama-dense-attention-N{n}",
                **common,
            )
        ]
    return {
        "qkv": [
            Workload(
                kernel=projection_kernel,
                d=LLAMA_D,
                projections=3,
                name=f"llama-{'structured' if sparse else 'dense'}-qkv-N{n}",
                **common,
            )
        ],
        "attention": attention,
        "output": [
            Workload(
                kernel=projection_kernel,
                d=LLAMA_D,
                name=f"llama-{'structured' if sparse else 'dense'}-output-N{n}",
                **common,
            )
        ],
        "ffn1": [
            Workload(
                kernel=projection_kernel,
                d=LLAMA_D,
                output_dim=LLAMA_FFN_D,
                name=f"llama-{'structured' if sparse else 'dense'}-ffn1-N{n}",
                **common,
            )
        ],
        "ffn2": [
            Workload(
                kernel=projection_kernel,
                d=LLAMA_FFN_D,
                output_dim=LLAMA_D,
                name=f"llama-{'structured' if sparse else 'dense'}-ffn2-N{n}",
                **common,
            )
        ],
    }


def _aggregate_mlx(simulator: MLXSimulator, workloads: list[Workload]) -> dict[str, Any]:
    components = [simulator.simulate(workload).to_dict() for workload in workloads]
    latency_us = sum(component["latency_us"] for component in components)
    energy_mj = sum(component["energy_mj"] for component in components)
    return {
        "latency_us": latency_us,
        "energy_mj": energy_mj,
        "operations": sum(component["operations"] for component in components),
        "offchip_bytes": sum(component["offchip_bytes"] for component in components),
        "average_power_w": energy_mj / max(latency_us, 1e-30) * 1000.0,
        "components": components,
    }


def _aggregate_gpu(
    gpu: GpuBaselineConfig, workloads: list[Workload], n: int, mode: str
) -> dict[str, Any]:
    components = [gpu.predict(compile_workload(workload), n, mode) for workload in workloads]
    return {
        "latency_us": sum(float(component["latency_us"]) for component in components),
        "energy_mj": sum(float(component["energy_mj"]) for component in components),
        "operations": sum(float(component["operations"]) for component in components),
        "offchip_bytes": sum(float(component["offchip_bytes"]) for component in components),
        "components": components,
    }


def reproduce_fig20() -> dict[str, Any]:
    hardware = HardwareConfig.from_yaml(PROJECT_ROOT / "configs/hardware/mlx_full.yaml")
    calibration = CalibrationConfig.from_yaml(PAPER_CALIBRATION)
    simulator = MLXSimulator(hardware, calibration)
    gpu = GpuBaselineConfig.from_yaml(XAVIER_CONFIG)
    target = load_targets()["fig20_xavier_kernels"]

    actual = {
        "versus_dense_tcu": {"speedup": [], "energy_saving": []},
        "versus_sparse_cuda": {"speedup": [], "energy_saving": []},
    }
    raw: list[dict[str, Any]] = []
    for n in (256, 8192):
        structured = _llama_kernel_workloads(n, sparse=True, batch=1)
        dense = _llama_kernel_workloads(n, sparse=False, batch=1)
        for kernel in ("qkv", "attention", "ffn1", "ffn2"):
            mlx = _aggregate_mlx(simulator, structured[kernel])
            dense_gpu = _aggregate_gpu(gpu, dense[kernel], n, "tensor")
            sparse_gpu = _aggregate_gpu(gpu, structured[kernel], n, "cuda")
            actual["versus_dense_tcu"]["speedup"].append(
                dense_gpu["latency_us"] / mlx["latency_us"]
            )
            actual["versus_dense_tcu"]["energy_saving"].append(
                dense_gpu["energy_mj"] / mlx["energy_mj"]
            )
            actual["versus_sparse_cuda"]["speedup"].append(
                sparse_gpu["latency_us"] / mlx["latency_us"]
            )
            actual["versus_sparse_cuda"]["energy_saving"].append(
                sparse_gpu["energy_mj"] / mlx["energy_mj"]
            )
            raw.append(
                {
                    "case": f"{kernel}-N{n}",
                    "mlx": mlx,
                    "xavier_dense_tcu": dense_gpu,
                    "xavier_sparse_cuda": sparse_gpu,
                }
            )

    for baseline in actual.values():
        baseline["speedup"].append(geometric_mean(baseline["speedup"]))
        baseline["energy_saving"].append(geometric_mean(baseline["energy_saving"]))

    target_series = {
        baseline: {
            metric: list(target[baseline][metric]) for metric in ("speedup", "energy_saving")
        }
        for baseline in ("versus_dense_tcu", "versus_sparse_cuda")
    }
    audits = {
        baseline: {
            metric: _audit_series(actual[baseline][metric], target_series[baseline][metric])
            for metric in ("speedup", "energy_saving")
        }
        for baseline in target_series
    }
    return {
        "figure": 20,
        "classification": "held-out-cross-device-prediction",
        "validation_eligible": True,
        "groups": target["groups"],
        "actual": actual,
        "target": target_series,
        "audit": audits,
        "mlx_hardware": hardware.to_dict(),
        "event_calibration": calibration.to_dict(),
        "xavier": gpu.to_dict(),
        "raw": raw,
    }


def _llama_memory_gb(n: int, batch: int = 8) -> dict[str, float]:
    projection_parameters_per_layer = 4 * LLAMA_D * LLAMA_D + 3 * LLAMA_D * LLAMA_FFN_D
    normalization_parameters = 2 * LLAMA_D * LLAMA_LAYERS + LLAMA_D
    embedding_parameters = 2 * LLAMA_VOCAB * LLAMA_D
    dense_parameters = (
        projection_parameters_per_layer * LLAMA_LAYERS
        + normalization_parameters
        + embedding_parameters
    )
    butterfly_density = 2 * math.log2(32) / 32
    sparse_parameters = (
        projection_parameters_per_layer
        * (LLAMA_LAYERS - LLAMA_MODIFIED_LAYERS + LLAMA_MODIFIED_LAYERS * butterfly_density)
        + normalization_parameters
        + embedding_parameters
    )
    dense_kv_elements = 2 * LLAMA_LAYERS * batch * n * LLAMA_HEADS * LLAMA_HEAD_D
    sparse_kv_elements = (
        2
        * batch
        * n
        * LLAMA_HEADS
        * LLAMA_HEAD_D
        * (LLAMA_LAYERS - LLAMA_MODIFIED_LAYERS + 0.5 * LLAMA_MODIFIED_LAYERS)
    )
    live_qkv_elements = 3 * batch * n * LLAMA_D
    return {
        "dense": ((dense_parameters + dense_kv_elements + live_qkv_elements) * FP16_BYTES / 1e9),
        "sparse": ((sparse_parameters + sparse_kv_elements + live_qkv_elements) * FP16_BYTES / 1e9),
        "dense_parameter_gb": dense_parameters * FP16_BYTES / 1e9,
        "sparse_parameter_gb": sparse_parameters * FP16_BYTES / 1e9,
        "butterfly_density": butterfly_density,
    }


def reproduce_fig21() -> dict[str, Any]:
    hardware = HardwareConfig.from_yaml(PROJECT_ROOT / "configs/hardware/mlx_full.yaml")
    calibration = CalibrationConfig.from_yaml(PAPER_CALIBRATION)
    simulator = MLXSimulator(hardware, calibration)
    gpu = GpuBaselineConfig.from_yaml(XAVIER_CONFIG)
    target = load_targets()["fig21_end_to_end"]
    sizes = list(target["sequence_lengths"])
    speedups: list[float] = []
    dense_memory: list[float] = []
    sparse_memory: list[float] = []
    status: list[str] = []
    raw: list[dict[str, Any]] = []
    for n in sizes:
        structured = _llama_kernel_workloads(n, sparse=True, batch=8)
        dense = _llama_kernel_workloads(n, sparse=False, batch=8)
        block_order = ("qkv", "attention", "output", "ffn1", "ffn2")
        mlx_structured = [_aggregate_mlx(simulator, structured[kernel]) for kernel in block_order]
        mlx_dense = [_aggregate_mlx(simulator, dense[kernel]) for kernel in block_order]
        xavier_dense = [_aggregate_gpu(gpu, dense[kernel], n, "tensor") for kernel in block_order]
        mlx_latency_us = LLAMA_MODIFIED_LAYERS * sum(
            component["latency_us"] for component in mlx_structured
        ) + (LLAMA_LAYERS - LLAMA_MODIFIED_LAYERS) * sum(
            component["latency_us"] for component in mlx_dense
        )
        xavier_latency_us = LLAMA_LAYERS * sum(
            component["latency_us"] for component in xavier_dense
        )
        speedups.append(xavier_latency_us / mlx_latency_us)
        memory = _llama_memory_gb(n)
        dense_memory.append(memory["dense"])
        sparse_memory.append(memory["sparse"])
        status.append(
            "within-xavier-capacity"
            if memory["dense"] <= gpu.memory_capacity_gb
            else "projected-over-xavier-capacity"
        )
        raw.append(
            {
                "sequence_length": n,
                "mlx_latency_us": mlx_latency_us,
                "xavier_latency_us": xavier_latency_us,
                "memory": memory,
                "mlx_structured_components": mlx_structured,
                "mlx_dense_components": mlx_dense,
                "xavier_dense_components": xavier_dense,
            }
        )
    actual = {
        "speedup_over_xavier": speedups,
        "dense_memory_gb": dense_memory,
        "sparse_memory_gb": sparse_memory,
    }
    target_series = {
        "speedup_over_xavier": list(target["speedup_over_xavier"]),
        "dense_memory_gb": list(target["dense_memory_gb"]),
        "sparse_memory_gb": list(target["sparse_memory_gb"]),
    }
    return {
        "figure": 21,
        "classification": "held-out-cross-device-prediction",
        "validation_eligible": True,
        "sequence_lengths": sizes,
        "xavier_execution_status": status,
        "actual": actual,
        "target": target_series,
        "audit": {name: _audit_series(actual[name], target_series[name]) for name in actual},
        "mlx_hardware": hardware.to_dict(),
        "event_calibration": calibration.to_dict(),
        "xavier": gpu.to_dict(),
        "raw": raw,
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
    if value == "20":
        return reproduce_fig20()
    if value == "21":
        return reproduce_fig21()
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
            "fig20": reproduce_fig20(),
            "fig21": reproduce_fig21(),
            "fig22": reproduce_fig22(),
            "fig23": reproduce_fig23(),
            "h2_ablations": run_h2_ablations(),
            "fig24": reproduce_fig24(),
            "fig25": reproduce_fig25(),
        }
    if value in {"h2", "h2-ablations"}:
        return run_h2_ablations()
    raise ValueError(f"implemented figures are 20-25, h2-ablations, or all; got {figure!r}")


def write_json(data: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
