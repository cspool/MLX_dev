#!/usr/bin/env python3
"""Collect target-free shape-matched RTX4090 traces for Figures 19/20/23."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.nn import functional
from torch.nn.attention import SDPBackend, sdpa_kernel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig19_20_23_rtx4090_trace_v1.yaml"


def digest(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def gpu_snapshot(index: int) -> dict[str, str]:
    fields = (
        "index,name,uuid,compute_cap,memory.total,driver_version,pstate,"
        "clocks.current.graphics,clocks.max.graphics,power.draw,power.limit,temperature.gpu"
    )
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={index}",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return dict(zip(fields.split(","), result.stdout.strip().split(", "), strict=True))


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_samples(samples: list[float]) -> dict[str, float]:
    return {
        "minimum_ms": min(samples),
        "p25_ms": quantile(samples, 0.25),
        "median_ms": quantile(samples, 0.50),
        "p75_ms": quantile(samples, 0.75),
        "maximum_ms": max(samples),
        "mean_ms": sum(samples) / len(samples),
    }


def sampled_checksum(output: torch.Tensor) -> float:
    flat = output.reshape(-1)
    stride = max(1, flat.numel() // 4096)
    return float(flat[::stride].float().sum().item())


def time_callable(
    function: Callable[[], torch.Tensor], *, warmup: int, iterations: int
) -> tuple[list[float], torch.Tensor]:
    output: torch.Tensor | None = None
    for _ in range(warmup):
        output = function()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        output = function()
        stop.record()
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)))
    if output is None:
        raise RuntimeError("timed callable produced no output")
    return samples, output


def random_tensor(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(shape, device="cuda", dtype=dtype) * 0.03125


def dense_projection_builder(
    *, tokens: int, input_dim: int, output_dim: int
) -> tuple[Callable[[], torch.Tensor], dict[str, Any]]:
    x = random_tensor((tokens, input_dim), torch.float16)
    weight = random_tensor((input_dim, output_dim), torch.float16)

    def execute() -> torch.Tensor:
        return torch.matmul(x, weight)

    metadata = {
        "service_class": "dense_tcu",
        "dtype": "float16",
        "tokens": tokens,
        "input_dimension": input_dim,
        "output_dimension": output_dim,
        "logical_flops": 2 * tokens * input_dim * output_dim,
    }
    return execute, metadata


def structured_projection_builder(
    *,
    tokens: int,
    width: int,
    block_size: int,
    copies: int,
    compression_ratio: float,
) -> tuple[Callable[[], torch.Tensor], dict[str, Any]]:
    active_width = int(width * compression_ratio) // block_size * block_size
    active_width = max(block_size, active_width)
    blocks = active_width // block_size
    stages = int(math.log2(block_size))
    x = random_tensor((tokens, blocks, block_size), torch.float32)
    weights = random_tensor((copies, stages, blocks, block_size, block_size), torch.float32)

    def execute() -> torch.Tensor:
        result = x
        for copy_index in range(copies):
            value = x
            for stage in range(stages):
                value = torch.einsum("tbi,bio->tbo", value, weights[copy_index, stage])
                value = torch.tanh(value * 0.03125)
            result = value
        return result

    metadata = {
        "service_class": "structured_cuda_core",
        "dtype": "float32",
        "tokens": tokens,
        "width": width,
        "active_width": active_width,
        "block_size": block_size,
        "blocks": blocks,
        "stages": stages,
        "copies": copies,
        "compression_ratio": compression_ratio,
        "logical_flops": 2 * tokens * active_width * block_size * stages * copies,
    }
    return execute, metadata


def fft2d_builder(
    *, batch: int, heads: int, sequence: int, head_dimension: int
) -> tuple[Callable[[], torch.Tensor], dict[str, Any]]:
    x = random_tensor((batch, heads, sequence, head_dimension), torch.float32)

    def execute() -> torch.Tensor:
        spectrum = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        return torch.fft.irfft2(
            spectrum, s=(sequence, head_dimension), dim=(-2, -1), norm="ortho"
        )

    metadata = {
        "service_class": "structured_cuda_core",
        "dtype": "float32",
        "batch": batch,
        "heads": heads,
        "sequence_length": sequence,
        "head_dimension": head_dimension,
        "transform": "rfft2+irfft2",
    }
    return execute, metadata


def fft_sequence_builder(
    *, batch: int, sequence: int, dimension: int
) -> tuple[Callable[[], torch.Tensor], dict[str, Any]]:
    x = random_tensor((batch, dimension, sequence), torch.float32)

    def execute() -> torch.Tensor:
        spectrum = torch.fft.rfft(x, dim=-1, norm="ortho")
        return torch.fft.irfft(spectrum, n=sequence, dim=-1, norm="ortho")

    metadata = {
        "service_class": "structured_cuda_core",
        "dtype": "float32",
        "batch": batch,
        "sequence_length": sequence,
        "dimension": dimension,
        "transform": "rfft+irfft",
    }
    return execute, metadata


def dense_attention_builder(
    *, batch: int, heads: int, sequence: int, head_dimension: int
) -> tuple[Callable[[], torch.Tensor], dict[str, Any]]:
    shape = (batch, heads, sequence, head_dimension)
    query = random_tensor(shape, torch.float16)
    key = random_tensor(shape, torch.float16)
    value = random_tensor(shape, torch.float16)

    def execute() -> torch.Tensor:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return functional.scaled_dot_product_attention(
                query, key, value, is_causal=True
            )

    metadata = {
        "service_class": "dense_flash_attention",
        "dtype": "float16",
        "batch": batch,
        "heads": heads,
        "sequence_length": sequence,
        "head_dimension": head_dimension,
        "logical_flops": 4 * batch * heads * sequence * sequence * head_dimension,
    }
    return execute, metadata


def case_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    figure19 = config["workloads"]["figure19"]
    for sequence in figure19["sequence_lengths"]:
        for component in figure19["trace_components"]:
            specs.append({"figure": 19, "sequence_length": int(sequence), "component": component})
    figure20 = config["workloads"]["figure20"]
    for sequence in figure20["sequence_lengths"]:
        for component in figure20["trace_components"]:
            specs.append({"figure": 20, "sequence_length": int(sequence), "component": component})
    figure23 = config["workloads"]["figure23"]
    for sequence in figure23["sequence_lengths"]:
        for component in figure23["trace_components"]:
            specs.append({"figure": 23, "sequence_length": int(sequence), "component": component})
    return specs


def build_case(
    config: dict[str, Any], spec: dict[str, Any]
) -> tuple[Callable[[], torch.Tensor], dict[str, Any]]:
    figure = int(spec["figure"])
    sequence = int(spec["sequence_length"])
    component = str(spec["component"])
    workload = config["workloads"][f"figure{figure}"]
    batch = int(workload["batch"])
    tokens = batch * sequence
    hidden = int(workload["hidden_dimension"])
    block = int(workload["block_size"])
    if figure == 19:
        if component == "fft2d":
            return fft2d_builder(
                batch=batch,
                heads=int(workload["heads"]),
                sequence=sequence,
                head_dimension=int(workload["head_dimension"]),
            )
        if component == "bsmm_ffn1":
            return structured_projection_builder(
                tokens=tokens,
                width=hidden,
                block_size=block,
                copies=int(workload["ffn_dimension"]) // hidden,
                compression_ratio=1.0,
            )
        if component == "bsmm_ffn2":
            return structured_projection_builder(
                tokens=tokens,
                width=int(workload["ffn_dimension"]),
                block_size=block,
                copies=1,
                compression_ratio=1.0,
            )
    if figure == 20:
        ffn = int(workload["ffn_dimension"])
        ratio = float(workload["compression_ratio"])
        if component == "dense_tcu_qkv":
            return dense_projection_builder(tokens=tokens, input_dim=hidden, output_dim=3 * hidden)
        if component == "sparse_cuda_qkv":
            return structured_projection_builder(
                tokens=tokens,
                width=hidden,
                block_size=block,
                copies=3,
                compression_ratio=ratio,
            )
        if component == "dense_flash_attention":
            return dense_attention_builder(
                batch=batch,
                heads=int(workload["heads"]),
                sequence=sequence,
                head_dimension=int(workload["head_dimension"]),
            )
        if component == "sparse_cuda_fft_attention":
            return fft_sequence_builder(batch=batch, sequence=sequence, dimension=hidden)
        if component == "dense_tcu_ffn1":
            return dense_projection_builder(tokens=tokens, input_dim=hidden, output_dim=ffn)
        if component == "sparse_cuda_ffn1":
            return structured_projection_builder(
                tokens=tokens,
                width=hidden,
                block_size=block,
                copies=math.ceil(ffn / hidden),
                compression_ratio=ratio,
            )
        if component == "dense_tcu_ffn2":
            return dense_projection_builder(tokens=tokens, input_dim=ffn, output_dim=hidden)
        if component == "sparse_cuda_ffn2":
            return structured_projection_builder(
                tokens=tokens,
                width=ffn,
                block_size=block,
                copies=1,
                compression_ratio=ratio,
            )
    if figure == 23:
        if component == "fft_cmp":
            return fft_sequence_builder(batch=batch, sequence=sequence, dimension=hidden)
        if component == "bsmm":
            return structured_projection_builder(
                tokens=tokens,
                width=hidden,
                block_size=block,
                copies=4,
                compression_ratio=1.0,
            )
    raise ValueError(f"unsupported trace case: {spec}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu"]["index"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.manual_seed(int(config["timing"]["deterministic_seed"]))
    torch.cuda.manual_seed_all(int(config["timing"]["deterministic_seed"]))
    torch.backends.cuda.matmul.allow_tf32 = bool(config["gpu"]["allow_tf32"])
    torch.backends.cudnn.allow_tf32 = bool(config["gpu"]["allow_tf32"])
    before = gpu_snapshot(int(config["gpu"]["index"]))
    records: list[dict[str, Any]] = []
    for index, spec in enumerate(case_specs(config)):
        torch.manual_seed(int(config["timing"]["deterministic_seed"]) + index)
        function, metadata = build_case(config, spec)
        iterations = int(config["timing"]["timed_iterations"])
        if int(spec["sequence_length"]) <= int(
            config["timing"]["small_case_max_sequence"]
        ):
            iterations += int(config["timing"]["small_case_extra_iterations"])
        samples, output = time_callable(
            function,
            warmup=int(config["timing"]["warmup_iterations"]),
            iterations=iterations,
        )
        output_finite = bool(torch.isfinite(output).all().item())
        checksum = sampled_checksum(output)
        record = {
            **spec,
            "key": f"fig{spec['figure']}-N{spec['sequence_length']}-{spec['component']}",
            "metadata": metadata,
            "timing_samples_ms": samples,
            "timing": summarize_samples(samples),
            "output_elements": output.numel(),
            "output_finite": output_finite,
            "sampled_checksum": checksum,
        }
        records.append(record)
        print(
            f"[H182] {record['key']} median_ms={record['timing']['median_ms']:.6f}",
            flush=True,
        )
        del output, function
        gc.collect()
        torch.cuda.empty_cache()
    after = gpu_snapshot(int(config["gpu"]["index"]))
    output_root = PROJECT_ROOT / config["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paper_performance_targets_consumed": False,
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0),
            "compute_capability": ".".join(
                str(value) for value in torch.cuda.get_device_capability(0)
            ),
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        },
        "gpu_before": before,
        "gpu_after": after,
        "cases": records,
        "checks": {
            "case_count": len(records) == int(config["acceptance"]["required_total_cases"]),
            "samples": all(
                len(record["timing_samples_ms"])
                == int(config["timing"]["timed_iterations"])
                + (
                    int(config["timing"]["small_case_extra_iterations"])
                    if record["sequence_length"]
                    <= int(config["timing"]["small_case_max_sequence"])
                    else 0
                )
                for record in records
            ),
            "finite": all(
                record["output_finite"]
                and math.isfinite(record["sampled_checksum"])
                and all(math.isfinite(value) and value > 0 for value in record["timing_samples_ms"])
                for record in records
            ),
            "gpu": before["name"] == config["gpu"]["expected_name"]
            and before["uuid"] == config["gpu"]["expected_uuid"]
            and before["compute_cap"] == config["gpu"]["expected_compute_capability"],
            "target_free": config["acceptance"]["paper_targets_consumed"] is False,
        },
    }
    path = PROJECT_ROOT / config["manifest_path"]
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"checks": manifest["checks"], "cases": len(records)}, indent=2))
    return 0 if all(manifest["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
