from __future__ import annotations

import math
from dataclasses import replace

from .schema import KernelProfile, StageSpec, Workload

FP16_BYTES = 2
COMPLEX_FP16_BYTES = 4


def _log2(value: int) -> int:
    if value < 1 or value & (value - 1):
        raise ValueError(f"expected a positive power of two, got {value}")
    return value.bit_length() - 1


def _retag(stages: list[StageSpec], offset: int) -> list[StageSpec]:
    return [replace(stage, tag=stage.tag + offset) for stage in stages]


def _bsmm_profile(workload: Workload) -> KernelProfile:
    b = workload.block_size
    output_dim = workload.resolved_output_dim
    stage_count = _log2(b)
    dense_ops = 2.0 * workload.batch * workload.n * workload.d * output_dim
    ratio = 2.0 * stage_count / b
    operations = dense_ops * ratio * workload.projections
    output_elements = workload.batch * workload.n * output_dim * workload.projections
    weight_bytes = workload.d * output_dim * ratio * workload.projections * FP16_BYTES
    input_bytes = workload.batch * workload.n * workload.d * FP16_BYTES
    output_bytes = output_elements * FP16_BYTES

    stages: list[StageSpec] = []
    for stage in range(stage_count):
        stride = 1 << stage
        stages.append(
            StageSpec(
                tag=stage,
                name=f"bsmm-stage-{stage}",
                compute_resource="compute_fma",
                operations=operations / stage_count,
                load_bytes=weight_bytes / stage_count + (input_bytes if stage == 0 else 0.0),
                transfer_bytes=output_bytes if stage + 1 < stage_count else 0.0,
                store_bytes=output_bytes if stage + 1 == stage_count else 0.0,
                route_distance=stride,
                kernel_class="bsmm",
            )
        )
    return KernelProfile(
        operations=operations,
        offchip_bytes=input_bytes + weight_bytes + output_bytes,
        output_elements=output_elements,
        stages=tuple(stages),
        metadata={"butterfly_density": ratio, "stage_count": stage_count},
    )


def _fft_profile(workload: Workload, compressed: bool) -> KernelProfile:
    length = min(workload.n, workload.chunk_length)
    forward_stages = _log2(length)
    retained = max(2, round(length * workload.compression_ratio))
    retained = 1 << round(math.log2(retained))
    inverse_stages = _log2(retained) if compressed else 0
    chunks = math.ceil(workload.n / length)
    vectors = workload.batch * workload.d * chunks * workload.projections

    # Five real FLOPs per point per radix-2 stage is a conventional FFT work proxy.
    forward_ops = 5.0 * vectors * length * forward_stages
    inverse_ops = 5.0 * vectors * retained * inverse_stages
    operations = forward_ops + inverse_ops
    input_bytes = workload.batch * workload.n * workload.d * workload.projections * FP16_BYTES
    output_tokens = workload.n * (workload.compression_ratio if compressed else 1.0)
    output_bytes = workload.batch * output_tokens * workload.d * workload.projections * FP16_BYTES
    full_intermediate = (
        workload.batch * workload.n * workload.d * workload.projections * COMPLEX_FP16_BYTES
    )

    stage_ops: list[float] = []
    stage_names: list[str] = []
    stage_sizes: list[float] = []
    for stage in range(forward_stages):
        stage_ops.append(forward_ops / forward_stages)
        stage_names.append(f"fft-forward-{stage}")
        stage_sizes.append(full_intermediate)
    for stage in range(inverse_stages):
        stage_ops.append(inverse_ops / inverse_stages)
        stage_names.append(f"fft-inverse-{stage}")
        stage_sizes.append(full_intermediate * workload.compression_ratio)

    stages: list[StageSpec] = []
    for index, (ops, name, size) in enumerate(
        zip(stage_ops, stage_names, stage_sizes, strict=True)
    ):
        stride = 1 << (index % forward_stages)
        stages.append(
            StageSpec(
                tag=index,
                name=name,
                compute_resource="compute_fma",
                operations=ops,
                load_bytes=input_bytes if index == 0 else 0.0,
                transfer_bytes=size if index + 1 < len(stage_ops) else 0.0,
                store_bytes=output_bytes if index + 1 == len(stage_ops) else 0.0,
                route_distance=stride,
                kernel_class="fft_cmp" if compressed else "fft",
            )
        )
    return KernelProfile(
        operations=operations,
        offchip_bytes=input_bytes + output_bytes,
        output_elements=workload.batch * output_tokens * workload.d * workload.projections,
        stages=tuple(stages),
        metadata={
            "chunk_length": length,
            "retained_length": retained,
            "stage_count": len(stages),
        },
    )


def _gemm_profile(workload: Workload) -> KernelProfile:
    output_dim = workload.resolved_output_dim
    operations = 2.0 * workload.batch * workload.n * workload.d * output_dim * workload.projections
    input_bytes = workload.batch * workload.n * workload.d * FP16_BYTES
    weight_bytes = workload.d * output_dim * workload.projections * FP16_BYTES
    output_bytes = workload.batch * workload.n * output_dim * workload.projections * FP16_BYTES
    stage = StageSpec(
        tag=0,
        name="gemm-tile",
        compute_resource="compute_fma",
        operations=operations,
        load_bytes=input_bytes + weight_bytes,
        store_bytes=output_bytes,
        kernel_class="gemm",
    )
    return KernelProfile(
        operations=operations,
        offchip_bytes=input_bytes + weight_bytes + output_bytes,
        output_elements=workload.batch * workload.n * output_dim * workload.projections,
        stages=(stage,),
        metadata={"stage_count": 1},
    )


def _swa_profile(workload: Workload) -> KernelProfile:
    score_ops = 2.0 * workload.batch * workload.n * workload.window * workload.d
    value_ops = score_ops
    reduce_ops = workload.batch * workload.n * workload.window
    exp_ops = 4.0 * reduce_ops
    operations = score_ops + value_ops + 2.0 * reduce_ops + exp_ops
    qkv_bytes = 3.0 * workload.batch * workload.n * workload.d * FP16_BYTES
    score_bytes = workload.batch * workload.n * workload.window * FP16_BYTES
    output_bytes = workload.batch * workload.n * workload.d * FP16_BYTES
    stages = (
        StageSpec(
            0,
            "swa-qk",
            "compute_fma",
            score_ops,
            qkv_bytes,
            score_bytes,
            route_distance=workload.query_block,
            kernel_class="swa",
        ),
        StageSpec(
            1,
            "swa-rowmax",
            "compute_fmax",
            reduce_ops,
            transfer_bytes=score_bytes,
            route_distance=workload.query_block,
            kernel_class="swa",
        ),
        StageSpec(
            2,
            "swa-exp-sum",
            "compute_fexp",
            exp_ops + reduce_ops,
            transfer_bytes=score_bytes,
            route_distance=workload.query_block,
            kernel_class="swa",
        ),
        StageSpec(
            3,
            "swa-sv",
            "compute_fma",
            value_ops,
            transfer_bytes=output_bytes,
            store_bytes=output_bytes,
            route_distance=workload.query_block,
            kernel_class="swa",
        ),
    )
    return KernelProfile(
        operations=operations,
        offchip_bytes=qkv_bytes + output_bytes,
        output_elements=workload.batch * workload.n * workload.d,
        stages=stages,
        metadata={"stage_count": len(stages)},
    )


def _attention_profile(workload: Workload) -> KernelProfile:
    profile = _swa_profile(replace(workload, kernel="swa", window=workload.n))
    stages = tuple(replace(stage, kernel_class="attention") for stage in profile.stages)
    return replace(
        profile,
        stages=stages,
        metadata={**profile.metadata, "attention_length": workload.n},
    )


def _transformer_profile(workload: Workload) -> KernelProfile:
    # One structured block: QKV BSMM, semantic FFT compression, attention, output
    # projection, and two FFN projections. The shapes are explicit approximations
    # because the paper does not publish its compiler IR or exact tile schedule.
    parts = [
        _bsmm_profile(replace(workload, kernel="bsmm", projections=3)),
        _fft_profile(replace(workload, kernel="fft_cmp", projections=1), compressed=True),
        _swa_profile(
            replace(
                workload,
                kernel="swa",
                n=max(1, int(workload.n * workload.compression_ratio)),
                window=min(workload.window, max(1, int(workload.n * workload.compression_ratio))),
            )
        ),
        _bsmm_profile(replace(workload, kernel="bsmm", projections=3)),
    ]
    stages: list[StageSpec] = []
    offset = 0
    for part in parts:
        stages.extend(_retag(list(part.stages), offset))
        offset += len(part.stages)
    return KernelProfile(
        operations=sum(part.operations for part in parts),
        offchip_bytes=sum(part.offchip_bytes for part in parts),
        output_elements=max(part.output_elements for part in parts),
        stages=tuple(stages),
        metadata={"stage_count": len(stages), "part_count": len(parts)},
    )


def compile_workload(workload: Workload) -> KernelProfile:
    if workload.kernel == "attention":
        return _attention_profile(workload)
    if workload.kernel == "bsmm":
        return _bsmm_profile(workload)
    if workload.kernel == "fft":
        return _fft_profile(workload, compressed=False)
    if workload.kernel == "fft_cmp":
        return _fft_profile(workload, compressed=True)
    if workload.kernel == "gemm":
        return _gemm_profile(workload)
    if workload.kernel == "swa":
        return _swa_profile(workload)
    if workload.kernel == "transformer":
        return _transformer_profile(workload)
    raise AssertionError(f"unhandled kernel: {workload.kernel}")
