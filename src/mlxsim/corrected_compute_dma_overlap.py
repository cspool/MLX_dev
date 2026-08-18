"""Corrected-cycle composition for the target-free H111 overlap envelope."""

from __future__ import annotations

import math
from typing import Any

from mlxsim.compute_dma_overlap import compose_point


def compose_corrected_point(
    *,
    key: str,
    corrected_cycles: int,
    h110_issue_utilization: float,
    path: dict[str, Any],
    bandwidth: int,
    physical_pes: int,
    simd_width: int,
    effective_ops_per_fma: int,
    exact_peak_effective_ops_per_cycle: float,
    nominal_peak_effective_ops_per_cycle: float,
    nominal_peak_relative_difference_limit: float,
) -> dict[str, Any]:
    """Compose one H110/H107 point without using residence estimates."""
    if corrected_cycles <= 0 or physical_pes <= 0 or simd_width <= 0:
        raise ValueError("corrected compute dimensions must be positive")
    if effective_ops_per_fma <= 0:
        raise ValueError("effective operations per FMA must be positive")

    point = compose_point(
        key=key,
        h102_cycles=corrected_cycles,
        path=path,
        bandwidth=bandwidth,
        peak_effective_ops_per_cycle=exact_peak_effective_ops_per_cycle,
    )
    point.pop("figure25_reproduction")
    point["paper_reproduction_claim"] = None

    fma_count = int(path["fma_count"])
    effective_flops = int(path["effective_flops"])
    fma_capacity = physical_pes * simd_width
    reconstructed_issue_utilization = fma_count / (
        corrected_cycles * fma_capacity
    )
    direct_effective_throughput = effective_flops / corrected_cycles
    nominal_difference = abs(
        exact_peak_effective_ops_per_cycle
        - nominal_peak_effective_ops_per_cycle
    ) / nominal_peak_effective_ops_per_cycle
    point["corrected_compute"] = {
        "source": "h110_validated_cycle_fold",
        "cycles": corrected_cycles,
        "fma_count": fma_count,
        "fma_capacity_per_cycle": fma_capacity,
        "direct_fma_issues_per_cycle": fma_count / corrected_cycles,
        "direct_effective_ops_per_cycle": direct_effective_throughput,
        "direct_fma_issue_utilization": reconstructed_issue_utilization,
        "h110_fma_issue_utilization": h110_issue_utilization,
    }
    point["peak_contract"] = {
        "physical_pes": physical_pes,
        "simd_width": simd_width,
        "effective_ops_per_fma": effective_ops_per_fma,
        "exact_peak_effective_ops_per_cycle": (
            exact_peak_effective_ops_per_cycle
        ),
        "nominal_peak_effective_ops_per_cycle": (
            nominal_peak_effective_ops_per_cycle
        ),
        "nominal_peak_relative_difference": nominal_difference,
    }
    point["checks"].update(
        {
            "effective_fma_work": effective_flops
            == fma_count * effective_ops_per_fma,
            "issue_reconstruction": math.isclose(
                reconstructed_issue_utilization,
                h110_issue_utilization,
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            "compute_throughput": math.isclose(
                direct_effective_throughput,
                effective_flops / point["schedule"]["compute_cycles"],
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            "exact_peak": exact_peak_effective_ops_per_cycle
            == physical_pes * simd_width * effective_ops_per_fma,
            "nominal_peak_consistency": nominal_difference
            <= nominal_peak_relative_difference_limit,
            "paper_claim_null": point["paper_reproduction_claim"] is None,
        }
    )
    return point


__all__ = ["compose_corrected_point"]
