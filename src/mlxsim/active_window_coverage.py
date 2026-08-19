"""Compile H120 workloads across source-constrained active-tag windows."""

from __future__ import annotations

from typing import Any

from mlxsim.fig22_coupled_multiport import compile_fig22_coupled_multiport


def compile_active_window_path(
    operator: str,
    size: int,
    window: int,
    sweep_config: dict[str, Any],
    h120_config: dict[str, Any],
    h118_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Compile one path while changing only H118's active-window field."""

    active_config = {
        **h118_config,
        "hardware": {**h118_config["hardware"], "active_window": int(window)},
    }
    overlay, memory, parent = compile_fig22_coupled_multiport(
        operator, int(size), h120_config, active_config
    )
    sweep = sweep_config["window_sweep"]
    hardware = sweep_config["hardware_invariants"]
    footprint = int(parent["parent"]["max_active_instruction_footprint_per_pe"])
    instruction_slots = int(sweep["instruction_slots_per_pe"])
    globally_feasible = int(window) in {
        int(value) for value in sweep["globally_feasible_windows"]
    }
    metadata = {
        "key": f"w{int(window)}--{operator}-{int(size)}",
        "workload_key": f"{operator}-{int(size)}",
        "operator": operator,
        "size": int(size),
        "window": int(window),
        "footprint": footprint,
        "instruction_slots_per_pe": instruction_slots,
        "path_capacity_feasible": footprint <= instruction_slots,
        "globally_feasible_window": globally_feasible,
        "parent": parent,
        "paper_performance_targets_consumed": False,
        "checks": {
            "active_window": int(overlay["active_window"]) == int(window),
            "footprint": footprint
            == int(parent["parent"]["max_active_instruction_footprint_per_pe"]),
            "path_capacity": parent["parent"]["checks"]["instruction_capacity"]
            is (footprint <= instruction_slots),
            "active_blocks_bound": int(window)
            <= int(sweep["active_blocks_per_pe"]),
            "mesh": parent["parent"]["checks"]["active_window"]
            and h118_config["hardware"]["mesh"] == hardware["mesh"],
            "simd": int(h118_config["hardware"]["simd_width"])
            == int(hardware["simd_width"]),
            "ports": int(memory["spad_ports"]) == int(hardware["spad_ports"]),
            "banks": int(memory["spad_ports"]) * int(memory["spad"]["banks"])
            == int(hardware["spad_total_banks"]),
            "issue_width": int(memory["spad_ports"])
            * int(memory["spad"]["issue_width"])
            == int(hardware["spad_total_issue_width"]),
            "dma": int(memory["dma_bytes_per_cycle"])
            == int(hardware["dma_bytes_per_cycle"])
            and int(memory["dma_setup_cycles"])
            == int(hardware["dma_setup_cycles"]),
            "target_free": overlay["metadata"][
                "paper_performance_targets_consumed"
            ]
            is False
            and memory["metadata"]["paper_performance_targets_consumed"] is False,
        },
    }
    return overlay, memory, metadata


__all__ = ["compile_active_window_path"]
