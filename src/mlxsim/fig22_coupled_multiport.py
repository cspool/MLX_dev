"""Compile H118 workloads with H69's diagram-derived port topology."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from mlxsim.fig22_coupled_workloads import compile_fig22_coupled_workload


def compile_fig22_coupled_multiport(
    operator: str,
    size: int,
    config: dict[str, Any],
    h118_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    overlay, one_port, parent, _ = compile_fig22_coupled_workload(
        operator, size, h118_config
    )
    candidate = config["candidate"]
    memory = deepcopy(one_port)
    memory["spad_ports"] = int(candidate["ports"])
    memory["spad_port_axis"] = candidate["operator_axis"][operator]
    memory["spad"] = deepcopy(candidate["per_port_spad"])
    memory["metadata"].update(
        {
            "experiment_id": config["experiment_id"],
            "parent_experiment_id": "H118",
            "port_topology_parent": "H69",
            "paper_performance_targets_consumed": False,
        }
    )
    ports = int(candidate["ports"])
    per_port = candidate["per_port_spad"]
    metadata = {
        "key": f"{operator}-{size}",
        "operator": operator,
        "size": int(size),
        "ports": ports,
        "axis": candidate["operator_axis"][operator],
        "total_banks": ports * int(per_port["banks"]),
        "total_issue_width": ports * int(per_port["issue_width"]),
        "parent": parent,
        "paper_performance_targets_consumed": False,
        "checks": {
            "banks_partition": ports * int(per_port["banks"])
            == int(candidate["total_banks"]),
            "issue_partition": ports * int(per_port["issue_width"])
            == int(candidate["total_issue_width"]),
            "port_count": ports == 4,
            "axis": candidate["operator_axis"][operator]
            == ("x" if operator == "bsmm" else "y"),
            "overlay_target_free": overlay["metadata"][
                "paper_performance_targets_consumed"
            ]
            is False,
            "memory_target_free": memory["metadata"][
                "paper_performance_targets_consumed"
            ]
            is False,
        },
    }
    return overlay, memory, metadata


__all__ = ["compile_fig22_coupled_multiport"]
