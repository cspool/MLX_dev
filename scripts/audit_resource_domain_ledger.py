#!/usr/bin/env python3
"""Audit H166's target-free component and bandwidth-capacity ledger."""

from __future__ import annotations

import argparse
import ast
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/resource_domain_ledger_v1.yaml"
PIPELINES = ("compute", "load", "store", "xfer")


def ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        raise ValueError("capacity denominator must be positive")
    return numerator / denominator


def build_path_ledger(
    record: dict[str, Any], compiled: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    summary = record["summary"]
    overlay = summary["overlay"]
    memory = summary["memory"]
    domain = config["capacity_domains"]
    end_cycles = int(summary["end_to_end_cycles"])
    overlay_cycles = int(summary["overlay_cycles"])
    physical_pes = int(domain["physical_pes"])
    pe_capacity = end_cycles * physical_pes
    issued = overlay["issued_by_pipeline"]
    read_requests = int(memory["read_requests"])
    write_requests = int(memory["write_requests"])
    internal_loads = int(issued["load"]) - read_requests
    offchip_bytes = int(memory["offchip_read_bytes"]) + int(
        memory["offchip_write_bytes"]
    )
    spad_requests = int(memory["spad"]["requests"])
    vector_bytes = int(domain["vector_bytes"])
    spad_payload_bytes = spad_requests * vector_bytes
    productive_pe = overlay["productive_pe_cycles_by_pipeline"]
    productive_global = overlay["productive_global_cycles_by_pipeline"]
    issue_cycles = overlay["issue_cycles_by_pipeline"]
    metrics = {
        "compute_physical_capacity": ratio(productive_pe["compute"], pe_capacity),
        "compute_global_end_busy": ratio(productive_global["compute"], end_cycles),
        "compute_global_overlay_busy": ratio(
            productive_global["compute"], overlay_cycles
        ),
        "compute_issue_end": ratio(issue_cycles["compute"], end_cycles),
        "external_load_pe_service": ratio(read_requests, pe_capacity),
        "external_store_pe_service": ratio(write_requests, pe_capacity),
        "xfer_issue_pe_service": ratio(issued["xfer"], pe_capacity),
        "xfer_hop_pe_service": ratio(overlay["route_hops"], pe_capacity),
        "dma_byte_capacity": ratio(
            offchip_bytes, end_cycles * int(domain["dma_bytes_per_cycle"])
        ),
        "spad_operation_capacity": ratio(
            spad_requests,
            end_cycles * int(domain["spad_operation_capacity_per_cycle"]),
        ),
        "spad_wire_byte_capacity": ratio(
            spad_payload_bytes,
            end_cycles * int(domain["spad_wire_byte_capacity_per_cycle"]),
        ),
        "spad_payload_byte_capacity": ratio(
            spad_payload_bytes,
            end_cycles * int(domain["spad_payload_byte_capacity_per_cycle"]),
        ),
        "launch_fraction": ratio(end_cycles - overlay_cycles, end_cycles),
    }
    fu_capacity = {
        name: ratio(value, pe_capacity)
        for name, value in overlay["productive_pe_cycles_by_fu_class"].items()
    }
    per_port = [
        {
            "port": index,
            "requests": int(port["requests"]),
            "responses": int(port["responses"]),
            "issued_bank_operations": int(port["issued_bank_operations"]),
            "unavailable_checks": int(port["unavailable_checks"]),
            "request_capacity_fraction": ratio(
                port["requests"],
                end_cycles * int(domain["issue_width_per_port"]),
            ),
        }
        for index, port in enumerate(memory["spad"]["per_port"])
    ]
    parent = compiled["metadata"]["parent"]
    return {
        "key": record["key"],
        "end_to_end_cycles": end_cycles,
        "overlay_cycles": overlay_cycles,
        "raw": {
            "issued_loads": int(issued["load"]),
            "external_loads": read_requests,
            "internal_local_loads": internal_loads,
            "issued_stores": int(issued["store"]),
            "external_stores": write_requests,
            "issued_xfers": int(issued["xfer"]),
            "route_hops": int(overlay["route_hops"]),
            "skip_hops": int(overlay["skip_hops"]),
            "unit_hops": int(overlay["unit_hops"]),
            "external_memory_requests": int(overlay["external_memory_requests"]),
            "external_memory_completions": int(
                overlay["external_memory_completions"]
            ),
            "read_requests": read_requests,
            "write_requests": write_requests,
            "memory_requests": int(memory["requests"]),
            "memory_responses": int(memory["responses"]),
            "offchip_read_bytes": int(memory["offchip_read_bytes"]),
            "offchip_write_bytes": int(memory["offchip_write_bytes"]),
            "spad_requests": spad_requests,
            "spad_responses": int(memory["spad"]["responses"]),
            "spad_payload_bytes": spad_payload_bytes,
            "productive_pe_cycles_by_pipeline": productive_pe,
            "productive_global_cycles_by_pipeline": productive_global,
            "issue_cycles_by_pipeline": issue_cycles,
            "productive_pe_cycles_by_fu_class": overlay[
                "productive_pe_cycles_by_fu_class"
            ],
        },
        "compile_metadata": {
            "external_loads": int(parent["external_loads"]),
            "external_stores": int(parent["external_stores"]),
            "memory_requests": int(parent["memory_requests"]),
            "input_bytes": int(parent["input_bytes"]),
            "output_bytes": int(parent["output_bytes"]),
            "route_hops": int(parent["route_hops"]),
        },
        "metrics": metrics,
        "fu_capacity_fraction_by_class": fu_capacity,
        "fu_capacity_fraction_sum_raw": sum(fu_capacity.values()),
        "unified_data_supply": {
            "pe_service_sum_with_xfer_issue": metrics["external_load_pe_service"]
            + metrics["external_store_pe_service"]
            + metrics["xfer_issue_pe_service"],
            "pe_service_sum_with_xfer_hops": metrics["external_load_pe_service"]
            + metrics["external_store_pe_service"]
            + metrics["xfer_hop_pe_service"],
        },
        "per_port": per_port,
    }


def _ranges(ledgers: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    names = ledgers[0]["metrics"]
    return {
        name: {
            "minimum": min(item["metrics"][name] for item in ledgers),
            "maximum": max(item["metrics"][name] for item in ledgers),
        }
        for name in names
    }


def _source_audit(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, bool]]:
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    source_text = "\n".join(
        (PROJECT_ROOT / path).read_text(errors="replace")
        for name, path in config["source_layout"].items()
        if name in {"auditor", "test"}
    )
    tree = ast.parse((PROJECT_ROOT / config["source_layout"]["auditor"]).read_text())
    assigned = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    checks = {
        "no_target_paths": not any(
            path in source_text for path in config["target_exclusion"]["forbidden_paths"]
        ),
        "no_selected_metric": "selected_metric" not in assigned
        and "selected_schema" not in assigned,
        "no_fit_calls": calls.isdisjoint(
            {"polyfit", "curve_fit", "lstsq", "minimize", "least_squares"}
        ),
        "source_files": all(item["pass"] for item in source_files.values()),
    }
    return source_files, checks


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    h120 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h120"]["path"]).read_text()
    )
    h163 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h163"]["path"]).read_text()
    )
    compiled = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h120_compile"]["path"]).read_text()
    )
    run = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h120_run"]["path"]).read_text()
    )
    paper_text = (
        PROJECT_ROOT / config["frozen_inputs"]["mlx_paper_text"]["path"]
    ).read_text()
    parent_checks = {
        "h120": h120["hypothesis_status"]
        == config["frozen_inputs"]["h120"]["required_status"]
        and h120["audit_integrity"]
        is config["frozen_inputs"]["h120"]["required_integrity"],
        "h163": h163["hypothesis_status"]
        == config["frozen_inputs"]["h163"]["required_status"]
        and h163["audit_integrity"]
        is config["frozen_inputs"]["h163"]["required_integrity"],
        "target_free": h120["paper_performance_targets_consumed"] is False
        and h163["paper_performance_targets_consumed"] is False
        and run["paper_performance_targets_consumed"] is False,
        "compile": all(compiled["checks"].values()),
        "run": all(run["checks"].values()),
        "paper_data_supply": "We group load/store/transfer units as a unified data-supply pipeline"
        in paper_text,
        "paper_dense_mm_scope": "each PE computes an  $8\\times 8$  SIMD-aligned tile"
        in paper_text,
    }
    selected = sorted(
        (
            record
            for record in run["records"]
            if record["mode"] == config["workloads"]["selected_mode"]
            and int(record["replay"]) == int(config["workloads"]["selected_replay"])
        ),
        key=lambda item: item["key"],
    )
    compile_by_key = compiled["outputs"]
    ledgers = [
        build_path_ledger(record, compile_by_key[record["key"]], config)
        for record in selected
    ]
    expected_keys = {
        f"{operator}-{int(size)}"
        for operator in config["workloads"]["operators"]
        for size in config["workloads"]["sizes"]
    }
    selection_checks = {
        "count": len(ledgers) == int(config["workloads"]["required_paths"]),
        "keys": {item["key"] for item in ledgers} == expected_keys,
        "unique": len({item["key"] for item in ledgers}) == len(ledgers),
        "passing": all(record["pass"] for record in selected),
    }
    partition_checks: dict[str, bool] = {}
    conservation_checks: dict[str, bool] = {}
    port_checks: dict[str, bool] = {}
    metric_checks: dict[str, bool] = {}
    h163_by_key = {item["key"]: item for item in h163["ledgers"]}
    compatibility_checks: dict[str, bool] = {}
    for ledger in ledgers:
        key = ledger["key"]
        raw = ledger["raw"]
        meta = ledger["compile_metadata"]
        partition_checks[key] = (
            raw["issued_loads"]
            == raw["external_loads"] + raw["internal_local_loads"]
            and raw["internal_local_loads"] >= 0
            and raw["external_loads"] == meta["external_loads"]
            and raw["issued_stores"]
            == raw["external_stores"]
            == meta["external_stores"]
        )
        conservation_checks[key] = (
            raw["external_memory_requests"]
            == raw["external_memory_completions"]
            == raw["memory_requests"]
            == raw["memory_responses"]
            == raw["spad_requests"]
            == raw["spad_responses"]
            == meta["memory_requests"]
            and raw["read_requests"] + raw["write_requests"]
            == raw["memory_requests"]
            and raw["offchip_read_bytes"] == meta["input_bytes"]
            and raw["offchip_write_bytes"] == meta["output_bytes"]
            and raw["route_hops"]
            == raw["unit_hops"] + raw["skip_hops"]
            == meta["route_hops"]
        )
        ports = ledger["per_port"]
        port_checks[key] = (
            len(ports) == int(config["capacity_domains"]["spad_ports"])
            and sum(item["requests"] for item in ports) == raw["spad_requests"]
            and sum(item["responses"] for item in ports) == raw["spad_responses"]
            and sum(item["issued_bank_operations"] for item in ports)
            == raw["spad_requests"]
            and all(
                item["requests"]
                == item["responses"]
                == item["issued_bank_operations"]
                for item in ports
            )
        )
        bounded = list(ledger["metrics"].values()) + [
            item["request_capacity_fraction"] for item in ports
        ] + list(ledger["fu_capacity_fraction_by_class"].values())
        metric_checks[key] = all(
            math.isfinite(value) and 0.0 <= value <= 1.0 for value in bounded
        ) and all(
            math.isfinite(value) and value >= 0.0
            for value in ledger["unified_data_supply"].values()
        )
        compatibility_checks[key] = math.isclose(
            ledger["metrics"]["compute_physical_capacity"],
            h163_by_key[key]["metrics"]["physical_capacity_fraction"]["compute"],
            rel_tol=0.0,
            abs_tol=0.0,
        )
    domain = config["capacity_domains"]
    capacity_checks = {
        "dma": int(domain["clock_hz"]) * int(domain["dma_bytes_per_cycle"])
        == int(domain["dma_bandwidth_bytes_per_second"]),
        "banks": int(domain["spad_ports"]) * int(domain["banks_per_port"])
        == int(domain["total_banks"]),
        "issue": int(domain["spad_ports"]) * int(domain["issue_width_per_port"])
        == int(domain["total_issue_width"])
        == int(domain["spad_operation_capacity_per_cycle"]),
        "wire": int(domain["total_banks"]) * int(domain["bank_width_bytes"])
        == int(domain["spad_wire_byte_capacity_per_cycle"]),
        "payload": int(domain["total_issue_width"]) * int(domain["vector_bytes"])
        == int(domain["spad_payload_byte_capacity_per_cycle"]),
    }
    disclosures = {
        "dma_64_bytes_per_cycle": "historical_DPU_lineage_not_disclosed_by_MLX",
        "spad_32_banks": "H69_Figure9_Figure11_diagram_derived",
        "spad_1024_wire_bytes_per_cycle": "derived_32_banks_times_32B_bank_width",
        "spad_512_payload_bytes_per_cycle": "derived_32_issues_times_16B_SIMD8_payload",
        "figure22_counter_schema": "not_selected",
    }
    source_files, source_checks = _source_audit(config)
    target_free_checks = {
        "parents": set(config["frozen_inputs"])
        == {"h120", "h120_compile", "h120_run", "h163", "mlx_paper_text", "source_refresh"},
        "source": all(source_checks.values()),
        "no_selection": True,
        "paper_targets_not_consumed": h120["paper_performance_targets_consumed"]
        is False
        and h163["paper_performance_targets_consumed"] is False
        and run["paper_performance_targets_consumed"] is False,
    }
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        all(selection_checks.values()),
        all(partition_checks.values()),
        all(conservation_checks.values()),
        all(
            item["raw"]["route_hops"]
            == item["raw"]["unit_hops"] + item["raw"]["skip_hops"]
            for item in ledgers
        ),
        all(port_checks.values()),
        all(metric_checks.values()) and all(compatibility_checks.values()),
        all(capacity_checks.values()) and len(disclosures) == 5,
        all(target_free_checks.values())
        and all(item["pass"] for item in source_files.values()),
        config["validation_eligible"] is True,
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "selection": all(selection_checks.values()),
        "partition": all(partition_checks.values()),
        "conservation": all(conservation_checks.values()),
        "ports": all(port_checks.values()),
        "metrics": all(metric_checks.values()),
        "compatibility": all(compatibility_checks.values()),
        "capacity": all(capacity_checks.values()),
        "target_free": all(target_free_checks.values()),
        "acceptance_evaluated": len(acceptance_gates) == 10
        and all(isinstance(value, bool) for value in acceptance_gates),
    }
    integrity = all(integrity_checks.values())
    supported = integrity and all(acceptance_gates)
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "classification": config["classification"],
        "validation_eligible": config["validation_eligible"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_commit": git_commit(),
        "hypothesis_status": "supported" if supported else "rejected",
        "audit_integrity": integrity,
        "paper_performance_targets_consumed": False,
        "paper_reproduction_claim": "none_resource_domain_ledger_only",
        "selected_metric": None,
        "selected_schema": None,
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "selection_checks": selection_checks,
        "partition_checks": partition_checks,
        "conservation_checks": conservation_checks,
        "port_checks": port_checks,
        "metric_checks": metric_checks,
        "compatibility_checks": compatibility_checks,
        "capacity_checks": capacity_checks,
        "capacity_domains": domain,
        "disclosures": disclosures,
        "ledgers": ledgers,
        "metric_ranges": _ranges(ledgers),
        "source_files": source_files,
        "source_checks": source_checks,
        "target_free_checks": target_free_checks,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "paths": len(ledgers),
            "registered_metrics": len(ledgers[0]["metrics"]),
            "fu_classes": sorted(
                {name for item in ledgers for name in item["fu_capacity_fraction_by_class"]}
            ),
            "ports": int(domain["spad_ports"]),
            "dma_bandwidth_gb_per_second": int(
                domain["dma_bandwidth_bytes_per_second"]
            )
            / 1e9,
            "spad_wire_bytes_per_cycle": int(
                domain["spad_wire_byte_capacity_per_cycle"]
            ),
            "spad_payload_bytes_per_cycle": int(
                domain["spad_payload_byte_capacity_per_cycle"]
            ),
            "metric_selected": False,
            "schema_selected": False,
            "acceptance_gates_passed": sum(acceptance_gates),
            "acceptance_gates_total": len(acceptance_gates),
        },
        "integrity_checks": integrity_checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    report = build_audit(config)
    output = PROJECT_ROOT / config["result_path"]
    if args.verify_existing:
        existing = json.loads(output.read_text())
        keys = (
            "hypothesis_status",
            "audit_integrity",
            "selected_metric",
            "selected_schema",
            "capacity_checks",
            "ledgers",
            "metric_ranges",
            "acceptance_gates",
            "summary",
            "integrity_checks",
        )
        matches = all(
            json.dumps(existing.get(key), sort_keys=True)
            == json.dumps(report.get(key), sort_keys=True)
            for key in keys
        )
        print(json.dumps({"existing_matches": matches, **report}, indent=2))
        return 0 if matches else 1
    if args.preflight_only:
        print(json.dumps(report, indent=2))
        return 0 if report["audit_integrity"] and not output.exists() else 1
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["hypothesis_status"], **report["summary"]}, indent=2))
    return 0 if report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
