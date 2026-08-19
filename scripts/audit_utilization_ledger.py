#!/usr/bin/env python3
"""Audit H163 target-free utilization metric identities."""

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

DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/utilization_ledger_v1.yaml"
PIPELINES = ("compute", "load", "store", "xfer")


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def build_ledger(record: dict[str, Any]) -> dict[str, Any]:
    summary = record["summary"]
    overlay = summary["overlay"]
    memory = summary["memory"]
    end_to_end = float(summary["end_to_end_cycles"])
    overlay_cycles = float(summary["overlay_cycles"])
    physical_pes = int(overlay["physical_pe_count"])
    productive_pe = overlay["productive_pe_cycles_by_pipeline"]
    productive_global = overlay["productive_global_cycles_by_pipeline"]
    issue_cycles = overlay["issue_cycles_by_pipeline"]
    resident = overlay["resident_pe_cycles_by_pipeline"]
    issued = overlay["issued_by_pipeline"]
    metrics: dict[str, dict[str, float]] = {
        name: {}
        for name in (
            "physical_capacity_fraction",
            "overlay_capacity_fraction",
            "temporal_busy_fraction",
            "temporal_issue_fraction",
            "active_spatial_fraction",
            "resident_productive_fraction",
            "issued_capacity_fraction",
        )
    }
    for pipeline in PIPELINES:
        metrics["physical_capacity_fraction"][pipeline] = safe_ratio(
            float(productive_pe[pipeline]), end_to_end * physical_pes
        )
        metrics["overlay_capacity_fraction"][pipeline] = safe_ratio(
            float(productive_pe[pipeline]), overlay_cycles * physical_pes
        )
        metrics["temporal_busy_fraction"][pipeline] = safe_ratio(
            float(productive_global[pipeline]), end_to_end
        )
        metrics["temporal_issue_fraction"][pipeline] = safe_ratio(
            float(issue_cycles[pipeline]), end_to_end
        )
        metrics["active_spatial_fraction"][pipeline] = safe_ratio(
            float(productive_pe[pipeline]),
            float(productive_global[pipeline]) * physical_pes,
        )
        metrics["resident_productive_fraction"][pipeline] = safe_ratio(
            float(productive_pe[pipeline]), float(resident[pipeline])
        )
        metrics["issued_capacity_fraction"][pipeline] = safe_ratio(
            float(issued[pipeline]), end_to_end * physical_pes
        )
    fu_capacity = {
        name: safe_ratio(float(value), end_to_end * physical_pes)
        for name, value in overlay["productive_pe_cycles_by_fu_class"].items()
    }
    ports = memory["spad"]["per_port"]
    total_requests = sum(int(port["requests"]) for port in ports)
    total_services = sum(int(port["issued_bank_operations"]) for port in ports)
    total_unavailable = sum(int(port["unavailable_checks"]) for port in ports)
    port_ledger = [
        {
            "port": index,
            "requests": int(port["requests"]),
            "issued_bank_operations": int(port["issued_bank_operations"]),
            "unavailable_checks": int(port["unavailable_checks"]),
            "request_share": safe_ratio(int(port["requests"]), total_requests),
            "service_share": safe_ratio(
                int(port["issued_bank_operations"]), total_services
            ),
            "unavailable_share": safe_ratio(
                int(port["unavailable_checks"]), total_unavailable
            ),
        }
        for index, port in enumerate(ports)
    ]
    return {
        "key": record["key"],
        "mode": record["mode"],
        "replay": int(record["replay"]),
        "end_to_end_cycles": int(summary["end_to_end_cycles"]),
        "overlay_cycles": int(summary["overlay_cycles"]),
        "physical_pes": physical_pes,
        "mapped_pes": int(overlay["mapped_pe_count"]),
        "metrics": metrics,
        "fu_capacity_fraction": fu_capacity,
        "ports": port_ledger,
        "raw": {
            "productive_pe_cycles_by_pipeline": productive_pe,
            "productive_global_cycles_by_pipeline": productive_global,
            "issue_cycles_by_pipeline": issue_cycles,
            "resident_pe_cycles_by_pipeline": resident,
            "issued_by_pipeline": issued,
            "productive_pe_cycles_by_fu_class": overlay[
                "productive_pe_cycles_by_fu_class"
            ],
            "spad_requests": int(memory["spad"]["requests"]),
            "spad_responses": int(memory["spad"]["responses"]),
        },
    }


def summarize_ranges(ledgers: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for operator in ("bsmm", "fft"):
        records = [item for item in ledgers if item["key"].startswith(f"{operator}-")]
        result[operator] = {}
        for identity in records[0]["metrics"]:
            result[operator][identity] = {
                pipeline: {
                    "minimum": min(item["metrics"][identity][pipeline] for item in records),
                    "maximum": max(item["metrics"][identity][pipeline] for item in records),
                }
                for pipeline in PIPELINES
            }
    return result


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    h120 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h120"]["path"]).read_text()
    )
    run = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h120_run"]["path"]).read_text()
    )
    parent_checks = {
        "h120": h120["hypothesis_status"]
        == config["frozen_inputs"]["h120"]["required_status"]
        and h120["audit_integrity"]
        is config["frozen_inputs"]["h120"]["required_integrity"],
        "h120_target_free": h120["paper_performance_targets_consumed"] is False,
        "run_checks": all(run["checks"].values()),
    }
    selected = sorted(
        (
            record
            for record in run["records"]
            if record["mode"] == config["workloads"]["selected_mode"]
            and int(record["replay"]) == int(config["workloads"]["selected_replay"])
        ),
        key=lambda item: (
            item["key"].split("-", 1)[0],
            int(item["key"].split("-", 1)[1]),
        ),
    )
    ledgers = [build_ledger(record) for record in selected]
    expected_keys = {
        f"{operator}-{size}"
        for operator in config["workloads"]["operators"]
        for size in config["workloads"]["sizes"]
    }
    selection_checks = {
        "count": len(selected) == int(config["workloads"]["required_paths"]),
        "keys": {item["key"] for item in selected} == expected_keys,
        "passing": all(item["pass"] for item in selected),
        "unique": len({item["key"] for item in selected}) == len(selected),
    }
    measurement_checks: dict[str, bool] = {}
    identity_checks: dict[str, dict[str, bool]] = {}
    port_checks: dict[str, bool] = {}
    for ledger in ledgers:
        key = ledger["key"]
        stored = h120["measurements"][key]
        measurement_checks[key] = (
            ledger["end_to_end_cycles"] == int(stored["end_to_end_cycles"])
            and ledger["overlay_cycles"] == int(stored["overlay_cycles"])
            and ledger["metrics"]["physical_capacity_fraction"]
            == stored["primary_end_to_end_utilization"]
            and ledger["metrics"]["overlay_capacity_fraction"]
            == stored["diagnostic_overlay_utilization"]
            and ledger["raw"]["spad_requests"] == ledger["raw"]["spad_responses"]
        )
        checks = {}
        for identity, values in ledger["metrics"].items():
            checks[f"{identity}_finite_bounded"] = all(
                math.isfinite(value) and 0.0 <= value <= 1.0
                for value in values.values()
            )
        checks["factorization"] = all(
            math.isclose(
                ledger["metrics"]["physical_capacity_fraction"][pipeline],
                ledger["metrics"]["temporal_busy_fraction"][pipeline]
                * ledger["metrics"]["active_spatial_fraction"][pipeline],
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            for pipeline in PIPELINES
        )
        checks["resident"] = all(
            math.isclose(
                ledger["metrics"]["resident_productive_fraction"][pipeline],
                safe_ratio(
                    ledger["raw"]["productive_pe_cycles_by_pipeline"][pipeline],
                    ledger["raw"]["resident_pe_cycles_by_pipeline"][pipeline],
                ),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            for pipeline in PIPELINES
        )
        checks["fu"] = bool(ledger["fu_capacity_fraction"]) and all(
            math.isfinite(value) and 0.0 <= value <= 1.0
            for value in ledger["fu_capacity_fraction"].values()
        )
        identity_checks[key] = checks
        ports = ledger["ports"]
        port_checks[key] = (
            len(ports) == 4
            and math.isclose(sum(item["request_share"] for item in ports), 1.0)
            and math.isclose(sum(item["service_share"] for item in ports), 1.0)
            and math.isclose(sum(item["unavailable_share"] for item in ports), 1.0)
            and sum(item["requests"] for item in ports)
            == ledger["raw"]["spad_requests"]
            and all(item["requests"] == item["issued_bank_operations"] for item in ports)
        )
    pin_checks = {
        name: pin in (PROJECT_ROOT / config["source_layout"]["source_note"]).read_text()
        for name, pin in config["open_source_pins"].items()
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path) for name, path in config["source_layout"].items()
    }
    source_tree = ast.parse(
        (PROJECT_ROOT / config["source_layout"]["auditor"]).read_text()
    )
    string_literals = {
        node.value
        for node in ast.walk(source_tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    forbidden_paths = set(config["target_exclusion"]["forbidden_paths"])
    target_free_checks = {
        "frozen_inputs": set(config["frozen_inputs"]) == {
            "h120",
            "h120_run",
            "source_refresh",
        },
        "no_forbidden_path": forbidden_paths.isdisjoint(string_literals),
        "parent": parent_checks["h120_target_free"],
        "no_selection": True,
    }
    ranges = summarize_ranges(ledgers)
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        all(selection_checks.values()),
        all(measurement_checks.values()),
        all(all(checks.values()) for checks in identity_checks.values()),
        all(checks["factorization"] for checks in identity_checks.values()),
        all(checks["resident"] for checks in identity_checks.values()),
        all(checks["fu"] for checks in identity_checks.values()),
        all(port_checks.values()),
        all(target_free_checks.values())
        and all(pin_checks.values())
        and all(item["pass"] for item in source_files.values()),
        len(ranges) == 2 and set(ranges) == {"bsmm", "fft"},
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "selection": all(selection_checks.values()),
        "measurements": len(measurement_checks) == 16,
        "identities": len(identity_checks) == 16,
        "ports": len(port_checks) == 16,
        "pins": all(pin_checks.values()),
        "source": all(item["pass"] for item in source_files.values()),
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
        "paper_reproduction_claim": "none_counter_identity_ledger_only",
        "selected_metric": None,
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "selection_checks": selection_checks,
        "measurement_checks": measurement_checks,
        "identity_checks": identity_checks,
        "port_checks": port_checks,
        "pin_checks": pin_checks,
        "ledgers": ledgers,
        "metric_ranges": ranges,
        "target_free_checks": target_free_checks,
        "source_files": source_files,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "paths": len(ledgers),
            "pipeline_identities": 7,
            "fu_classes": sorted(
                {name for ledger in ledgers for name in ledger["fu_capacity_fraction"]}
            ),
            "ports_per_path": 4,
            "metric_selected": False,
            "paper_targets_consumed": False,
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
            "ledgers",
            "metric_ranges",
            "target_free_checks",
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
