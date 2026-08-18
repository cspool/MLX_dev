#!/usr/bin/env python3
"""Audit H123's exact-FMA GPGPU-Sim schedule witness."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/fig24_gpu_schedule_ambiguity_v1.yaml"


def last_integer(text: str, name: str) -> int | None:
    matches = re.findall(rf"^{re.escape(name)} = ([0-9]+)$", text, re.MULTILINE)
    return int(matches[-1]) if matches else None


def parse_run(path: Path) -> dict[str, Any]:
    text = path.read_text(errors="replace")
    summaries = re.findall(r"^MLX_FIG24_SCHEDULE_SUMMARY (\{.*\})$", text, re.MULTILINE)
    summary = json.loads(summaries[-1]) if summaries else None
    return {
        "summary": summary,
        "cycles": last_integer(text, "gpu_tot_sim_cycle"),
        "instructions": last_integer(text, "gpu_tot_sim_insn"),
        "ctas": last_integer(text, "gpu_tot_issued_cta"),
        "detailed": "GPGPU-Sim uArch: performance model detailed simulation" in text,
        "normal_exit": "GPGPU-Sim: *** exit detected ***" in text,
    }


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    h100 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h100"]["path"]).read_text()
    )
    h101 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h101"]["path"]).read_text()
    )
    h54 = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h54"]["path"]).read_text()
    )
    manifest = json.loads(
        (PROJECT_ROOT / config["frozen_inputs"]["h101_manifest"]["path"]).read_text()
    )
    parent_checks = {
        name: parent["hypothesis_status"] == spec["required_status"]
        and parent["audit_integrity"] is spec["required_integrity"]
        for name, parent in (("h100", h100), ("h101", h101), ("h54", h54))
        for spec in [config["frozen_inputs"][name]]
    }
    witness = config["witness"]
    contract = manifest["path_contracts"][witness["path_key"]]
    contract_checks = {
        "family": contract["family"] == witness["family"],
        "stages": contract["actual"]["stage_count"] == witness["stages"],
        "full_fma": contract["actual"]["fu"]["fma"]
        == witness["full_scalar_fma"],
        "case": contract["case"]["name"] == "BERT_512"
        and contract["operator"]["block_size"] == 16,
    }
    arithmetic_checks = {
        "witness_fraction": witness["witness_scalar_fma"]
        * witness["fractional_denominator"]
        == witness["full_scalar_fma"],
        "kernel_work": witness["element_count"]
        * witness["stages"]
        * witness["fma_per_element_stage"]
        == witness["witness_scalar_fma"],
        "cta_formula": witness["expected_total_ctas"]
        == [
            witness["stages"]
            * math.ceil(witness["element_count"] / block)
            for block in witness["block_threads"]
        ],
    }

    output_root = PROJECT_ROOT / config["output_root"]
    binary_record = output_root / "binary-sha256.txt"
    binary_check = binary_record.is_file() and bool(
        re.fullmatch(r"[0-9a-f]{64}  .+\n?", binary_record.read_text())
    )
    runs: dict[str, Any] = {}
    run_checks: dict[str, bool] = {}
    summaries = []
    for index, block in enumerate(witness["block_threads"]):
        root = output_root / f"block{block}"
        log_path = root / "run.log"
        parsed = parse_run(log_path)
        artifact = qualify(log_path)
        summary = parsed["summary"] or {}
        checks = {
            "artifact": artifact["pass"],
            "summary": bool(summary),
            "shape": summary.get("count") == witness["element_count"]
            and summary.get("stages") == witness["stages"]
            and summary.get("block_threads") == block,
            "work": summary.get("scalar_fma") == witness["witness_scalar_fma"],
            "ctas": summary.get("total_ctas")
            == witness["expected_total_ctas"][index]
            == parsed["ctas"],
            "checksum": summary.get("relative_error", math.inf)
            <= config["acceptance"]["checksum_relative_error_limit"],
            "cycles": isinstance(parsed["cycles"], int) and parsed["cycles"] > 0,
            "instructions": isinstance(parsed["instructions"], int)
            and parsed["instructions"] > 0,
            "detailed": parsed["detailed"],
            "exit": parsed["normal_exit"],
            "config": hashlib.sha256((root / "gpgpusim.config").read_bytes()).hexdigest()
            == config["frozen_inputs"]["orin_config"]["sha256"],
            "interconnect": hashlib.sha256(
                (root / "config_ampere_islip.icnt").read_bytes()
            ).hexdigest()
            == config["frozen_inputs"]["orin_interconnect"]["sha256"],
        }
        run_checks[str(block)] = all(checks.values())
        runs[str(block)] = {"artifact": artifact, "parsed": parsed, "checks": checks}
        summaries.append(summary)
    checksum_values = [float(summary["checksum"]) for summary in summaries]
    checksum_cross_check = max(checksum_values) - min(checksum_values) <= 1e-9
    cycles = [int(item["parsed"]["cycles"]) for item in runs.values()]
    cycle_spread = max(cycles) / min(cycles) - 1
    schedule_sensitive = cycle_spread > config["acceptance"]["minimum_cycle_spread"]

    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    source_text = "\n".join(
        (PROJECT_ROOT / path).read_text(errors="replace")
        for path in config["source_layout"].values()
    )
    target_free_checks = {
        "config": config["acceptance"]["targets_consumed"] is False,
        "no_targets": "fig24_structured_sweep" not in source_text,
        "no_ratio": "mlx_over_orin" not in source_text,
        "no_residual": "residual" + "_factor" not in source_text,
    }
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()) and all(parent_checks.values()),
        all(contract_checks.values()),
        all(arithmetic_checks.values()),
        binary_check and len(runs) == 3,
        all(run_checks.values()),
        checksum_cross_check
        and all(item["checks"]["checksum"] for item in runs.values()),
        len({summary["scalar_fma"] for summary in summaries}) == 1,
        schedule_sensitive,
        all(target_free_checks.values()) and all(item["pass"] for item in source_files.values()),
        config["acceptance"]["figure24_denominator_identified_expected"] is False,
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "parents": all(parent_checks.values()),
        "contract": all(contract_checks.values()),
        "arithmetic": all(arithmetic_checks.values()),
        "binary": binary_check,
        "runs": all(run_checks.values()),
        "checksums": checksum_cross_check,
        "source": all(target_free_checks.values())
        and all(item["pass"] for item in source_files.values()),
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
        "paper_reproduction_claim": "none_target_free_gpu_schedule_witness_only",
        "frozen_inputs": frozen,
        "parent_checks": parent_checks,
        "contract_checks": contract_checks,
        "arithmetic_checks": arithmetic_checks,
        "binary_check": binary_check,
        "runs": runs,
        "run_checks": run_checks,
        "checksum_cross_check": checksum_cross_check,
        "cycle_spread": cycle_spread,
        "schedule_sensitive": schedule_sensitive,
        "target_free_checks": target_free_checks,
        "acceptance_gates": acceptance_gates,
        "summary": {
            "runs": len(runs),
            "scalar_fma_per_run": witness["witness_scalar_fma"],
            "minimum_cycles": min(cycles),
            "maximum_cycles": max(cycles),
            "cycle_spread": cycle_spread,
            "schedule_sensitive": schedule_sensitive,
            "acceptance_gates_passed": sum(acceptance_gates),
            "acceptance_gates_total": len(acceptance_gates),
            "active_simulator_figures_reproduced": 0,
            "active_simulator_figures_total": 8,
        },
        "source_files": source_files,
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
            "runs",
            "cycle_spread",
            "schedule_sensitive",
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
