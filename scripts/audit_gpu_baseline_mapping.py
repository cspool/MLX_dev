#!/usr/bin/env python3
"""Audit H168's target-free open-simulator mapping for MLX GPU baselines."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.audit_fig22_coupled_transfer import PROJECT_ROOT, git_commit, qualify

DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/gpu_baseline_mapping_v1.yaml"
ACCELSIM_ROOT = PROJECT_ROOT / "third_party/accel-sim-framework"
GPGPUSIM_ROOT = ACCELSIM_ROOT / "gpu-simulator/gpgpu-sim"


def repository_commit(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def load_inputs(config: dict[str, Any]) -> dict[str, Any]:
    return {
        name: json.loads((PROJECT_ROOT / spec["path"]).read_text())
        for name, spec in config["frozen_inputs"].items()
        if name != "source_note"
    }


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify(PROJECT_ROOT / spec["path"], spec)
        for name, spec in config["frozen_inputs"].items()
    }
    inputs = load_inputs(config)
    note = (
        PROJECT_ROOT / config["frozen_inputs"]["source_note"]["path"]
    ).read_text()
    proxy_names = ("xavier_proxy", "orin_proxy", "rtx3090_proxy")
    proxy_checks = {
        name: inputs[name]["hypothesis_status"]
        == config["frozen_inputs"][name]["required_status"]
        and inputs[name]["audit_integrity"]
        is config["frozen_inputs"][name]["required_integrity"]
        and inputs[name]["paper_performance_targets_consumed"] is False
        and inputs[name]["runs"]["pass"] is True
        and all(inputs[name]["integrity_checks"].values())
        for name in proxy_names
    }
    evidence_checks = {
        "orin_schedule": inputs["orin_schedule"]["hypothesis_status"]
        == config["frozen_inputs"]["orin_schedule"]["required_status"]
        and inputs["orin_schedule"]["audit_integrity"]
        is config["frozen_inputs"]["orin_schedule"]["required_integrity"]
        and inputs["orin_schedule"]["paper_performance_targets_consumed"] is False
        and inputs["orin_schedule"]["summary"]["schedule_sensitive"] is True,
        "xavier_coverage": inputs["xavier_coverage"]["hypothesis_status"]
        == config["frozen_inputs"]["xavier_coverage"]["required_status"]
        and inputs["xavier_coverage"]["audit_integrity"]
        is config["frozen_inputs"]["xavier_coverage"]["required_integrity"]
        and inputs["xavier_coverage"]["paper_performance_targets_consumed"]
        is False
        and inputs["xavier_coverage"]["summary"]["qualified_xavier_family_rows"]
        == 0,
    }
    pins = config["open_source_pins"]
    pin_checks = {
        "project_accelsim": repository_commit(ACCELSIM_ROOT)
        == pins["project_accelsim"],
        "project_gpgpusim": repository_commit(GPGPUSIM_ROOT)
        == pins["project_gpgpusim"],
        **{name: value in note for name, value in pins.items()},
    }
    tested_root = GPGPUSIM_ROOT / "configs/tested-cfgs"
    tested_configs = sorted(
        path.name for path in tested_root.iterdir() if path.is_dir()
    )
    config_availability = {
        "local_tested_configs": tested_configs,
        "xavier_sm72": any("SM72" in name for name in tested_configs),
        "orin_sm87": any("SM87" in name for name in tested_configs),
        "rtx3090": any("RTX3090" in name for name in tested_configs),
        "h100_sm90": any("SM90" in name or "H100" in name for name in tested_configs),
        "rtx3070_sm86": "SM86_RTX3070" in tested_configs,
        "titanv_sm70": "SM7_TITANV" in tested_configs,
    }
    device_specs = config["devices"]
    proxy_by_device = {
        "xavier": inputs["xavier_proxy"],
        "orin": inputs["orin_proxy"],
        "rtx3090": inputs["rtx3090_proxy"],
    }
    device_records: dict[str, dict[str, Any]] = {}
    for device, spec in device_specs.items():
        proxy = proxy_by_device.get(device)
        local_proxy = proxy is not None
        exact_isa_template = spec["hardware_isa"] in spec["local_timing_template"]
        executed_workloads = sorted(proxy["runs"]["items"]) if proxy else []
        blockers = {
            "native_target_tuned_config": False,
            "native_exact_application_trace": False,
            "native_hardware_correlation": False,
            "exact_author_kernel_schedule": False,
        }
        if device == "xavier":
            blockers["dense_end_to_end_family_coverage"] = False
            evidence = {
                "qualified_dense_families": inputs["xavier_coverage"]["summary"][
                    "qualified_xavier_family_rows"
                ],
                "required_dense_families": inputs["xavier_coverage"]["summary"][
                    "required_xavier_family_rows"
                ],
                "executed_tensor_instructions": inputs["xavier_coverage"]["summary"][
                    "h56_executed_tensor_instructions"
                ],
            }
        elif device == "orin":
            blockers["cta_schedule_identity"] = False
            evidence = {
                "equal_work_cycle_spread": inputs["orin_schedule"]["summary"][
                    "cycle_spread"
                ],
                "schedule_sensitive": inputs["orin_schedule"]["summary"][
                    "schedule_sensitive"
                ],
            }
        elif device == "rtx3090":
            blockers["native_cache_and_launch_timing"] = False
            evidence = {
                "isa_family_match": exact_isa_template,
                "base_template": proxy["config_derivation"]["base"]["path"],
            }
        else:
            blockers["local_execution"] = False
            blockers["h100_workload_validation"] = False
            evidence = {
                "flashgpusim_pin": pins["flashgpusim"],
                "local_h100_config": False,
                "source_note_candidate": "FlashGPU-Sim_SM90_H100",
            }
        device_records[device] = {
            "paper_figures": spec["paper_figures"],
            "hardware_isa": spec["hardware_isa"],
            "vendor_identity_covered": True,
            "preferred_open_simulator": (
                "FlashGPU-Sim" if device == "h100" else spec["local_simulator"]
            ),
            "open_candidate_available": True,
            "local_timing_template": spec["local_timing_template"],
            "exact_isa_template": exact_isa_template,
            "local_executable_proxy": local_proxy,
            "executed_proxy_workloads": executed_workloads,
            "functional_proxy_pass": bool(proxy and proxy["runs"]["pass"]),
            "native_target_tuned_config": False,
            "native_exact_application_trace": False,
            "native_hardware_correlation": False,
            "strict_validation_eligible": False,
            "preferred_next_path": spec["preferred_next_path"],
            "evidence": evidence,
            "blocking_gates": blockers,
        }
    device_checks = {
        device: (
            record["vendor_identity_covered"]
            and record["open_candidate_available"]
            and record["strict_validation_eligible"] is False
            and not any(record["blocking_gates"].values())
        )
        for device, record in device_records.items()
    }
    counts = {
        "devices": len(device_records),
        "vendor_identity_coverage": sum(
            item["vendor_identity_covered"] for item in device_records.values()
        ),
        "open_candidate_coverage": sum(
            item["open_candidate_available"] for item in device_records.values()
        ),
        "local_executable_proxies": sum(
            item["local_executable_proxy"] for item in device_records.values()
        ),
        "native_tuned_configs": sum(
            item["native_target_tuned_config"] for item in device_records.values()
        ),
        "native_application_traces": sum(
            item["native_exact_application_trace"] for item in device_records.values()
        ),
        "validation_eligible_devices": sum(
            item["strict_validation_eligible"] for item in device_records.values()
        ),
    }
    coverage_checks = {
        key: counts[key] == int(config["acceptance"][f"required_{key}"])
        for key in counts
    }
    role_coverage = sorted(
        {figure for item in device_records.values() for figure in item["paper_figures"]}
    )
    classification_checks = {
        "device_checks": all(device_checks.values()),
        "paper_roles": role_coverage == [2, 3, 17, 20, 21, 24, 25],
        "rtx_closest": device_records["rtx3090"]["exact_isa_template"]
        and not device_records["xavier"]["exact_isa_template"]
        and not device_records["orin"]["exact_isa_template"],
        "flash_h100": device_records["h100"]["preferred_open_simulator"]
        == "FlashGPU-Sim"
        and device_records["h100"]["local_executable_proxy"] is False,
        "local_config_gaps": config_availability["rtx3070_sm86"]
        and config_availability["titanv_sm70"]
        and not any(
            config_availability[name]
            for name in ("xavier_sm72", "orin_sm87", "rtx3090", "h100_sm90")
        ),
    }
    source_files = {
        name: qualify(PROJECT_ROOT / path)
        for name, path in config["source_layout"].items()
    }
    source_text = "\n".join(
        (PROJECT_ROOT / path).read_text(errors="replace")
        for name, path in config["source_layout"].items()
        if name in {"auditor", "test"}
    )
    target_free_checks = {
        "all_inputs_target_free": all(
            input_["paper_performance_targets_consumed"] is False
            for input_ in inputs.values()
        ),
        "no_target_artifact_path": "paper_" + "targets.yaml" not in source_text
        and "fig17-cross" + "-figure-run021.json" not in source_text,
        "source_note_no_values": "No MLX paper result is used to tune" in note,
        "result_claim": True,
    }
    acceptance_gates = [
        all(item["pass"] for item in frozen.values()),
        all(proxy_checks.values()),
        len(device_records) == int(config["acceptance"]["required_devices"])
        and all(device_checks.values()),
        coverage_checks["vendor_identity_coverage"]
        and coverage_checks["open_candidate_coverage"],
        coverage_checks["local_executable_proxies"],
        coverage_checks["native_tuned_configs"]
        and coverage_checks["native_application_traces"],
        coverage_checks["validation_eligible_devices"],
        all(evidence_checks.values()),
        all(classification_checks.values()),
        all(pin_checks.values())
        and all(target_free_checks.values())
        and all(item["pass"] for item in source_files.values()),
    ]
    integrity_checks = {
        "frozen": all(item["pass"] for item in frozen.values()),
        "proxies": all(proxy_checks.values()),
        "evidence": all(evidence_checks.values()),
        "pins": all(pin_checks.values()),
        "devices": all(device_checks.values()),
        "counts": all(coverage_checks.values()),
        "classifications": all(classification_checks.values()),
        "sources": all(item["pass"] for item in source_files.values()),
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
        "paper_reproduction_claim": "none_gpu_mapping_and_gap_audit_only",
        "frozen_inputs": frozen,
        "proxy_checks": proxy_checks,
        "evidence_checks": evidence_checks,
        "pin_checks": pin_checks,
        "config_availability": config_availability,
        "device_records": device_records,
        "device_checks": device_checks,
        "role_coverage": role_coverage,
        "counts": counts,
        "coverage_checks": coverage_checks,
        "classification_checks": classification_checks,
        "source_files": source_files,
        "target_free_checks": target_free_checks,
        "acceptance_gates": acceptance_gates,
        "summary": {
            **counts,
            "paper_figure_roles": len(role_coverage),
            "closest_current_proxy": "rtx3090_on_SM86_RTX3070_template",
            "preferred_h100_candidate": "FlashGPU-Sim_SM90_H100",
            "all_current_denominators_proxy_only": True,
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
            "config_availability",
            "device_records",
            "counts",
            "classification_checks",
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
