#!/usr/bin/env python3
"""Audit H40's pinned DSAGEN + Accel-Sim executable substrate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/open_hybrid_v1.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--verify-existing", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected YAML mapping: {path}")
    return value


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def qualify_file(path: Path, *, executable: bool = False) -> dict[str, Any]:
    is_file = path.is_file()
    mode_executable = is_file and bool(path.stat().st_mode & 0o111)
    checks = {"is_file": is_file}
    if executable:
        checks["executable"] = mode_executable
    return {
        "path": str(path.relative_to(PROJECT_ROOT)) if is_file else str(path),
        "bytes": path.stat().st_size if is_file else None,
        "sha256": sha256_file(path) if is_file else None,
        "checks": checks,
        "pass": all(checks.values()),
    }


def git_output(path: Path, *arguments: str) -> str | None:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={path}", *arguments],
        cwd=path,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def qualify_repository(path: Path, expected: str) -> dict[str, Any]:
    actual = git_output(path, "rev-parse", "HEAD") if path.is_dir() else None
    checks = {"is_directory": path.is_dir(), "revision": actual == expected}
    return {
        "path": str(path.relative_to(PROJECT_ROOT)) if path.is_dir() else str(path),
        "expected_revision": expected,
        "actual_revision": actual,
        "checks": checks,
        "pass": all(checks.values()),
    }


def repository_specs(config: dict[str, Any]) -> list[tuple[str, Path, str]]:
    dsagen = config["sources"]["dsagen"]
    accel = config["sources"]["accel_sim"]
    dsagen_root = PROJECT_ROOT / "third_party/dsa-framework"
    accel_root = PROJECT_ROOT / "third_party/accel-sim-framework"
    specs = [("dsagen", dsagen_root, dsagen["revision"])]
    for name, revision in dsagen["required_submodules"].items():
        specs.append((f"dsagen/{name}", dsagen_root / name, revision))
    nested = dsagen["resolved_toolchain_dependencies"]
    gnu_root = dsagen_root / "chipyard/toolchains/riscv-tools/riscv-gnu-toolchain"
    specs.extend(
        [
            ("dsagen/riscv-gnu-toolchain", gnu_root, nested["riscv-gnu-toolchain"]),
            ("dsagen/riscv-binutils", gnu_root / "riscv-binutils", nested["riscv-binutils"]),
            ("accel-sim", accel_root, accel["revision"]),
            (
                "accel-sim/gpgpu-sim",
                accel_root / "gpu-simulator/gpgpu-sim",
                accel["resolved_dependencies"]["gpgpu_sim"]["revision"],
            ),
            (
                "accel-sim/pybind11",
                accel_root / "gpu-simulator/extern/pybind11",
                accel["resolved_dependencies"]["pybind11"]["revision"],
            ),
        ]
    )
    return specs


def qualify_license(path: Path, label: str) -> dict[str, Any]:
    report = qualify_file(path)
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    checks = {
        **report["checks"],
        "redistribution_permission": "Redistribution and use" in text,
        "warranty_disclaimer": "AS IS" in text,
    }
    return {**report, "label": label, "checks": checks, "pass": all(checks.values())}


def line_number(text: str, token: str) -> int | None:
    offset = text.find(token)
    return None if offset < 0 else text.count("\n", 0, offset) + 1


def qualify_mechanisms(config: dict[str, Any]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for specification in config["mechanism_map"]:
        path = PROJECT_ROOT / specification["path"]
        source = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
        tokens = {
            token: {"line": line_number(source, token), "present": token in source}
            for token in specification["required_tokens"]
        }
        checks = {
            "source_file": path.is_file(),
            "all_tokens": all(item["present"] for item in tokens.values()),
            "decision_registered": specification["decision"]
            in {"reuse", "adapt", "extend", "replace_policy", "derive_abstraction"},
            "mlx_extension_registered": bool(specification["mlx_extension"].strip()),
        }
        reports.append(
            {
                **{key: specification[key] for key in ("id", "upstream", "path", "symbol")},
                "decision": specification["decision"],
                "mlx_extension": specification["mlx_extension"],
                "tokens": tokens,
                "checks": checks,
                "pass": all(checks.values()),
            }
        )
    return reports


def required_markers(text: str, markers: list[str]) -> dict[str, bool]:
    return {marker: marker in text for marker in markers}


def last_int(text: str, pattern: str) -> int | None:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    return int(matches[-1]) if matches else None


def parse_accelsim_metrics(text: str) -> dict[str, Any]:
    kernels = re.findall(r"^Processing kernel (.+)$", text, flags=re.MULTILINE)
    wall = re.findall(r"^wall_seconds=([0-9.]+)", text, flags=re.MULTILINE)
    return {
        "cumulative_cycles": last_int(text, r"^gpu_tot_sim_cycle\s*=\s*(\d+)"),
        "cumulative_instructions": last_int(text, r"^gpu_tot_sim_insn\s*=\s*(\d+)"),
        "cumulative_ctas": last_int(text, r"^gpu_tot_issued_cta\s*=\s*(\d+)"),
        "kernel_count": len(kernels),
        "kernel_traces": kernels,
        "wall_seconds": float(wall[-1]) if wall else None,
    }


def parse_dsagen_metrics(text: str) -> dict[str, Any]:
    return {
        "cycles": last_int(text, r"^Cycles:\s*(\d+)"),
        "cgra_instances": last_int(text, r"^CGRA Instances:\s*(\d+)"),
        "cgra_instructions": last_int(text, r"^CGRA Insts / Cycle:\s*(\d+)\s*/"),
        "dma_read_bytes": last_int(text, r"^Read DMA:\s*(\d+) B"),
        "dma_write_bytes": last_int(text, r"^Write DMA:\s*(\d+) B"),
        "exit_tick": last_int(text, r"^Exiting @ tick (\d+)"),
        "sanity_check_passed": "sanity check passed successfully!" in text,
        "simulated_exit_code_nonzero": "Simulated exit code not 0!" in text,
    }


def qualify_smoke(specification: dict[str, Any], parser: str) -> dict[str, Any]:
    path = PROJECT_ROOT / specification["evidence_log"]
    artifact = qualify_file(path)
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    markers = required_markers(text, specification["required_markers"])
    metrics = parse_dsagen_metrics(text) if parser == "dsagen" else parse_accelsim_metrics(text)
    metric_checks = (
        {
            "sanity_check": metrics["sanity_check_passed"],
            "normal_exit": not metrics["simulated_exit_code_nonzero"],
            "real_cgra_work": metrics["cgra_instances"] == 256
            and metrics["cgra_instructions"] == 1024,
            "real_dma_work": metrics["dma_read_bytes"] == 16384
            and metrics["dma_write_bytes"] == 8192,
        }
        if parser == "dsagen"
        else {
            "normal_exit": "GPGPU-Sim: *** exit detected ***" in text,
            "two_kernels": metrics["kernel_count"] == 2,
            "real_gpu_work": metrics["cumulative_cycles"] == 14903
            and metrics["cumulative_instructions"] == 9290080
            and metrics["cumulative_ctas"] == 512,
        }
    )
    checks = {
        "evidence_file": artifact["pass"],
        "required_markers": all(markers.values()),
        **metric_checks,
    }
    return {
        "evidence": artifact,
        "markers": markers,
        "metrics": metrics,
        "checks": checks,
        "pass": all(checks.values()),
    }


def qualify_dsa_instruction_encoding() -> dict[str, Any]:
    objdump = (
        PROJECT_ROOT
        / "third_party/dsa-framework/ss-tools/riscv-dsa-binutils/bin/"
        "riscv64-unknown-linux-gnu-objdump"
    )
    executable = (
        PROJECT_ROOT
        / "third_party/dsa-framework/dsa-apps/sdk/compiled/ss-vecadd-gnu.out"
    )
    result = subprocess.run(
        [str(objdump), "-drwC", str(executable)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    ) if objdump.is_file() and executable.is_file() else None
    text = result.stdout if result is not None else ""
    expected = (
        "ss_cfg_param\ta3,a4,98",
        "ss_cfg_param\ta1,a6,229",
        "ss_cfg_param\ta1,a4,1220",
        "ss_wait\ta1,a0,0",
    )
    markers = required_markers(text, list(expected))
    checks = {
        "objdump": objdump.is_file(),
        "executable": executable.is_file(),
        "objdump_exit_zero": result is not None and result.returncode == 0,
        "all_masks_encoded": all(markers.values()),
    }
    return {"markers": markers, "checks": checks, "pass": all(checks.values())}


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    repositories = {
        name: qualify_repository(path, revision)
        for name, path, revision in repository_specs(config)
    }
    licenses = {
        "dsagen": qualify_license(
            PROJECT_ROOT / "third_party/dsa-framework/LICENSE", "BSD-2-Clause"
        ),
        "accel_sim": qualify_license(
            PROJECT_ROOT / "third_party/accel-sim-framework/LICENSE",
            "permissive BSD text",
        ),
        "gpgpu_sim": qualify_license(
            PROJECT_ROOT
            / "third_party/accel-sim-framework/gpu-simulator/gpgpu-sim/COPYRIGHT",
            "BSD-3-Clause-style",
        ),
    }
    mechanisms = qualify_mechanisms(config)
    dsagen_smoke = qualify_smoke(config["spatial_smoke"], "dsagen")
    accelsim_smoke = qualify_smoke(config["gpu_smoke"], "accelsim")
    binaries = {
        "dsagen_gem5": qualify_file(PROJECT_ROOT / config["spatial_smoke"]["simulator"], executable=True),
        "dsagen_vecadd": qualify_file(PROJECT_ROOT / config["spatial_smoke"]["executable"], executable=True),
        "accel_sim": qualify_file(PROJECT_ROOT / config["gpu_smoke"]["simulator"], executable=True),
    }
    patches = {
        path: qualify_file(PROJECT_ROOT / path)
        for path in config["build_toolchains"]["dsagen"]["compatibility_patches"]
    }
    encoding = qualify_dsa_instruction_encoding()
    decisions = {item["decision"] for item in mechanisms if item["upstream"] == "gpgpu_sim"}
    extension_text = " ".join(item["mlx_extension"].lower() for item in mechanisms)
    pass_criteria = {
        "licensed_reused_sources": all(item["pass"] for item in licenses.values()),
        "pinned_recursive_sources": all(item["pass"] for item in repositories.values()),
        "spatial_simulator_executes_real_dsa": dsagen_smoke["pass"],
        "gpu_simulator_executes_real_trace": accelsim_smoke["pass"],
        "all_required_mechanisms_have_source_symbols": all(item["pass"] for item in mechanisms),
        "actionable_tagged_block_boundary": "tag" in extension_text and "block" in extension_text,
        "actionable_skip_hop_boundary": "skip-hop" in extension_text,
        "actionable_layer_arbitration_boundary": "layer" in extension_text
        and "arbitration" in extension_text,
        "gpu_code_is_reference_not_simt_import": decisions == {"derive_abstraction"},
        "dsa_instruction_masks_correct": encoding["pass"],
        "compatibility_patches_present": all(item["pass"] for item in patches.values()),
        "paper_target_values_consumed": False,
    }
    audit_integrity = all(
        value for key, value in pass_criteria.items() if key != "paper_target_values_consumed"
    ) and pass_criteria["paper_target_values_consumed"] is False
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_commit": git_output(PROJECT_ROOT, "rev-parse", "HEAD"),
        "hypothesis_status": "supported" if audit_integrity else "rejected",
        "audit_integrity": audit_integrity,
        "selection": {
            "primary_timing_substrate": "DSAGEN dsa-gem5",
            "gpu_baseline": "Accel-Sim trace-driven GPGPU-Sim",
            "programmable_pe_reference": "GPGPU-Sim resource mechanisms only",
            "analytical_crosscheck": "Timeloop (not the dynamic timing engine)",
            "inspect_only": "Assassyn (no code reuse at the frozen unlicensed checkout)",
        },
        "repositories": repositories,
        "licenses": licenses,
        "binaries": binaries,
        "patches": patches,
        "dsa_instruction_encoding": encoding,
        "mechanisms": mechanisms,
        "spatial_smoke": dsagen_smoke,
        "gpu_smoke": accelsim_smoke,
        "pass_criteria": pass_criteria,
        "wall_seconds": time.perf_counter() - started,
    }


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    report = build_audit(config)
    output = PROJECT_ROOT / config["result_path"]
    if args.verify_existing:
        if not output.is_file():
            raise FileNotFoundError(output)
        existing = json.loads(output.read_text(encoding="utf-8"))
        keys = ("hypothesis_status", "audit_integrity", "pass_criteria")
        matches = all(existing.get(key) == report.get(key) for key in keys)
        print(json.dumps({"existing_matches": matches, **report}, indent=2, sort_keys=True))
        return 0 if matches else 1
    if args.preflight_only:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["audit_integrity"] and not output.exists() else 1
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
