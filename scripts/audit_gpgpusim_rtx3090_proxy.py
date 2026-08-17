#!/usr/bin/env python3
"""Audit H51's execution-driven GPGPU-Sim RTX3090 proxy."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from scripts.audit_dsagen_mlx_dma_memory import git_revision, load_yaml, qualify_file
from scripts.build_gpgpusim_rtx3090_config import derive_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/gpgpusim_rtx3090_proxy_v1.yaml"
EVIDENCE_ROOT = PROJECT_ROOT / "artifacts/environment/h51"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--verify-existing", action="store_true")
    return parser.parse_args()


def last_int(text: str, key: str) -> int | None:
    matches = re.findall(rf"^{re.escape(key)} = (\d+)$", text, flags=re.MULTILINE)
    return int(matches[-1]) if matches else None


def parse_run(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    summaries = re.findall(r"^MLX_GPU_PROXY_SUMMARY (\{.*\})$", text, flags=re.MULTILINE)
    return {
        "artifact": qualify_file(path),
        "summary": json.loads(summaries[-1]) if summaries else None,
        "cycles": last_int(text, "gpu_tot_sim_cycle"),
        "instructions": last_int(text, "gpu_tot_sim_insn"),
        "ctas": last_int(text, "gpu_tot_issued_cta"),
        "detailed_mode": "GPGPU-Sim PTX: simulation mode 0" in text,
        "normal_exit": "GPGPU-Sim: *** exit detected ***" in text,
        "version": "GPGPU-Sim Simulator Version 4.2.0" in text,
    }


def config_audit(config: dict[str, Any]) -> dict[str, Any]:
    base_path = PROJECT_ROOT / config["source"]["base_config"]
    derived_path = EVIDENCE_ROOT / "config/gpgpusim.config"
    manifest_path = EVIDENCE_ROOT / "config/rtx3090-config-manifest.json"
    base_text = base_path.read_text(encoding="utf-8")
    expected_text, counts = derive_config(base_text)
    derived_text = derived_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = {
        "exact_derivation": expected_text == derived_text,
        "replacement_counts": counts == manifest.get("replacement_counts"),
        "substitutions": manifest.get("registered_substitutions")
        == config["registered_substitutions"],
        "clusters": derived_text.count("-gpgpu_n_clusters 82") == 1,
        "memory_partitions": derived_text.count("-gpgpu_n_mem 24") == 1,
        "clocks": derived_text.count(
            "-gpgpu_clock_domains 1695:1695:1695:5250"
        )
        == 1,
        "interconnect_identity": manifest.get("interconnect_source", {}).get("sha256")
        == manifest.get("interconnect_copy", {}).get("sha256"),
        "no_targets": manifest.get("paper_performance_targets_consumed") is False,
    }
    return {
        "base": qualify_file(base_path),
        "derived": qualify_file(derived_path),
        "interconnect": qualify_file(EVIDENCE_ROOT / "config/config_ampere_islip.icnt"),
        "manifest": qualify_file(manifest_path),
        "checks": checks,
        "pass": all(checks.values()),
    }


def source_audit(config: dict[str, Any]) -> dict[str, Any]:
    layout = config["source_layout"]
    token_map = {
        "config_builder": ["derive_config", "gpgpu_n_clusters 82", "gpgpu_n_mem 24"],
        "cuda_source": ["bsmm_stage", "fft_stage", "swa_kernel", "MLX_GPU_PROXY_SUMMARY"],
        "build_runner": ["compute_86", "setup_environment", "cuobjdump"],
        "auditor": ["gpu_tot_sim_cycle", "exact_derivation", "relative_error"],
    }
    files = {}
    for key, tokens in token_map.items():
        path = PROJECT_ROOT / layout[key]
        text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
        checks = {token: token in text for token in tokens}
        files[key] = {
            "path": layout[key],
            "tokens": checks,
            "pass": path.is_file() and all(checks.values()),
        }
    forbidden = [
        token
        for token in ("fig18", "fig20", "fig21", "fig24", "fig25", "paper_targets")
        if token
        in (PROJECT_ROOT / layout["cuda_source"]).read_text(
            encoding="utf-8", errors="replace"
        ).lower()
    ]
    checks = {
        "files": all(item["pass"] for item in files.values()),
        "targets_absent": not forbidden,
    }
    return {
        "files": files,
        "forbidden_target_tokens": forbidden,
        "checks": checks,
        "pass": all(checks.values()),
    }


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    parent_spec = config["parent"]
    parent_artifact = qualify_file(PROJECT_ROOT / parent_spec["path"], parent_spec)
    parent = json.loads((PROJECT_ROOT / parent_spec["path"]).read_text(encoding="utf-8"))
    derived_config = config_audit(config)
    source = source_audit(config)
    expected = {
        "vectoradd": {"operator": "vectoradd", "cycles": 5593, "instructions": 21504, "ctas": 8},
        "bsmm": {"operator": "bsmm", "cycles": 22812, "instructions": 1343488, "ctas": 328},
        "fft": {"operator": "fft", "cycles": 23469, "instructions": 3862528, "ctas": 328},
        "swa": {"operator": "swa", "cycles": 12814, "instructions": 6119168, "ctas": 82},
    }
    runs = {}
    run_checks = {}
    for name, expectation in expected.items():
        item = parse_run(EVIDENCE_ROOT / f"runs/{name}/run.log")
        summary = item["summary"] or {}
        checks = {
            "artifact": item["artifact"]["pass"],
            "operator": summary.get("operator") == expectation["operator"],
            "checksum": summary.get("relative_error", 1.0) <= 1e-6,
            "cycles": item["cycles"] == expectation["cycles"],
            "instructions": item["instructions"] == expectation["instructions"],
            "ctas": item["ctas"] == expectation["ctas"],
            "detailed_mode": item["detailed_mode"],
            "normal_exit": item["normal_exit"],
            "version": item["version"],
        }
        item["checks"] = checks
        item["pass"] = all(checks.values())
        runs[name] = item
        run_checks[name] = item["pass"]

    gpgpu_root = PROJECT_ROOT / "third_party/accel-sim-framework/gpu-simulator/gpgpu-sim"
    revision = subprocess.run(
        ["git", "-c", f"safe.directory={gpgpu_root}", "rev-parse", "HEAD"],
        cwd=gpgpu_root,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    binary_hash_path = EVIDENCE_ROOT / "runs/binary-sha256.txt"
    binary_hash_line = binary_hash_path.read_text(encoding="utf-8").strip().split()[0]
    binary_path = PROJECT_ROOT / "build/gpgpusim-rtx3090-proxy/mlx_gpu_proxy"
    binary_artifact = qualify_file(binary_path)
    integrity_checks = {
        "parent": parent_artifact["pass"]
        and parent.get("hypothesis_status") == parent_spec["required_status"]
        and parent.get("audit_integrity") is parent_spec["required_integrity"],
        "revision": revision == config["source"]["gpgpu_sim_revision"],
        "config": derived_config["pass"],
        "source": source["pass"],
        "runs": all(run_checks.values()),
        "binary": binary_artifact["pass"] and binary_artifact["sha256"] == binary_hash_line,
        "no_targets": True,
    }
    integrity = all(integrity_checks.values())
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "classification": config["classification"],
        "hypothesis_status": "supported" if integrity else "rejected",
        "audit_integrity": integrity,
        "git_revision": git_revision(PROJECT_ROOT),
        "gpgpu_sim_revision": revision,
        "parent": parent_artifact,
        "config_derivation": derived_config,
        "source": source,
        "binary": {
            "artifact": binary_artifact,
            "record": qualify_file(binary_hash_path),
            "recorded_sha256": binary_hash_line,
        },
        "runs": {"items": runs, "checks": run_checks, "pass": all(run_checks.values())},
        "excluded_attempts": {
            "nvbit_1_7_3": {
                "status": "unsupported_host_driver",
                "host_driver": config["toolchain"]["nvbit_attempt"]["host_driver"],
                "error": config["toolchain"]["nvbit_attempt"]["error"],
                "used_for_status": False,
            },
            "config_count": qualify_file(EVIDENCE_ROOT / "runs-failed-config-count/config-builder.json"),
            "nounset": qualify_file(EVIDENCE_ROOT / "runs-failed-nounset/config-builder.json"),
        },
        "integrity_checks": integrity_checks,
        "paper_performance_targets_consumed": False,
    }


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config.resolve())
    report = build_audit(config)
    output = PROJECT_ROOT / config["result_path"]
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.verify_existing:
        if not output.is_file() or output.read_text(encoding="utf-8") != rendered:
            raise SystemExit("existing H51 result does not match a fresh audit")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            {
                "hypothesis_status": report["hypothesis_status"],
                "audit_integrity": report["audit_integrity"],
                "cycles": {name: item["cycles"] for name, item in report["runs"]["items"].items()},
            },
            indent=2,
        )
    )
    return 0 if report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
