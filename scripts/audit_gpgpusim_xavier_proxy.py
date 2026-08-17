#!/usr/bin/env python3
"""Audit H56's execution-driven GPGPU-Sim Jetson Xavier proxy."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from scripts.audit_dsagen_mlx_dma_memory import git_revision, load_yaml, qualify_file
from scripts.audit_gpgpusim_rtx3090_proxy import parse_run
from scripts.build_gpgpusim_xavier_config import derive_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/gpgpusim_xavier_proxy_v1.yaml"
EVIDENCE_ROOT = PROJECT_ROOT / "artifacts/environment/h56"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--verify-existing", action="store_true")
    return parser.parse_args()


def build_audit(config: dict) -> dict:
    parent_spec = config["parent"]
    parent_artifact = qualify_file(PROJECT_ROOT / parent_spec["path"], parent_spec)
    parent = json.loads((PROJECT_ROOT / parent_spec["path"]).read_text(encoding="utf-8"))
    base_path = PROJECT_ROOT / config["source"]["base_config"]
    derived_path = EVIDENCE_ROOT / "config/gpgpusim.config"
    manifest_path = EVIDENCE_ROOT / "config/xavier-config-manifest.json"
    expected_text, counts = derive_config(base_path.read_text(encoding="utf-8"))
    derived_text = derived_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_checks = {
        "exact": expected_text == derived_text,
        "counts": counts == manifest.get("replacement_counts"),
        "substitutions": manifest.get("registered_substitutions")
        == config["registered_substitutions"],
        "clusters": "-gpgpu_n_clusters 8" in derived_text,
        "cores": "-gpgpu_n_cores_per_cluster 1" in derived_text,
        "memory": "-gpgpu_n_mem 16" in derived_text,
        "clocks": "-gpgpu_clock_domains 1377:1377:1377:2133" in derived_text,
        "interconnect": manifest["interconnect_source"]["sha256"]
        == manifest["interconnect_copy"]["sha256"],
        "no_targets": manifest.get("paper_performance_targets_consumed") is False,
    }
    expected = {
        "vectoradd": ("vectoradd", 505, 21504, 8),
        "bsmm": ("bsmm", 4352, 1343488, 328),
        "fft": ("fft", 9062, 3873280, 328),
        "swa": ("swa", 13455, 6119168, 82),
    }
    runs = {}
    run_checks = {}
    for name, (operator, cycles, instructions, ctas) in expected.items():
        item = parse_run(EVIDENCE_ROOT / f"runs/{name}/run.log")
        summary = item["summary"] or {}
        checks = {
            "operator": summary.get("operator") == operator,
            "checksum": summary.get("relative_error", 1.0) <= 1e-6,
            "cycles": item["cycles"] == cycles,
            "instructions": item["instructions"] == instructions,
            "ctas": item["ctas"] == ctas,
            "detailed": item["detailed_mode"],
            "exit": item["normal_exit"],
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
    binary_record = EVIDENCE_ROOT / "runs/binary-sha256.txt"
    binary_hash = binary_record.read_text(encoding="utf-8").split()[0]
    binary = qualify_file(PROJECT_ROOT / "build/gpgpusim-xavier-proxy/mlx_gpu_proxy")
    sources = {
        key: qualify_file(PROJECT_ROOT / path)
        for key, path in config["source_layout"].items()
    }
    integrity_checks = {
        "parent": parent_artifact["pass"]
        and parent.get("hypothesis_status") == parent_spec["required_status"]
        and parent.get("audit_integrity") is parent_spec["required_integrity"],
        "config": all(config_checks.values()),
        "runs": all(run_checks.values()),
        "revision": revision == parent.get("gpgpu_sim_revision"),
        "binary": binary["pass"] and binary["sha256"] == binary_hash,
        "sources": all(item["pass"] for item in sources.values()),
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
        "config_derivation": {
            "base": qualify_file(base_path),
            "derived": qualify_file(derived_path),
            "manifest": qualify_file(manifest_path),
            "checks": config_checks,
        },
        "source": sources,
        "binary": {
            "artifact": binary,
            "record": qualify_file(binary_record),
            "recorded_sha256": binary_hash,
        },
        "runs": {"items": runs, "checks": run_checks, "pass": all(run_checks.values())},
        "integrity_checks": integrity_checks,
        "paper_performance_targets_consumed": False,
        "proxy_limitations": [
            "tested SM70 TitanV timing is retained for an SM72 device",
            "configuration is not vendor validated",
            "dense Tensor Core kernels are not yet represented",
        ],
    }


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config.resolve())
    report = build_audit(config)
    output = PROJECT_ROOT / config["result_path"]
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.verify_existing:
        if not output.is_file() or output.read_text(encoding="utf-8") != rendered:
            raise SystemExit("existing H56 result does not match a fresh audit")
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
