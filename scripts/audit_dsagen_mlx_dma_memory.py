#!/usr/bin/env python3
"""Audit H47's real MinorCPU LSQ/cache/DDR memory path."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simulators/dsagen_mlx_dma_memory_v1.yaml"
EVIDENCE_ROOT = PROJECT_ROOT / "artifacts/environment/h47"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--verify-existing", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError(f"expected YAML mapping: {path}")
    return document


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def qualify_file(path: Path, expected: dict[str, Any] | None = None) -> dict[str, Any]:
    exists = path.is_file()
    size = path.stat().st_size if exists else None
    digest = sha256_file(path) if exists else None
    checks = {"is_file": exists}
    if expected is not None:
        if "bytes" in expected:
            checks["bytes"] = size == int(expected["bytes"])
        if "sha256" in expected:
            checks["sha256"] = digest == expected["sha256"]
    try:
        display = str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        display = str(path)
    return {
        "path": display,
        "bytes": size,
        "sha256": digest,
        "checks": checks,
        "pass": all(checks.values()),
    }


def git_revision(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={path}", "rev-parse", "HEAD"],
        cwd=path,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def parse_prefixed_json(text: str, prefix: str) -> dict[str, Any] | None:
    matches = re.findall(rf"^{re.escape(prefix)} (\{{.*\}})$", text, flags=re.MULTILINE)
    return json.loads(matches[-1]) if matches else None


def parse_stats(path: Path) -> dict[str, int | float]:
    stats: dict[str, int | float] = {}
    if not path.is_file():
        return stats
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        try:
            value = float(fields[1])
        except ValueError:
            continue
        stats[fields[0]] = int(value) if value.is_integer() else value
    return stats


def parse_dsagen(text: str) -> dict[str, Any]:
    def last(pattern: str) -> int | None:
        matches = re.findall(pattern, text, flags=re.MULTILINE)
        return int(matches[-1]) if matches else None

    return {
        "roi_cycles": last(r"^Cycles:\s*(\d+)"),
        "cgra_instances": last(r"^CGRA Instances:\s*(\d+)"),
        "cgra_instructions": last(r"^CGRA Insts / Cycle:\s*(\d+)\s*/"),
        "sanity": "sanity check passed successfully!" in text,
        "normal_exit": "exiting with last active thread context" in text
        and "Simulated exit code not 0!" not in text,
    }


def read_log(path: Path) -> dict[str, Any]:
    artifact = qualify_file(path)
    text = path.read_text(encoding="utf-8", errors="replace") if artifact["pass"] else ""
    return {
        "artifact": artifact,
        "overlay": parse_prefixed_json(text, "MLX_OVERLAY_SUMMARY"),
        "spad_adapter": parse_prefixed_json(text, "MLX_SPAD_ADAPTER_SUMMARY"),
        "dma_adapter": parse_prefixed_json(text, "MLX_DMA_ADAPTER_SUMMARY"),
        "guest_symbols": parse_prefixed_json(text, "MLX_DMA_GUEST_SYMBOLS"),
        "guest": parse_prefixed_json(text, "MLX_DMA_GUEST_SUMMARY"),
        "dsagen": parse_dsagen(text),
    }


def source_audit(config: dict[str, Any]) -> dict[str, Any]:
    layout = config["source_layout"]
    token_map = {
        "adapter_header": ["class Gem5DmaAdapter", "TransferIndex = 126", "step()"],
        "adapter_source": [
            "lsq->sd_transfers[TransferIndex]",
            "lsq->sendStoreToStoreBuffer(response)",
            "popMlxDmaStoreCompletion",
            "pushRequest",
        ],
        "lsq_header": ["is_mlx_dma", "mlx_token", "mlxDmaMasterId"],
        "lsq_source": [
            'getMasterId(&cpu_, "mlx_dma")',
            "mlxDmaStoreCompletions.emplace_back",
            "sdInfo && sdInfo->is_mlx_dma",
        ],
        "guest_source": [
            "mlx_dma_cold_region",
            "mlx_dma_evict_region",
            "selected_write_checksum",
        ],
        "compiler": [
            "compile_dma_microtrace",
            "read_elf_symbols",
            '"paper_performance_targets_consumed": False',
        ],
        "runner": [
            "system\\.mem_ctrls\\.num_reads::\\.cpu\\.mlx_dma",
            '"read_byte_sum":512',
        ],
    }
    files: dict[str, Any] = {}
    for key, tokens in token_map.items():
        path = PROJECT_ROOT / layout[key]
        text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
        checks = {token: token in text for token in tokens}
        files[key] = {
            "path": layout[key],
            "tokens": checks,
            "pass": path.is_file() and all(checks.values()),
        }

    overlay_header = PROJECT_ROOT / "third_party/dsa-framework/dsa-gem5/src/cpu/minor/ssim/mlx_overlay.hh"
    overlay_source = PROJECT_ROOT / "third_party/dsa-framework/dsa-gem5/src/cpu/minor/ssim/mlx_overlay.cc"
    accel_source = PROJECT_ROOT / "third_party/dsa-framework/dsa-gem5/src/cpu/minor/ssim/accel.cc"
    extra_tokens = {
        "compiler_core": (
            PROJECT_ROOT / "src/mlxsim/dsagen_dma.py",
            ["memory_address_sequence", '"start_in_roi": True'],
        ),
        "overlay_header": (overlay_header, ["DmaAdapter", "start_in_roi"]),
        "overlay_source": (overlay_source, ['name == "dsagen_dma"', "MemoryBackend::DmaAdapter"]),
        "accel_source": (
            accel_source,
            ["Gem5DmaAdapter", "MLX_DMA_ADAPTER_SUMMARY", "MLX overlay has in-flight"],
        ),
    }
    for key, (path, tokens) in extra_tokens.items():
        text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
        checks = {token: token in text for token in tokens}
        files[key] = {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "tokens": checks,
            "pass": path.is_file() and all(checks.values()),
        }

    patch_path = PROJECT_ROOT / layout["tracked_patch"]
    patch = qualify_file(patch_path)
    patch_text = patch_path.read_text(encoding="utf-8", errors="replace") if patch["pass"] else ""
    forbidden = re.findall(
        r"\b(?:warp|simt|cta|coher(?:ence|ent)?)\b", patch_text, flags=re.IGNORECASE
    )
    gem5_root = PROJECT_ROOT / "third_party/dsa-framework/dsa-gem5"
    reverse = subprocess.run(
        ["git", "apply", "--check", "--reverse", str(patch_path)],
        cwd=gem5_root,
        check=False,
        capture_output=True,
        text=True,
    )
    patch_checks = {
        "present": patch["pass"],
        "reverse_applies": reverse.returncode == 0,
        "forbidden_gpu_state_absent": not forbidden,
    }
    return {
        "files": files,
        "patch": {
            **patch,
            "checks": patch_checks,
            "forbidden_gpu_tokens": forbidden,
            "reverse_stderr": reverse.stderr,
            "pass": all(patch_checks.values()),
        },
        "pass": all(item["pass"] for item in files.values()) and all(patch_checks.values()),
    }


def compiler_audit() -> dict[str, Any]:
    paths = {
        "fixed": EVIDENCE_ROOT / "mlx-dma-fixed.json",
        "dma": EVIDENCE_ROOT / "mlx-dma-real.json",
        "fixed_replay": EVIDENCE_ROOT / "replay/mlx-dma-fixed.json",
        "dma_replay": EVIDENCE_ROOT / "replay/mlx-dma-real.json",
        "manifest": EVIDENCE_ROOT / "mlx-dma-compile-manifest.json",
    }
    files = {name: qualify_file(path) for name, path in paths.items()}
    documents = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
        if name in {"fixed", "dma", "manifest"} and path.is_file()
    }
    fixed = documents.get("fixed", {})
    dma = documents.get("dma", {})
    manifest = documents.get("manifest", {})
    fixed_without_backend = copy.deepcopy(fixed)
    dma_without_backend = copy.deepcopy(dma)
    fixed_backend = fixed_without_backend.pop("memory_backend", None)
    dma_backend = dma_without_backend.pop("memory_backend", None)

    blocks = dma.get("blocks") or []
    sequences_ok = True
    tags_ok = True
    counts = {"load": 0, "compute": 0, "store": 0}
    for index, block in enumerate(blocks):
        tags_ok &= block.get("tag") == index + 1
        tags_ok &= block.get("pe") == [index % 4, index // 4]
        tags_ok &= block.get("trip_count") == 4
        for instruction in block.get("instructions") or []:
            pipeline = instruction.get("pipeline")
            if pipeline in counts:
                counts[pipeline] += int(block.get("trip_count", 0))
            if pipeline in {"load", "store"}:
                sequence = instruction.get("memory_address_sequence") or []
                sequences_ok &= len(sequence) == 4
                sequences_ok &= all(address % 8 == 0 for address in sequence)

    guest = manifest.get("guest_elf") or {}
    guest_path = Path(guest.get("path", ""))
    checks = {
        "files": all(item["pass"] for item in files.values()),
        "fixed_replay": files["fixed"]["sha256"] == files["fixed_replay"]["sha256"],
        "dma_replay": files["dma"]["sha256"] == files["dma_replay"]["sha256"],
        "only_backend_differs": fixed_without_backend == dma_without_backend,
        "backends": fixed_backend == "fixed" and dma_backend == "dsagen_dma",
        "roi_gate": fixed.get("start_in_roi") is True and dma.get("start_in_roi") is True,
        "blocks": len(blocks) == 16 and tags_ok,
        "counts": counts == {"load": 64, "compute": 64, "store": 64},
        "address_sequences": sequences_ok,
        "guest_hash": guest_path.is_file()
        and guest.get("sha256") == sha256_file(guest_path),
        "symbols": (guest.get("symbols") or {}).get("mlx_dma_cold_region", {}).get("size")
        == 131072
        and (guest.get("symbols") or {}).get("mlx_dma_write_region", {}).get("size") == 4096,
        "no_paper_targets": manifest.get("paper_performance_targets_consumed") is False,
    }
    return {
        "files": files,
        "counts": counts,
        "guest_elf": guest,
        "checks": checks,
        "pass": all(checks.values()),
    }


def build_audit(config: dict[str, Any]) -> dict[str, Any]:
    frozen = {
        name: qualify_file(PROJECT_ROOT / specification["path"], specification)
        for name, specification in config["frozen_inputs"].items()
        if isinstance(specification, dict) and "path" in specification
    }
    source = source_audit(config)
    compiler = compiler_audit()

    fixed = read_log(EVIDENCE_ROOT / "runs/fixed/run.log")
    dma = read_log(EVIDENCE_ROOT / "runs/dma/run.log")
    fixed_stats_file = qualify_file(EVIDENCE_ROOT / "runs/fixed/m5out/stats.txt")
    dma_stats_file = qualify_file(EVIDENCE_ROOT / "runs/dma/m5out/stats.txt")
    stats = parse_stats(EVIDENCE_ROOT / "runs/dma/m5out/stats.txt")
    fixed_overlay = fixed["overlay"] or {}
    dma_overlay = dma["overlay"] or {}
    adapter = dma["dma_adapter"] or {}

    cache_keys = {
        "l1_read_accesses": "system.cpu.dcache.ReadReq_accesses::.cpu.mlx_dma",
        "l1_read_misses": "system.cpu.dcache.ReadReq_misses::.cpu.mlx_dma",
        "l1_write_accesses": "system.cpu.dcache.WriteReq_accesses::.cpu.mlx_dma",
        "l1_write_misses": "system.cpu.dcache.WriteReq_misses::.cpu.mlx_dma",
        "l2_read_accesses": "system.l2.ReadSharedReq_accesses::.cpu.mlx_dma",
        "l2_read_misses": "system.l2.ReadSharedReq_misses::.cpu.mlx_dma",
        "l2_store_rfo_hits": "system.l2.ReadExReq_hits::.cpu.mlx_dma",
        "dram_reads": "system.mem_ctrls.num_reads::.cpu.mlx_dma",
        "dram_read_bytes": "system.mem_ctrls.bytes_read::.cpu.mlx_dma",
    }
    cache_dram = {name: stats.get(key) for name, key in cache_keys.items()}
    cache_checks = {
        "l1_reads": cache_dram["l1_read_accesses"] == cache_dram["l1_read_misses"] == 64,
        "l1_writes": cache_dram["l1_write_accesses"] == cache_dram["l1_write_misses"] == 64,
        "l2_cold_reads": cache_dram["l2_read_accesses"] == cache_dram["l2_read_misses"] == 64,
        "l2_store_rfo": cache_dram["l2_store_rfo_hits"] == 64,
        "dram": cache_dram["dram_reads"] == 64
        and cache_dram["dram_read_bytes"] == 4096,
    }

    fixed_checks = {
        "done": fixed_overlay.get("done") is True,
        "backend": fixed_overlay.get("memory_backend") == "fixed",
        "cycles": fixed_overlay.get("cycles") == 17,
        "instructions": fixed_overlay.get("instructions_issued")
        == fixed_overlay.get("instructions_completed")
        == 192,
        "no_external_memory": fixed_overlay.get("external_memory_requests") == 0
        and fixed_overlay.get("external_memory_completions") == 0,
        "checksum": (fixed["guest"] or {}).get("store_checksum") == 84480,
        "guest": fixed["dsagen"]["sanity"] and fixed["dsagen"]["normal_exit"],
    }
    dma_checks = {
        "done": dma_overlay.get("done") is True,
        "backend": dma_overlay.get("memory_backend") == "dsagen_dma",
        "instructions": dma_overlay.get("instructions_issued")
        == dma_overlay.get("instructions_completed")
        == 192,
        "external_memory": dma_overlay.get("external_memory_requests")
        == dma_overlay.get("external_memory_completions")
        == 128,
        "directional_requests": adapter.get("read_requests")
        == adapter.get("read_responses")
        == 64
        and adapter.get("write_requests") == adapter.get("write_responses") == 64,
        "zero_failures": adapter.get("failed_responses") == 0
        and adapter.get("outstanding") == 0,
        "real_latency": adapter.get("max_response_cycles", 0) > 1,
        "concurrency": adapter.get("max_outstanding", 0) > 1,
        "read_data": adapter.get("read_byte_sum") == 512,
        "write_data": (dma["guest"] or {}).get("store_checksum") == 0,
        "same_guest_symbols": dma.get("guest_symbols") == fixed.get("guest_symbols"),
        "guest": dma["dsagen"]["sanity"] and dma["dsagen"]["normal_exit"],
        "slower_than_fixed": dma_overlay.get("cycles", 0) > fixed_overlay.get("cycles", 0),
    }

    regression_paths = {
        "spad_bsmm": EVIDENCE_ROOT / "regressions/cdc/bsmm-b8/run.log",
        "spad_fft": EVIDENCE_ROOT / "regressions/cdc/fft-l8/run.log",
        "spad_stress": EVIDENCE_ROOT / "regressions/cdc/bsmm-b16-stress/run.log",
        "fixed_overlay": EVIDENCE_ROOT / "regressions/overlay/enabled/run.log",
        "disabled_overlay": EVIDENCE_ROOT / "regressions/overlay/disabled/run.log",
    }
    regressions = {name: read_log(path) for name, path in regression_paths.items()}

    def base_ok(item: dict[str, Any]) -> bool:
        metrics = item["dsagen"]
        return (
            metrics["roi_cycles"] == 569
            and metrics["cgra_instances"] == 256
            and metrics["cgra_instructions"] == 1024
            and metrics["sanity"]
            and metrics["normal_exit"]
        )

    regression_checks = {
        "spad_bsmm": (regressions["spad_bsmm"]["overlay"] or {}).get(
            "external_memory_requests"
        )
        == 36
        and (regressions["spad_bsmm"]["spad_adapter"] or {}).get("responses") == 36
        and base_ok(regressions["spad_bsmm"]),
        "spad_fft": (regressions["spad_fft"]["overlay"] or {}).get(
            "external_memory_requests"
        )
        == 36
        and (regressions["spad_fft"]["spad_adapter"] or {}).get("responses") == 36
        and base_ok(regressions["spad_fft"]),
        "spad_stress": (regressions["spad_stress"]["overlay"] or {})
        .get("stalls_by_reason", {})
        .get("memory_queue_full", 0)
        > 0
        and (regressions["spad_stress"]["spad_adapter"] or {}).get("responses") == 96
        and base_ok(regressions["spad_stress"]),
        "fixed_overlay": (regressions["fixed_overlay"]["overlay"] or {}).get("cycles")
        == 5
        and base_ok(regressions["fixed_overlay"]),
        "disabled_overlay": regressions["disabled_overlay"]["overlay"] is None
        and base_ok(regressions["disabled_overlay"]),
    }

    binaries = {
        "gem5": qualify_file(
            PROJECT_ROOT / "third_party/dsa-framework/dsa-gem5/build/RISCV/gem5.opt"
        ),
        "guest": qualify_file(
            PROJECT_ROOT / "third_party/dsa-framework/dsa-apps/sdk/compiled/ss-mlx-dma.out"
        ),
        "guest_build_log": qualify_file(EVIDENCE_ROOT / "guest-build.log"),
    }
    evidence_files = {
        "protocol": qualify_file(PROJECT_ROOT / "experiments/h47-dsagen-mlx-dma-memory/protocol.md"),
        "config": qualify_file(DEFAULT_CONFIG),
        "compiler_stdout": qualify_file(EVIDENCE_ROOT / "compiler-stdout.json"),
        "replay_stdout": qualify_file(EVIDENCE_ROOT / "replay/compiler-stdout.json"),
        "fixed_stats": fixed_stats_file,
        "dma_stats": dma_stats_file,
        **binaries,
    }
    pass_criteria = {
        "frozen_inputs": all(item["pass"] for item in frozen.values()),
        "source": source["pass"],
        "compiler": compiler["pass"],
        "fixed_control": all(fixed_checks.values()),
        "dma_execution": all(dma_checks.values()),
        "cache_and_dram": all(cache_checks.values()),
        "regressions": all(regression_checks.values()),
        "evidence_files": all(item["pass"] for item in evidence_files.values()),
        "no_paper_targets": config["classification"]
        == "mechanism_confirmatory_no_paper_target"
        and config["frozen_inputs"]["paper"]["consumed_performance_targets"] == [],
    }
    audit_integrity = all(pass_criteria.values())
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "classification": config["classification"],
        "hypothesis_status": "supported" if audit_integrity else "rejected",
        "audit_integrity": audit_integrity,
        "claim_scope": {
            "supported": [
                "MLX overlay loads traverse MinorCPU LSQ, L1D, L2, and DDR",
                "MLX overlay stores traverse the real LSQ/store-buffer response path",
                "the off-chip adapter preserves tagged spatial execution and PE-local pipelines",
            ],
            "not_claimed": [
                "authors' unpublished MLX simulator equivalence",
                "paper performance accuracy",
                "numerically correct full-model inference",
                "GPU warp, SIMT, CTA, or coherence behavior inside a PE",
            ],
        },
        "git_revision": git_revision(PROJECT_ROOT),
        "dsagen_revision": git_revision(
            PROJECT_ROOT / "third_party/dsa-framework/dsa-gem5"
        ),
        "frozen_inputs": frozen,
        "source": source,
        "compiler": compiler,
        "runs": {
            "fixed": {**fixed, "checks": fixed_checks, "pass": all(fixed_checks.values())},
            "dma": {**dma, "checks": dma_checks, "pass": all(dma_checks.values())},
        },
        "cache_dram_evidence": {
            "metrics": cache_dram,
            "checks": cache_checks,
            "pass": all(cache_checks.values()),
        },
        "regressions": {
            "runs": regressions,
            "checks": regression_checks,
            "pass": all(regression_checks.values()),
        },
        "excluded_development_evidence": {
            "smoke_attempts": list(range(1, 11)),
            "failed_long_wait_candidate": {
                "fixed": qualify_file(
                    EVIDENCE_ROOT / "runs-failed-long-wait-candidate/fixed/run.log"
                ),
                "dma": qualify_file(
                    EVIDENCE_ROOT / "runs-failed-long-wait-candidate/dma/run.log"
                ),
                "reason": "requestor command counters absent after obsolete long host wait",
            },
            "used_for_hypothesis_status": False,
        },
        "evidence_files": evidence_files,
        "pass_criteria": pass_criteria,
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
            raise SystemExit("existing H47 result does not match a fresh audit")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("experiment_id", "run_id", "hypothesis_status", "audit_integrity")}, indent=2))
    return 0 if report["audit_integrity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
