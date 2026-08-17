#!/usr/bin/env python3
"""Measure frozen Llama2-7B Q/K/V token spectra for MLX Figures 5 and 6."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
import transformers
import yaml
from transformers import AutoConfig, AutoModel, AutoTokenizer

from mlxsim.spectrum import (
    audit_measured_spectra,
    audit_spectrum_target_sources,
    grouped_projected_power,
    load_spectrum_targets,
    sha256_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/llama2_spectrum_v1.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true", help="run one window; never validation")
    return parser.parse_args()


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={PROJECT_ROOT}", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _qualify_model(model_cfg: dict[str, Any]) -> dict[str, Any]:
    model_path = PROJECT_ROOT / model_cfg["path"]
    files: dict[str, Any] = {}
    all_hashes_pass = True
    for filename, expected in model_cfg["required_official_hashes"].items():
        path = model_path / filename
        actual = sha256_file(path) if path.is_file() else None
        passed = actual == expected
        files[filename] = {
            "bytes": path.stat().st_size if path.is_file() else None,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "pass": passed,
        }
        all_hashes_pass &= passed

    config_path = model_path / "config.json"
    serialized = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    signature = {
        name: serialized.get(name) for name in model_cfg["config_signature"]
    }
    signature_pass = signature == model_cfg["config_signature"]
    report = {
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        "official_source": model_cfg["official_source"],
        "official_revision": model_cfg["official_revision"],
        "mirror_source": model_cfg["mirror_source"],
        "mirror_revision": model_cfg["mirror_revision"],
        "files": files,
        "config_signature": signature,
        "config_signature_pass": signature_pass,
        "pass": all_hashes_pass and signature_pass,
    }
    if not report["pass"]:
        raise RuntimeError(f"model input qualification failed: {json.dumps(report, sort_keys=True)}")
    return report


def _load_text(path: Path, column: str, separator: str) -> tuple[str, int]:
    table = pq.read_table(path, columns=[column])
    rows = table.column(column).to_pylist()
    return separator.join(str(row) for row in rows), len(rows)


def main() -> int:
    args = _parse_args()
    config = _load_config(args.config)
    output = args.output or PROJECT_ROOT / config["outputs"]["report"]
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    if not args.smoke and output.exists():
        raise SystemExit(f"refusing to overwrite official result: {output}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for H16")

    started = time.perf_counter()
    model_qualification = _qualify_model(config["model"])
    target_manifest = load_spectrum_targets(PROJECT_ROOT / config["paper_targets"]["manifest"])
    source_checks = audit_spectrum_target_sources(target_manifest)
    if not all(check["pass"] for check in source_checks.values()):
        raise RuntimeError(f"paper target source qualification failed: {source_checks}")

    dataset_path = PROJECT_ROOT / config["dataset"]["path"]
    dataset_hash = sha256_file(dataset_path)
    if dataset_hash != config["dataset"]["sha256"]:
        raise RuntimeError(
            f"dataset SHA-256 mismatch: expected {config['dataset']['sha256']}, got {dataset_hash}"
        )
    text, dataset_rows = _load_text(
        dataset_path, config["dataset"]["text_column"], config["dataset"]["row_separator"]
    )

    model_path = PROJECT_ROOT / config["model"]["path"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=False, local_files_only=True
    )
    encoded = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=config["sampling"]["add_special_tokens"],
    )["input_ids"]
    sequence_length = int(config["sampling"]["sequence_length"])
    start_token = int(config["sampling"]["start_token"])
    requested_windows = 1 if args.smoke else int(config["sampling"]["window_count"])
    required_tokens = start_token + requested_windows * sequence_length
    if encoded.shape[1] < required_tokens:
        raise RuntimeError(f"need {required_tokens} tokens, found {encoded.shape[1]}")

    device = torch.device(config["runtime"]["device"])
    serialized_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    dtype_keyword = (
        {"dtype": torch.bfloat16}
        if int(transformers.__version__.split(".", maxsplit=1)[0]) >= 5
        else {"torch_dtype": torch.bfloat16}
    )
    model = AutoModel.from_pretrained(
        model_path,
        config=serialized_config,
        device_map={"": device},
        low_cpu_mem_usage=True,
        local_files_only=True,
        **dtype_keyword,
    )
    model.eval()
    model.config.use_cache = False
    layers = model.layers if hasattr(model, "layers") else model.model.layers
    if len(layers) != config["model"]["config_signature"]["num_hidden_layers"]:
        raise RuntimeError(f"unexpected layer count: {len(layers)}")

    projections = tuple(config["spectrum"]["projections"])
    module_names = {"q": "q_proj", "k": "k_proj", "v": "v_proj"}
    group_count = int(config["spectrum"]["frequency_groups"])
    accumulators = {
        projection: torch.zeros((len(layers), group_count), dtype=torch.float64, device=device)
        for projection in projections
    }
    handles = []
    for layer_index, layer in enumerate(layers):
        for projection in projections:
            module = getattr(layer.self_attn, module_names[projection])

            def capture(_module, _inputs, output, *, i=layer_index, p=projection):
                energy = grouped_projected_power(output.detach(), group_count)
                accumulators[p][i].add_(energy.to(dtype=torch.float64))

            handles.append(module.register_forward_hook(capture))

    try:
        with torch.inference_mode():
            for window_index in range(requested_windows):
                start = start_token + window_index * sequence_length
                end = start + sequence_length
                input_ids = encoded[:, start:end].to(device)
                model(input_ids=input_ids, use_cache=False, return_dict=True)
                print(
                    f"measured {window_index + 1}/{requested_windows} windows",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        for handle in handles:
            handle.remove()

    curves: dict[str, list[list[float]]] = {}
    for projection in projections:
        curves[projection] = [
            (accumulators[projection][layer_index] / requested_windows).cpu().tolist()
            for layer_index in range(len(layers))
        ]

    audit = audit_measured_spectra(
        curves,
        target_manifest,
        relative_threshold=config["spectrum"]["dominant_frequency"][
            "relative_to_global_peak"
        ],
    )
    validation_eligible = not args.smoke
    report = {
        "run_id": "run_019" if validation_eligible else "smoke_h16",
        "hypothesis": "H16",
        "classification": (
            config["classification"] if validation_eligible else "runtime_smoke_not_an_experiment"
        ),
        "validation_eligible": validation_eligible,
        "git_commit": _git_commit(),
        "protocol": config,
        "model_qualification": model_qualification,
        "target_source_checks": source_checks,
        "dataset": {
            "path": config["dataset"]["path"],
            "sha256": dataset_hash,
            "rows": dataset_rows,
            "token_count": int(encoded.shape[1]),
        },
        "sampling": {
            **config["sampling"],
            "window_count": requested_windows,
        },
        "curves": curves,
        "audit": audit if validation_eligible else None,
        "smoke_audit_preview": audit if args.smoke else None,
        "runtime": {
            "wall_time_seconds": time.perf_counter() - started,
            "device": str(device),
            "cuda_device": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "model_dtype": str(next(model.parameters()).dtype),
        },
    }
    encoded_report = json.dumps(report, indent=2, sort_keys=True)
    if args.output or not args.smoke:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded_report + "\n", encoding="utf-8")
    print(json.dumps(audit["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
