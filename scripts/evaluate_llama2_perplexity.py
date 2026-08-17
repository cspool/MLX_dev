#!/usr/bin/env python3
"""Evaluate the frozen Llama2-7B WikiText-2 perplexity protocol."""

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
from torch.nn import functional
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from mlxsim.llama_perplexity import (
    audit_perplexity,
    complete_window_ranges,
    qualify_model_files,
    sha256_file,
    window_accounting,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/llama2_perplexity_v1.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true", help="score one window only")
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={PROJECT_ROOT}", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _load_text(path: Path, column: str, separator: str) -> tuple[str, int]:
    table = pq.read_table(path, columns=[column])
    rows = table.column(column).to_pylist()
    return separator.join(str(row) for row in rows), len(rows)


def main() -> int:
    args = _parse_args()
    config = _load_yaml(args.config)
    official = not args.smoke
    output = args.output or PROJECT_ROOT / config["run"]["output"]
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    if official and output.exists():
        raise SystemExit(f"refusing to overwrite official result: {output}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for H19")
    if transformers.__version__ != config["runtime"]["transformers_version"]:
        raise SystemExit(
            "Transformers version mismatch: expected "
            f"{config['runtime']['transformers_version']}, found {transformers.__version__}"
        )

    started = time.perf_counter()
    model_qualification = qualify_model_files(config["model"])
    if not model_qualification["pass"]:
        raise RuntimeError(f"model input qualification failed: {model_qualification}")

    dataset_path = PROJECT_ROOT / config["dataset"]["path"]
    dataset_hash = sha256_file(dataset_path)
    if dataset_hash != config["dataset"]["sha256"]:
        raise RuntimeError(
            f"dataset hash mismatch: expected {config['dataset']['sha256']}, got {dataset_hash}"
        )
    text, row_count = _load_text(
        dataset_path, config["dataset"]["text_column"], config["dataset"]["row_separator"]
    )
    if row_count != int(config["dataset"]["expected_rows"]):
        raise RuntimeError(f"dataset row-count mismatch: expected 4358, got {row_count}")

    model_path = PROJECT_ROOT / config["model"]["path"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=False, local_files_only=True
    )
    encoded = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=config["sampling"]["add_special_tokens"],
    )["input_ids"]
    token_count = int(encoded.shape[1])
    if token_count != int(config["dataset"]["expected_token_count"]):
        raise RuntimeError(
            "token-count canary mismatch: expected "
            f"{config['dataset']['expected_token_count']}, got {token_count}"
        )
    sequence_length = int(config["sampling"]["sequence_length"])
    accounting = window_accounting(token_count, sequence_length)
    if accounting["windows"] != int(config["sampling"]["expected_window_count"]):
        raise RuntimeError(f"window-count mismatch: {accounting}")
    if accounting["predicted_tokens"] != int(config["sampling"]["expected_predicted_tokens"]):
        raise RuntimeError(f"predicted-token mismatch: {accounting}")
    ranges = complete_window_ranges(token_count, sequence_length)
    if len(ranges) != accounting["windows"] or any(
        end - start != sequence_length for start, end in ranges
    ):
        raise RuntimeError(f"complete-window construction mismatch: {accounting}")
    if args.smoke:
        ranges = ranges[:1]

    device = torch.device(config["runtime"]["device"])
    model_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=model_config,
        device_map={"": device},
        low_cpu_mem_usage=True,
        local_files_only=True,
        dtype=torch.bfloat16,
    )
    model.eval()
    model.config.use_cache = False

    total_nll = 0.0
    predicted_tokens = 0
    with torch.inference_mode():
        for index, (start, end) in enumerate(ranges, start=1):
            input_ids = encoded[:, start:end].to(device)
            logits = model(input_ids=input_ids, use_cache=False).logits[:, :-1, :].float()
            labels = input_ids[:, 1:]
            total_nll += float(
                functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    reduction="sum",
                ).item()
            )
            predicted_tokens += int(labels.numel())
            if index % 25 == 0 or index == len(ranges):
                print(f"evaluated {index}/{len(ranges)} windows", file=sys.stderr, flush=True)

    target_manifest = _load_yaml(PROJECT_ROOT / config["target"]["source"])
    canonical_target = float(
        target_manifest["fig15_quality"]["generation_original_perplexity"][
            "llama2_wikitext2_1k"
        ]
    )
    if canonical_target != float(config["target"]["perplexity"]):
        raise RuntimeError(
            f"target mismatch: config {config['target']['perplexity']}, canonical {canonical_target}"
        )
    audit = audit_perplexity(
        total_nll=total_nll,
        predicted_tokens=predicted_tokens,
        target=canonical_target,
        relative_error_gate=float(config["target"]["relative_error_gate"]),
    )
    report = {
        "run_id": config["run"]["id"] if official else "smoke_h19",
        "hypothesis": config["run"]["hypothesis"],
        "classification": (
            config["classification"] if official else "runtime_smoke_not_an_experiment"
        ),
        "validation_eligible": official,
        "git_commit": _git_commit(),
        "protocol": config,
        "model_qualification": model_qualification,
        "dataset": {
            "path": config["dataset"]["path"],
            "sha256": dataset_hash,
            "rows": row_count,
            "token_count": token_count,
        },
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "is_fast": bool(tokenizer.is_fast),
            "add_special_tokens": config["sampling"]["add_special_tokens"],
        },
        "sampling": {
            **config["sampling"],
            "executed_windows": len(ranges),
            "executed_predicted_tokens": predicted_tokens,
            "full_input_accounting": accounting,
        },
        "audit": audit if official else None,
        "smoke_audit_preview": audit if args.smoke else None,
        "runtime": {
            "wall_time_seconds": time.perf_counter() - started,
            "device": str(device),
            "cuda_device": torch.cuda.get_device_name(device),
            "model_dtype": str(next(model.parameters()).dtype),
            "loss_dtype": "torch.float32",
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        },
    }
    summary = {
        "perplexity": audit["perplexity"],
        "relative_error": audit["relative_error"],
        "pass": audit["pass"] if official else None,
        "validation_eligible": official,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if official:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if args.smoke or audit["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
