#!/usr/bin/env python3
"""Compare all-token and chunk-end-only PPL for the run023 FFT-LoRA adapter."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch
import yaml
from peft import PeftModel
from torch.nn import functional
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from mlxsim.llama_fft import install_compressed_attention, leakage_free_prediction_positions
from mlxsim.llama_perplexity import complete_window_ranges, sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/training/llama2_fft_lora_s075_v1.yaml"
DEFAULT_RESULT = PROJECT_ROOT / "artifacts/results/llama2-fft-lora-s075-run023.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/environment/llama2-fft-run023-causality.json"


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={PROJECT_ROOT}", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    if output.exists():
        raise SystemExit(f"refusing to overwrite causality audit: {output}")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base = yaml.safe_load(
        (PROJECT_ROOT / config["base_evaluation_config"]).read_text(encoding="utf-8")
    )
    result = json.loads(args.result.read_text(encoding="utf-8"))
    checkpoint = PROJECT_ROOT / result["checkpoint"]["path"]
    expected_adapter_hash = result["checkpoint"]["files"]["adapter_model.safetensors"]["sha256"]
    actual_adapter_hash = sha256_file(checkpoint / "adapter_model.safetensors")
    if actual_adapter_hash != expected_adapter_hash:
        raise RuntimeError("run023 adapter hash mismatch")

    dataset_path = PROJECT_ROOT / base["dataset"]["path"]
    rows = pq.read_table(dataset_path, columns=["text"]).column("text").to_pylist()
    text = base["dataset"]["row_separator"].join(str(row) for row in rows)
    model_path = PROJECT_ROOT / base["model"]["path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, local_files_only=True)
    token_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    sequence_length = int(config["training_data"]["sequence_length"])
    ranges = complete_window_ranges(int(token_ids.shape[1]), sequence_length)

    device = torch.device(config["optimization"]["device"])
    model_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=model_config,
        device_map={"": device},
        low_cpu_mem_usage=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    structured = config["structured_attention"]
    install_compressed_attention(
        model,
        layer_indices=list(structured["modified_layer_indices"]),
        chunk_length=int(structured["chunk_length"]),
        compression_ratio=float(structured["compression_ratio"]),
    )
    model = PeftModel.from_pretrained(model, checkpoint, is_trainable=False)
    model.eval()
    model.config.use_cache = False
    safe_positions = leakage_free_prediction_positions(
        sequence_length, int(structured["chunk_length"])
    )
    safe_index = torch.tensor(safe_positions, dtype=torch.long, device=device)
    full_nll = 0.0
    full_tokens = 0
    safe_nll = 0.0
    safe_tokens = 0
    with torch.inference_mode():
        for index, (start, end) in enumerate(ranges, start=1):
            input_ids = token_ids[:, start:end].to(device)
            logits = model(input_ids=input_ids, use_cache=False).logits.float()
            full_logits = logits[:, :-1, :]
            full_labels = input_ids[:, 1:]
            full_nll += float(
                functional.cross_entropy(
                    full_logits.reshape(-1, full_logits.shape[-1]),
                    full_labels.reshape(-1),
                    reduction="sum",
                ).item()
            )
            full_tokens += int(full_labels.numel())
            selected_logits = logits.index_select(1, safe_index)
            selected_labels = input_ids.index_select(1, safe_index + 1)
            safe_nll += float(
                functional.cross_entropy(
                    selected_logits.reshape(-1, selected_logits.shape[-1]),
                    selected_labels.reshape(-1),
                    reduction="sum",
                ).item()
            )
            safe_tokens += int(selected_labels.numel())
            if index % 50 == 0 or index == len(ranges):
                print(f"audited {index}/{len(ranges)} windows", file=sys.stderr, flush=True)

    full_ppl = math.exp(full_nll / full_tokens)
    safe_ppl = math.exp(safe_nll / safe_tokens)
    expected_full_ppl = float(result["post_training_evaluation"]["perplexity"])
    full_reproduction_error = abs(full_ppl - expected_full_ppl) / expected_full_ppl
    report = {
        "classification": "posthoc_causality_diagnostic_not_target_validation",
        "git_commit": _git_commit(),
        "source_result": str(args.result.relative_to(PROJECT_ROOT)),
        "adapter": {
            "path": str(checkpoint.relative_to(PROJECT_ROOT)),
            "expected_sha256": expected_adapter_hash,
            "actual_sha256": actual_adapter_hash,
            "pass": actual_adapter_hash == expected_adapter_hash,
        },
        "window_count": len(ranges),
        "chunk_length": int(structured["chunk_length"]),
        "leakage_free_logit_positions_per_window": safe_positions,
        "all_token": {
            "predicted_tokens": full_tokens,
            "total_nll": full_nll,
            "perplexity": full_ppl,
            "run023_perplexity": expected_full_ppl,
            "relative_reproduction_error": full_reproduction_error,
        },
        "chunk_end_only": {
            "predicted_tokens": safe_tokens,
            "total_nll": safe_nll,
            "perplexity": safe_ppl,
        },
        "perplexity_ratio_chunk_end_to_all": safe_ppl / full_ppl,
        "pass": full_reproduction_error <= 1e-6,
        "interpretation": "Chunk-end logits do not include their next-token label in the same FFT chunk; other logit positions may mix later inputs.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "all_token_perplexity": full_ppl,
                "chunk_end_perplexity": safe_ppl,
                "ratio": safe_ppl / full_ppl,
                "pass": report["pass"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
