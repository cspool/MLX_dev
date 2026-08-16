#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from torch.nn import functional
from transformers import AutoModelForCausalLM, AutoTokenizer

from mlxsim.quality import contiguous_window_ranges, perplexity_from_nll


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--dataset-parquet", required=True, type=Path)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_text(path: Path) -> tuple[str, int]:
    table = pq.read_table(path, columns=["text"])
    rows = table.column("text").to_pylist()
    return "\n\n".join(str(row) for row in rows), len(rows)


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the 7B perplexity run")

    text, dataset_rows = _load_text(args.dataset_parquet)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    ranges = contiguous_window_ranges(encoded.shape[1], args.sequence_length)
    if args.max_windows is not None:
        if args.max_windows <= 0:
            raise SystemExit("--max-windows must be positive")
        ranges = ranges[: args.max_windows]

    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        low_cpu_mem_usage=True,
    )
    model.eval()
    if hasattr(model.config, "use_cache"):
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
            predicted_tokens += labels.numel()
            if index % 25 == 0 or index == len(ranges):
                print(f"evaluated {index}/{len(ranges)} windows", file=sys.stderr, flush=True)

    config_fields = (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
    )
    report: dict[str, Any] = {
        "classification": "native-checkpoint-perplexity-evaluation",
        "model_path": str(args.model),
        "model_config": {name: getattr(model.config, name, None) for name in config_fields},
        "dataset_path": str(args.dataset_parquet),
        "dataset_sha256": _sha256(args.dataset_parquet),
        "dataset_rows": dataset_rows,
        "token_count": int(encoded.shape[1]),
        "sequence_length": args.sequence_length,
        "windows": len(ranges),
        "predicted_tokens": predicted_tokens,
        "total_negative_log_likelihood": total_nll,
        "perplexity": perplexity_from_nll(total_nll, predicted_tokens),
        "dtype": "torch.bfloat16",
        "device": str(device),
        "window_policy": "contiguous non-overlapping; cross-window transitions excluded",
        "max_windows": args.max_windows,
    }
    encoded_report = json.dumps(report, indent=2, sort_keys=True)
    print(encoded_report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{encoded_report}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
