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
import transformers
import yaml
from torch.nn import functional
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from mlxsim.quality import contiguous_window_ranges, perplexity_from_nll

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = PROJECT_ROOT / "artifacts" / "targets" / "paper_targets.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--dataset-parquet", required=True, type=Path)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
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


def _load_pinned_internlm2_tokenizer(model_path: Path) -> tuple[Any, list[int]]:
    tokenizer_class = get_class_from_dynamic_module(
        "tokenization_internlm2.InternLM2Tokenizer",
        model_path,
    )
    tokenizer = tokenizer_class(vocab_file=str(model_path / "tokenizer.model"))
    canary_text = "The quick brown fox jumps over the lazy dog."
    canary_ids = tokenizer(canary_text, add_special_tokens=False)["input_ids"]
    expected_ids = [918, 4131, 14018, 38648, 34256, 1053, 410, 15810, 5718, 281]
    if canary_ids != expected_ids:
        raise RuntimeError(
            f"pinned InternLM2 tokenizer canary mismatch: expected {expected_ids}, got {canary_ids}"
        )
    return tokenizer, canary_ids


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the 7B perplexity run")
    official = args.max_windows is None
    if official and transformers.__version__ != "4.41.0":
        raise SystemExit(
            "official H10 requires the checkpoint-declared Transformers 4.41.0; "
            f"found {transformers.__version__}"
        )
    if official and args.output is not None and args.output.exists():
        raise SystemExit(f"refusing to overwrite official result: {args.output}")

    text, dataset_rows = _load_text(args.dataset_parquet)
    tokenizer, tokenizer_canary_ids = _load_pinned_internlm2_tokenizer(args.model)
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    ranges = contiguous_window_ranges(encoded.shape[1], args.sequence_length)
    if args.max_windows is not None:
        if args.max_windows <= 0:
            raise SystemExit("--max-windows must be positive")
        ranges = ranges[: args.max_windows]

    device = torch.device(args.device)
    serialized_config = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
    model_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    serialized_null_rope_scaling = serialized_config.get("rope_scaling") is None
    restored_null_rope_scaling = (
        serialized_null_rope_scaling and model_config.rope_scaling is not None
    )
    if serialized_null_rope_scaling:
        # Transformers 5 normalizes null into a new rope_type dictionary, while
        # the pinned InternLM2/Transformers-4.41 remote code expects literal None.
        model_config.rope_scaling = None
    dtype_keyword = (
        {"dtype": torch.bfloat16}
        if int(transformers.__version__.split(".", maxsplit=1)[0]) >= 5
        else {"torch_dtype": torch.bfloat16}
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=model_config,
        trust_remote_code=True,
        device_map={"": device},
        low_cpu_mem_usage=True,
        **dtype_keyword,
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

    paper_targets = yaml.safe_load(args.targets.read_text(encoding="utf-8"))
    target_perplexity = paper_targets["fig15_quality"]["generation_original_perplexity"][
        "internlm2_wikitext2_1k"
    ]
    measured_perplexity = perplexity_from_nll(total_nll, predicted_tokens)
    relative_error = abs(measured_perplexity - target_perplexity) / target_perplexity

    config_fields = (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
    )
    report: dict[str, Any] = {
        "classification": (
            "native-checkpoint-perplexity-evaluation"
            if official
            else "runtime-smoke-not-an-experiment"
        ),
        "model_path": str(args.model),
        "model_config": {name: getattr(model.config, name, None) for name in config_fields},
        "dataset_path": str(args.dataset_parquet),
        "dataset_sha256": _sha256(args.dataset_parquet),
        "dataset_rows": dataset_rows,
        "token_count": int(encoded.shape[1]),
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "implementation": "pinned remote slow tokenizer over tokenizer.model",
            "canary_ids": tokenizer_canary_ids,
        },
        "sequence_length": args.sequence_length,
        "windows": len(ranges),
        "predicted_tokens": predicted_tokens,
        "total_negative_log_likelihood": total_nll,
        "perplexity": measured_perplexity,
        "paper_target_perplexity": target_perplexity,
        "relative_error": relative_error,
        "validation_eligible": official,
        "passes_10pct_gate": relative_error <= 0.10 if official else None,
        "dtype": "torch.bfloat16",
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "transformers_version": transformers.__version__,
        "device": str(device),
        "window_policy": "contiguous non-overlapping; cross-window transitions excluded",
        "runtime_compatibility": {
            "restored_serialized_null_rope_scaling": restored_null_rope_scaling,
        },
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
