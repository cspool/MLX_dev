#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import accelerate
import einops
import google.protobuf
import huggingface_hub
import numpy
import pyarrow
import safetensors
import sentencepiece
import tokenizers
import torch
import transformers

EXPECTED = {
    "accelerate": "0.30.1",
    "einops": "0.8.1",
    "huggingface_hub": "0.36.2",
    "numpy": "2.4.6",
    "protobuf": "6.33.6",
    "pyarrow": "25.0.1",
    "safetensors": "0.8.0",
    "sentencepiece": "0.2.1",
    "tokenizers": "0.19.1",
    "torch": "2.12.0+cu132",
    "transformers": "4.41.0",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    versions = {
        "accelerate": accelerate.__version__,
        "einops": einops.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "numpy": numpy.__version__,
        "protobuf": google.protobuf.__version__,
        "pyarrow": pyarrow.__version__,
        "safetensors": safetensors.__version__,
        "sentencepiece": sentencepiece.__version__,
        "tokenizers": tokenizers.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    mismatches = {
        name: {"expected": expected, "actual": versions.get(name)}
        for name, expected in EXPECTED.items()
        if versions.get(name) != expected
    }
    report = {
        "classification": "checkpoint-compatible isolated evaluation stack",
        "versions": versions,
        "version_mismatches": mismatches,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cuda_devices": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
        "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{encoded}\n", encoding="utf-8")
    if mismatches:
        raise SystemExit("isolated InternLM evaluation stack has version mismatches")
    if not report["cuda_available"] or not report["bf16_supported"]:
        raise SystemExit("isolated InternLM evaluation stack lacks BF16 CUDA")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
