#!/usr/bin/env python3
"""Compare the Transformers 5 Llama backend with tokenizer.model SentencePiece IDs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pyarrow.parquet as pq
import sentencepiece as spm
import transformers
import yaml
from transformers import AutoTokenizer

from mlxsim.llama_perplexity import compare_token_sequences, sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/analysis/llama2_perplexity_v1.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/environment/llama2-tokenizer-equivalence.json"


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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    if output.exists():
        raise SystemExit(f"refusing to overwrite compatibility artifact: {output}")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dataset_path = PROJECT_ROOT / config["dataset"]["path"]
    model_path = PROJECT_ROOT / config["model"]["path"]
    rows = pq.read_table(dataset_path, columns=[config["dataset"]["text_column"]])
    text = config["dataset"]["row_separator"].join(
        str(value) for value in rows.column(config["dataset"]["text_column"]).to_pylist()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=False, local_files_only=True
    )
    actual = tokenizer(text, add_special_tokens=False)["input_ids"]
    sentencepiece = spm.SentencePieceProcessor(
        model_file=str(model_path / "tokenizer.model")
    )
    reference = sentencepiece.encode(text, out_type=int)
    comparison = compare_token_sequences(actual, reference)
    report = {
        "classification": "runtime_compatibility_audit_not_experiment",
        "git_commit": _git_commit(),
        "dataset": {
            "path": config["dataset"]["path"],
            "sha256": sha256_file(dataset_path),
        },
        "tokenizer_model": {
            "path": str((model_path / "tokenizer.model").relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(model_path / "tokenizer.model"),
        },
        "transformers_backend": {
            "class": type(tokenizer).__name__,
            "module": type(tokenizer).__module__,
            "is_fast": bool(tokenizer.is_fast),
            "transformers_version": transformers.__version__,
        },
        "sentencepiece_version": spm.__version__,
        "comparison": comparison,
        "pass": comparison["equal"]
        and comparison["actual_count"] == int(config["dataset"]["expected_token_count"]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"pass": report["pass"], **comparison}, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
