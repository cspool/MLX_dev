#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
import yaml
from datasets import Dataset
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from mlxsim.quality import squad_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "training" / "bert_squad_baseline_v1.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run two update steps on small deterministic subsets; never an official result",
    )
    return parser.parse_args()


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file(path: Path, expected_sha256: str) -> None:
    actual = _sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(f"SHA-256 mismatch for {path}: expected {expected_sha256}, got {actual}")


def _load_squad(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    examples: list[dict[str, Any]] = []
    for article in payload["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                examples.append(
                    {
                        "id": qa["id"],
                        "question": qa["question"],
                        "context": context,
                        "answers": {
                            "text": [answer["text"] for answer in qa["answers"]],
                            "answer_start": [answer["answer_start"] for answer in qa["answers"]],
                        },
                    }
                )
    return examples


def _prepare_train_features(
    examples: dict[str, list[Any]],
    *,
    tokenizer: Any,
    max_length: int,
    stride: int,
) -> dict[str, Any]:
    questions = [question.lstrip() for question in examples["question"]]
    tokenized = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=False,
    )
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")
    tokenized["start_positions"] = []
    tokenized["end_positions"] = []

    for feature_index, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][feature_index]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized.sequence_ids(feature_index)
        sample_index = sample_mapping[feature_index]
        answers = examples["answers"][sample_index]
        if not answers["answer_start"]:
            tokenized["start_positions"].append(cls_index)
            tokenized["end_positions"].append(cls_index)
            continue

        start_character = answers["answer_start"][0]
        end_character = start_character + len(answers["text"][0])
        token_start = 0
        while sequence_ids[token_start] != 1:
            token_start += 1
        token_end = len(input_ids) - 1
        while sequence_ids[token_end] != 1:
            token_end -= 1

        if offsets[token_start][0] > start_character or offsets[token_end][1] < end_character:
            tokenized["start_positions"].append(cls_index)
            tokenized["end_positions"].append(cls_index)
            continue
        while token_start < len(offsets) and offsets[token_start][0] <= start_character:
            token_start += 1
        while offsets[token_end][1] >= end_character:
            token_end -= 1
        tokenized["start_positions"].append(token_start - 1)
        tokenized["end_positions"].append(token_end + 1)
    return tokenized


def _prepare_validation_features(
    examples: dict[str, list[Any]],
    *,
    tokenizer: Any,
    max_length: int,
    stride: int,
) -> dict[str, Any]:
    questions = [question.lstrip() for question in examples["question"]]
    tokenized = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=False,
    )
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    tokenized["example_id"] = []
    for feature_index, sample_index in enumerate(sample_mapping):
        sequence_ids = tokenized.sequence_ids(feature_index)
        tokenized["example_id"].append(examples["id"][sample_index])
        tokenized["offset_mapping"][feature_index] = [
            offset if sequence_ids[token_index] == 1 else None
            for token_index, offset in enumerate(tokenized["offset_mapping"][feature_index])
        ]
    return tokenized


def _postprocess_answers(
    examples: Dataset,
    features: Dataset,
    start_logits: np.ndarray,
    end_logits: np.ndarray,
    *,
    n_best_size: int,
    max_answer_length: int,
) -> dict[str, str]:
    feature_indices: dict[str, list[int]] = defaultdict(list)
    for index, feature in enumerate(features):
        feature_indices[feature["example_id"]].append(index)

    predictions: dict[str, str] = {}
    for example in examples:
        best_score = -float("inf")
        best_answer = ""
        for feature_index in feature_indices[example["id"]]:
            offsets = features[feature_index]["offset_mapping"]
            starts = np.argsort(start_logits[feature_index])[-n_best_size:][::-1]
            ends = np.argsort(end_logits[feature_index])[-n_best_size:][::-1]
            for start_index in starts:
                for end_index in ends:
                    if start_index >= len(offsets) or end_index >= len(offsets):
                        continue
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    score = float(
                        start_logits[feature_index][start_index]
                        + end_logits[feature_index][end_index]
                    )
                    if score > best_score:
                        start_character = offsets[start_index][0]
                        end_character = offsets[end_index][1]
                        best_score = score
                        best_answer = example["context"][start_character:end_character]
        predictions[example["id"]] = best_answer
    return predictions


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "-c", f"safe.directory={PROJECT_ROOT}", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def main() -> int:
    args = _parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_path = _resolve(config["model"]["path"])
    train_path = _resolve(config["dataset"]["train"])
    validation_path = _resolve(config["dataset"]["validation"])
    output_dir = args.output_dir or _resolve(config["outputs"]["checkpoint_dir"])
    report_path = args.report or _resolve(config["outputs"]["report"])

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the frozen BERT/SQuAD run")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {output_dir}")
    if not args.smoke and report_path.exists():
        raise SystemExit(f"refusing to overwrite existing official report: {report_path}")

    _verify_file(model_path / "model.safetensors", config["model"]["weights_sha256"])
    _verify_file(train_path, config["dataset"]["train_sha256"])
    _verify_file(validation_path, config["dataset"]["validation_sha256"])
    started = time.perf_counter()

    train_examples_list = _load_squad(train_path)
    validation_examples_list = _load_squad(validation_path)
    if args.smoke:
        train_examples_list = train_examples_list[:128]
        validation_examples_list = validation_examples_list[:64]
    train_examples = Dataset.from_list(train_examples_list)
    validation_examples = Dataset.from_list(validation_examples_list)

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    preprocessing = config["preprocessing"]
    train_features = train_examples.map(
        _prepare_train_features,
        batched=True,
        remove_columns=train_examples.column_names,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": preprocessing["max_sequence_length"],
            "stride": preprocessing["document_stride"],
        },
        desc="tokenizing SQuAD train",
    )
    validation_features = validation_examples.map(
        _prepare_validation_features,
        batched=True,
        remove_columns=validation_examples.column_names,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": preprocessing["max_sequence_length"],
            "stride": preprocessing["document_stride"],
        },
        desc="tokenizing SQuAD validation",
    )
    model_validation_features = validation_features.remove_columns(
        ["example_id", "offset_mapping"]
    )

    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    optimization = config["optimization"]
    evaluation = config["evaluation"]
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        do_train=True,
        per_device_train_batch_size=optimization["per_device_train_batch_size"],
        gradient_accumulation_steps=optimization["gradient_accumulation_steps"],
        num_train_epochs=optimization["epochs"],
        max_steps=2 if args.smoke else -1,
        learning_rate=optimization["learning_rate"],
        lr_scheduler_type=optimization["scheduler"],
        warmup_steps=optimization["warmup_ratio"],
        optim=optimization["optimizer"],
        weight_decay=optimization["weight_decay"],
        max_grad_norm=optimization["max_gradient_norm"],
        bf16=optimization["dtype"] == "bfloat16",
        tf32=optimization["tf32"],
        seed=optimization["seed"],
        data_seed=optimization["seed"],
        per_device_eval_batch_size=evaluation["per_device_batch_size"],
        dataloader_num_workers=optimization["dataloader_workers"],
        eval_strategy="no",
        save_strategy="no",
        logging_steps=1 if args.smoke else 100,
        report_to="none",
    )
    collator = DataCollatorWithPadding(
        tokenizer,
        pad_to_multiple_of=preprocessing["pad_to_multiple_of"],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_features,
        data_collator=collator,
        processing_class=tokenizer,
    )
    train_result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)

    prediction_output = trainer.predict(model_validation_features)
    start_logits, end_logits = prediction_output.predictions[:2]
    predictions = _postprocess_answers(
        validation_examples,
        validation_features,
        start_logits,
        end_logits,
        n_best_size=evaluation["n_best_size"],
        max_answer_length=evaluation["max_answer_length"],
    )
    references = {
        example["id"]: example["answers"]["text"] for example in validation_examples
    }
    metrics = squad_metrics(predictions, references)
    targets = config["paper_targets"]
    relative_errors = {
        "f1": abs(metrics["f1"] - targets["f1_pct"]) / targets["f1_pct"],
        "exact_match": (
            abs(metrics["exact_match"] - targets["exact_match_pct"])
            / targets["exact_match_pct"]
        ),
    }
    official = not args.smoke
    model_output = output_dir / "model.safetensors"
    report: dict[str, Any] = {
        "classification": (
            config["classification"] if official else "runtime_smoke_not_an_experiment"
        ),
        "experiment_id": config["experiment_id"],
        "git_commit": _git_commit(),
        "model_revision": config["model"]["revision"],
        "model_input_sha256": config["model"]["weights_sha256"],
        "model_output_sha256": _sha256(model_output),
        "dataset_train_sha256": config["dataset"]["train_sha256"],
        "dataset_validation_sha256": config["dataset"]["validation_sha256"],
        "train_examples": len(train_examples),
        "train_features": len(train_features),
        "validation_examples": len(validation_examples),
        "validation_features": len(validation_features),
        "metrics_pct": metrics,
        "paper_targets_pct": {
            "f1": targets["f1_pct"],
            "exact_match": targets["exact_match_pct"],
        },
        "relative_errors": relative_errors,
        "maximum_relative_error": max(relative_errors.values()),
        "passes_10pct_gate": max(relative_errors.values()) <= targets["maximum_relative_error"],
        "train_metrics": train_result.metrics,
        "prediction_metrics": prediction_output.metrics,
        "wall_time_seconds": time.perf_counter() - started,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "protocol": config,
        "smoke": args.smoke,
    }
    encoded_report = json.dumps(report, indent=2, sort_keys=True)
    print(encoded_report)
    if official or args.report:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(f"{encoded_report}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
