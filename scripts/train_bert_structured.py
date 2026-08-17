#!/usr/bin/env python3
"""Train the pre-registered H15 structured BERT last-k sweep."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
import yaml
from datasets import Dataset
from train_bert_squad import (
    _load_squad,
    _postprocess_answers,
    _prepare_train_features,
    _prepare_validation_features,
)
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from mlxsim.bert_structured import inject_structured_bert_layers, structured_parameter_summary
from mlxsim.quality import squad_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/training/bert_structured_sweep_v1.yaml"


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, expected: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")


def _git_commit() -> str:
    return subprocess.run(
        ["git", "-c", f"safe.directory={PROJECT_ROOT}", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--k", type=int, action="append", dest="selected_k")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for H15")
    model_path = _resolve(config["model"]["path"])
    train_path = _resolve(config["dataset"]["train"])
    validation_path = _resolve(config["dataset"]["validation"])
    report_path = args.report or _resolve(config["outputs"]["report"])
    checkpoint_root = _resolve(config["outputs"]["checkpoint_root"])
    _verify(model_path / "model.safetensors", config["model"]["weights_sha256"])
    _verify(train_path, config["dataset"]["train_sha256"])
    _verify(validation_path, config["dataset"]["validation_sha256"])
    if not args.smoke and report_path.exists():
        raise SystemExit(f"refusing to overwrite official report: {report_path}")

    all_k = config["structured"]["modified_last_k_layers"]
    selected_k = args.selected_k or all_k
    if any(k not in all_k for k in selected_k):
        raise SystemExit(f"--k must be selected from {all_k}")
    if args.smoke:
        selected_k = selected_k[:1]

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
    collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=preprocessing["pad_to_multiple_of"]
    )

    optimization = config["optimization"]
    structured = config["structured"]
    evaluation = config["evaluation"]
    target_index = {
        k: index for index, k in enumerate(config["paper_targets"]["modified_last_k_layers"])
    }
    sweep_started = time.perf_counter()
    results: list[dict[str, Any]] = []
    for k in selected_k:
        run_started = time.perf_counter()
        seed = optimization["seed_base"] + k
        set_seed(seed)
        model = AutoModelForQuestionAnswering.from_pretrained(model_path).to("cuda:0")
        fit_reports = inject_structured_bert_layers(
            model,
            modified_last_k_layers=k,
            compression_ratio=structured["compression_ratio"],
            chunk_length=structured["chunk_length"],
            block_size=structured["butterfly_block_size"],
            fit_steps=structured["factor_fit"]["steps"],
            fit_learning_rate=structured["factor_fit"]["learning_rate"],
            fit_seed_base=structured["factor_fit"]["seed_base"],
        )
        set_seed(seed)
        output_dir = checkpoint_root / f"k{k}"
        if not args.smoke and output_dir.exists() and any(output_dir.iterdir()):
            raise SystemExit(f"refusing to overwrite checkpoint: {output_dir}")
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            do_train=True,
            per_device_train_batch_size=optimization["per_device_train_batch_size"],
            gradient_accumulation_steps=optimization["gradient_accumulation_steps"],
            num_train_epochs=optimization["epochs"],
            max_steps=2 if args.smoke else -1,
            learning_rate=optimization["learning_rate"],
            lr_scheduler_type=optimization["scheduler"],
            warmup_ratio=optimization["warmup_ratio"],
            optim=optimization["optimizer"],
            weight_decay=optimization["weight_decay"],
            max_grad_norm=optimization["max_gradient_norm"],
            bf16=optimization["dtype"] == "bfloat16",
            tf32=optimization["tf32"],
            seed=seed,
            data_seed=seed,
            per_device_eval_batch_size=evaluation["per_device_batch_size"],
            dataloader_num_workers=optimization["dataloader_workers"],
            eval_strategy="no",
            save_strategy="no",
            logging_steps=1 if args.smoke else 100,
            report_to="none",
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_features,
            data_collator=collator,
            processing_class=tokenizer,
        )
        train_result = trainer.train()
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
        index = target_index[k]
        targets = {
            "f1": config["paper_targets"]["f1_pct"][index],
            "exact_match": config["paper_targets"]["exact_match_pct"][index],
        }
        relative_errors = {
            name: abs(metrics[name] - target) / target for name, target in targets.items()
        }
        checkpoint_sha256 = None
        if not args.smoke:
            trainer.save_model(str(output_dir))
            tokenizer.save_pretrained(output_dir)
            checkpoint_sha256 = _sha256(output_dir / "model.safetensors")
        results.append(
            {
                "modified_last_k_layers": k,
                "modified_layer_indices": list(range(12 - k, 12)),
                "seed": seed,
                "metrics_pct": metrics,
                "paper_targets_pct": targets,
                "relative_errors": relative_errors,
                "maximum_relative_error": max(relative_errors.values()),
                "passes_10pct_gate": max(relative_errors.values())
                <= config["paper_targets"]["maximum_relative_error"],
                "factor_fit": {
                    "projection_count": len(fit_reports),
                    "mean_final_relative_mse": float(
                        np.mean([report["final_relative_mse"] for report in fit_reports])
                    ),
                    "max_final_relative_mse": max(
                        report["final_relative_mse"] for report in fit_reports
                    ),
                    "reports": fit_reports,
                },
                "parameter_summary": structured_parameter_summary(model),
                "train_metrics": train_result.metrics,
                "prediction_metrics": prediction_output.metrics,
                "checkpoint_sha256": checkpoint_sha256,
                "wall_time_seconds": time.perf_counter() - run_started,
            }
        )
        del trainer, model, prediction_output, start_logits, end_logits
        gc.collect()
        torch.cuda.empty_cache()

    all_errors = [
        error for result in results for error in result["relative_errors"].values()
    ]
    official = not args.smoke and selected_k == all_k
    report = {
        "run_id": "run_018" if official else None,
        "hypothesis": "H15",
        "classification": (
            config["classification"] if official else "partial_or_smoke_not_an_experiment"
        ),
        "validation_eligible": official,
        "git_commit": _git_commit(),
        "model_input_sha256": config["model"]["weights_sha256"],
        "dataset_train_sha256": config["dataset"]["train_sha256"],
        "dataset_validation_sha256": config["dataset"]["validation_sha256"],
        "train_examples": len(train_examples),
        "train_features": len(train_features),
        "validation_examples": len(validation_examples),
        "validation_features": len(validation_features),
        "results": results,
        "summary": {
            "point_count": len(all_errors),
            "mape": sum(all_errors) / len(all_errors),
            "max_relative_error": max(all_errors),
            "all_points_pass": all(result["passes_10pct_gate"] for result in results),
        },
        "protocol": config,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "wall_time_seconds": time.perf_counter() - sweep_started,
        "smoke": args.smoke,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if official or args.report:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
