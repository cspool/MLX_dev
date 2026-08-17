#!/usr/bin/env python3
"""Train and evaluate the frozen H28 dense WinoGrande LoRA reconstruction."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import datasets
import numpy as np
import peft
import pyarrow.parquet as pq
import torch
import transformers
import yaml
from evaluate_llama2_winogrande import (
    _configure_process_git_safe_directory,
    _qualify_harness,
    _qualify_loaded_task,
    _qualify_runtime,
    _resolve_target,
    _sample_log_text,
)
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from lm_eval.utils import handle_non_serializable
from peft import get_peft_model
from torch.nn import functional
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import get_cosine_schedule_with_warmup

from mlxsim.llama_fft import make_lora_config
from mlxsim.llama_perplexity import qualify_model_files
from mlxsim.winogrande import audit_accuracy, qualify_parquet_dataset
from mlxsim.winogrande_training import (
    audit_lora_training_model,
    collate_pairwise,
    continuation_choice_scores,
    encode_examples,
    qualify_adapter_checkpoint,
    qualify_reloaded_adapter,
    shuffled_selection,
    tokenization_audit,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/training/llama2_winogrande_dense_lora_v1.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--smoke", action="store_true")
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


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _runtime_qualification(config: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    base_report = _qualify_runtime(base)
    checks = {
        "base_runtime": base_report["pass"],
        "peft_version": peft.__version__ == config["lora"]["peft_version"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")
        == config["optimization"]["required_cuda_visible_devices"],
    }
    return {
        "base": base_report,
        "actual_peft_version": peft.__version__,
        "expected_peft_version": config["lora"]["peft_version"],
        "checks": checks,
        "pass": all(checks.values()),
    }


def _qualify_tokenization(audit: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    objective = config["objective"]
    expected = {
        "requests": int(objective["expected_requests"]),
        "min_sequence_tokens": int(objective["expected_min_sequence_tokens"]),
        "max_sequence_tokens": int(objective["expected_max_sequence_tokens"]),
        "max_context_tokens": int(objective["expected_max_context_tokens"]),
        "min_continuation_tokens": int(objective["expected_min_continuation_tokens"]),
        "max_continuation_tokens": int(objective["expected_max_continuation_tokens"]),
        "sha256": objective["tokenization_sha256"],
        "first_request_ids": list(objective["first_request_ids"]),
        "first_request_context_tokens": int(objective["first_request_context_tokens"]),
        "last_request_ids": list(objective["last_request_ids"]),
        "last_request_context_tokens": int(objective["last_request_context_tokens"]),
    }
    checks = {name: audit[name] == value for name, value in expected.items()}
    checks["below_maximum_length"] = (
        audit["max_sequence_tokens"] <= int(objective["maximum_length"])
    )
    return {
        **audit,
        "expected": expected,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _qualify_selection(selection: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    optimization = config["optimization"]
    checks = {
        "selected_count": len(selection["selected_indices"])
        == int(optimization["shuffled_selected_rows"]),
        "dropped_indices": selection["dropped_indices"]
        == list(optimization["dropped_indices"]),
        "selected_hash": selection["selected_indices_sha256_uint32be"]
        == optimization["selected_indices_sha256_uint32be"],
    }
    return {
        "selected_count": len(selection["selected_indices"]),
        "dropped_indices": selection["dropped_indices"],
        "selected_indices_sha256_uint32be": selection[
            "selected_indices_sha256_uint32be"
        ],
        "first_selected_indices": selection["selected_indices"][:10],
        "last_selected_indices": selection["selected_indices"][-10:],
        "checks": checks,
        "pass": all(checks.values()),
    }


def _train_and_save_adapter(
    *,
    config: dict[str, Any],
    base: dict[str, Any],
    examples: list[Any],
    selected_indices: list[int],
    tokenizer: Any,
    checkpoint: Path,
    smoke: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    optimization = config["optimization"]
    lora = config["lora"]
    device = torch.device(optimization["device"])
    model_path = PROJECT_ROOT / base["model"]["path"]
    _set_seed(int(optimization["model_seed"]))
    torch.cuda.reset_peak_memory_stats(device)

    serialized_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=serialized_config,
        device_map={"": device},
        low_cpu_mem_usage=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model = get_peft_model(model, make_lora_config(lora))
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": bool(optimization["gradient_checkpointing_use_reentrant"])
        }
    )
    model.enable_input_require_grads()
    trainable_audit = audit_lora_training_model(
        model,
        expected_layers=list(lora["layers_to_transform"]),
        expected_modules=list(lora["target_modules"]),
        expected_trainable_parameters=int(lora["expected_trainable_parameters"]),
        expected_trainable_tensors=int(lora["expected_trainable_tensors"]),
        expected_total_parameters=int(lora["expected_total_parameters_with_adapter"]),
        maximum_trainable_fraction=float(lora["maximum_trainable_fraction"]),
    )
    if not trainable_audit["pass"]:
        raise RuntimeError(f"LoRA trainable-parameter audit failed: {trainable_audit}")

    training_indices = (
        selected_indices[: int(optimization["effective_batch_examples"])]
        if smoke
        else selected_indices
    )
    micro_batch = int(optimization["micro_batch_examples"])
    accumulation = int(optimization["gradient_accumulation_steps"])
    if len(training_indices) % (micro_batch * accumulation):
        raise RuntimeError("training selection does not form complete effective batches")
    micro_steps = len(training_indices) // micro_batch
    optimizer_steps = micro_steps // accumulation
    if not smoke:
        if micro_steps != int(optimization["micro_steps"]):
            raise RuntimeError(f"micro-step mismatch: {micro_steps}")
        if optimizer_steps != int(optimization["optimizer_steps"]):
            raise RuntimeError(f"optimizer-step mismatch: {optimizer_steps}")

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(optimization["learning_rate"]),
        betas=(float(optimization["adam_beta1"]), float(optimization["adam_beta2"])),
        eps=float(optimization["adam_epsilon"]),
        weight_decay=float(optimization["weight_decay"]),
        fused=bool(optimization["fused_optimizer"]),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(optimization["warmup_steps"]),
        num_training_steps=int(optimization["optimizer_steps"]),
    )

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise RuntimeError("Llama tokenizer exposes no pad or EOS token")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    micro_losses: list[float] = []
    optimizer_history: list[dict[str, float | int]] = []
    completed_optimizer_steps = 0
    started = time.perf_counter()
    for micro_index in range(micro_steps):
        offset = micro_index * micro_batch
        indices = training_indices[offset : offset + micro_batch]
        batch = collate_pairwise([examples[index] for index in indices], pad_token_id=pad_token_id)
        batch = {name: tensor.to(device) for name, tensor in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            ).logits
        scores = continuation_choice_scores(
            logits, batch["input_ids"], batch["continuation_mask"]
        )
        loss = functional.cross_entropy(scores, batch["targets"])
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite loss at micro-step {micro_index + 1}: {loss}")
        micro_losses.append(float(loss.detach()))
        (loss / accumulation).backward()

        if (micro_index + 1) % accumulation == 0:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters, float(optimization["max_grad_norm"])
            )
            learning_rate = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            completed_optimizer_steps += 1
            if (
                completed_optimizer_steps == 1
                or completed_optimizer_steps % int(optimization["log_optimizer_steps"]) == 0
                or completed_optimizer_steps == optimizer_steps
            ):
                recent = micro_losses[-accumulation:]
                record = {
                    "optimizer_step": completed_optimizer_steps,
                    "mean_micro_loss": sum(recent) / len(recent),
                    "last_micro_loss": recent[-1],
                    "gradient_norm_before_clip": float(gradient_norm),
                    "learning_rate_used": learning_rate,
                    "learning_rate_next": float(optimizer.param_groups[0]["lr"]),
                }
                optimizer_history.append(record)
                print(
                    f"H28 trained {completed_optimizer_steps}/{optimizer_steps} optimizer steps; "
                    f"loss={record['mean_micro_loss']:.5f}; "
                    f"lr={record['learning_rate_next']:.3e}",
                    flush=True,
                )

    training = {
        "executed_examples": len(training_indices),
        "micro_batch_examples": micro_batch,
        "gradient_accumulation_steps": accumulation,
        "micro_steps": micro_steps,
        "optimizer_steps": completed_optimizer_steps,
        "initial_micro_loss": micro_losses[0],
        "final_micro_loss": micro_losses[-1],
        "mean_micro_loss": sum(micro_losses) / len(micro_losses),
        "minimum_micro_loss": min(micro_losses),
        "maximum_micro_loss": max(micro_losses),
        "all_losses_finite": all(math.isfinite(value) for value in micro_losses),
        "optimizer_history": optimizer_history,
        "wall_time_seconds": time.perf_counter() - started,
    }
    model.save_pretrained(checkpoint, safe_serialization=True)
    checkpoint_qualification = qualify_adapter_checkpoint(
        checkpoint,
        expected_layers=list(lora["layers_to_transform"]),
        expected_modules=list(lora["target_modules"]),
        expected_rank=int(lora["rank"]),
        expected_alpha=int(lora["alpha"]),
        expected_dropout=float(lora["dropout"]),
        expected_parameters=int(lora["expected_trainable_parameters"]),
        expected_tensors=int(lora["expected_trainable_tensors"]),
    )
    if not checkpoint_qualification["pass"]:
        raise RuntimeError(f"adapter checkpoint qualification failed: {checkpoint_qualification}")
    runtime = {
        "training_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "cuda_device": torch.cuda.get_device_name(device),
    }

    del batch, logits, scores, loss, model, optimizer, scheduler, trainable_parameters
    gc.collect()
    torch.cuda.empty_cache()
    return trainable_audit, training, {**checkpoint_qualification, **runtime}


def _evaluate_adapter(
    *,
    config: dict[str, Any],
    base: dict[str, Any],
    checkpoint: Path,
    task_manager: TaskManager,
    target_accuracy_pct: float,
    smoke: bool,
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    evaluation = base["evaluation"]
    task_name = base["harness"]["task_name"]
    model_path = (PROJECT_ROOT / base["model"]["path"]).resolve()
    model_args = {
        "pretrained": str(model_path),
        "revision": base["model"]["official_revision"],
        "dtype": base["runtime"]["model_dtype"],
        "max_length": int(evaluation["max_length"]),
        "use_fast_tokenizer": bool(evaluation["use_fast_tokenizer"]),
        "add_bos_token": evaluation["add_bos_token"],
        "parallelize": False,
        "trust_remote_code": False,
        "local_files_only": True,
        "peft": str(checkpoint.resolve()),
    }
    evaluation_model = HFLM(
        **model_args,
        batch_size=int(evaluation["batch_size"]),
        device=base["runtime"]["device"],
    )
    reload_qualification = qualify_reloaded_adapter(
        evaluation_model.model,
        checkpoint,
        expected_layers=list(config["lora"]["layers_to_transform"]),
        expected_modules=list(config["lora"]["target_modules"]),
        expected_rank=int(config["lora"]["rank"]),
        expected_alpha=int(config["lora"]["alpha"]),
        expected_dropout=float(config["lora"]["dropout"]),
        expected_parameters=int(config["lora"]["expected_trainable_parameters"]),
        expected_tensors=int(config["lora"]["expected_trainable_tensors"]),
    )
    if not reload_qualification["pass"]:
        raise RuntimeError(f"adapter reload qualification failed: {reload_qualification}")

    results = simple_evaluate(
        model=evaluation_model,
        model_args=model_args,
        tasks=[task_name],
        num_fewshot=int(evaluation["num_fewshot"]),
        batch_size=int(evaluation["batch_size"]),
        device=base["runtime"]["device"],
        cache_requests=bool(evaluation["cache_requests"]),
        limit=1 if smoke else None,
        bootstrap_iters=int(evaluation["bootstrap_iters"]),
        write_out=bool(evaluation["write_out"]),
        log_samples=bool(evaluation["log_samples"]),
        system_instruction=evaluation["system_instruction"],
        apply_chat_template=bool(evaluation["apply_chat_template"]),
        fewshot_as_multiturn=bool(evaluation["fewshot_as_multiturn"]),
        task_manager=task_manager,
        verbosity="INFO",
        random_seed=int(evaluation["random_seed"]),
        numpy_random_seed=int(evaluation["numpy_random_seed"]),
        torch_random_seed=int(evaluation["torch_random_seed"]),
        fewshot_random_seed=int(evaluation["fewshot_random_seed"]),
    )
    if results is None:
        raise RuntimeError("lm-eval returned no rank-zero adapter result")
    samples = results["samples"][task_name]
    aggregate = float(results["results"][task_name]["acc,none"])
    audit = audit_accuracy(
        sample_values=[float(sample["acc"]) for sample in samples],
        aggregate_accuracy=aggregate,
        paper_target_pct=target_accuracy_pct,
        relative_error_gate=float(config["evaluation"]["relative_error_gate"]),
    )
    expected_samples = 1 if smoke else int(config["evaluation"]["expected_samples"])
    audit["expected_sample_count"] = expected_samples
    audit["sample_count_pass"] = len(samples) == expected_samples
    audit["pass"] = bool(audit["pass"] and audit["sample_count_pass"])
    sample_text = _sample_log_text(samples)
    sample_record = {
        "records": len(samples),
        "sha256": hashlib.sha256(sample_text.encode("utf-8")).hexdigest(),
    }
    harness_results = {key: value for key, value in results.items() if key != "samples"}
    del evaluation_model
    gc.collect()
    torch.cuda.empty_cache()
    return (
        audit,
        sample_text,
        {"sample_log": sample_record, "results": harness_results},
        reload_qualification,
    )


def main() -> int:
    args = _parse_args()
    config = _load_yaml(args.config)
    base = _load_yaml(PROJECT_ROOT / config["base_evaluation_config"])
    official = not args.smoke
    output = PROJECT_ROOT / config["run"]["output"]
    samples_output = PROJECT_ROOT / config["run"]["samples_output"]
    checkpoint = PROJECT_ROOT / config["run"]["checkpoint"]
    incomplete_checkpoint = checkpoint.with_name(checkpoint.name + ".incomplete")
    if official:
        existing = [
            path
            for path in (output, samples_output, checkpoint, incomplete_checkpoint)
            if path.exists()
        ]
        if existing:
            raise SystemExit(f"refusing to overwrite H28 artifacts: {existing}")

    started = time.perf_counter()
    _configure_process_git_safe_directory()
    runtime_qualification = _runtime_qualification(config, base)
    harness_qualification = _qualify_harness(base)
    model_qualification = qualify_model_files(base["model"])
    train_path = PROJECT_ROOT / config["training_data"]["path"]
    train_dataset_qualification = qualify_parquet_dataset(
        train_path, config["training_data"]
    )
    validation_path = PROJECT_ROOT / base["dataset"]["qualification_path"]
    validation_dataset_qualification = qualify_parquet_dataset(
        validation_path, base["dataset"]
    )
    preflight = {
        "runtime": runtime_qualification["pass"],
        "harness": harness_qualification["pass"],
        "model": model_qualification["pass"],
        "train_dataset": train_dataset_qualification["pass"],
        "validation_dataset": validation_dataset_qualification["pass"],
    }
    if not all(preflight.values()):
        raise RuntimeError(f"H28 preflight qualification failed: {preflight}")

    task_manager = TaskManager(include_path=PROJECT_ROOT / base["harness"]["include_path"])
    loaded = task_manager.load([base["harness"]["task_name"]])
    task_qualification = _qualify_loaded_task(
        loaded["tasks"][base["harness"]["task_name"]], base
    )
    if not task_qualification["pass"]:
        raise RuntimeError(f"H28 validation task qualification failed: {task_qualification}")

    rows = pq.read_table(train_path).to_pylist()
    model_path = PROJECT_ROOT / base["model"]["path"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=False, local_files_only=True
    )
    examples = encode_examples(
        rows, tokenizer, target_delimiter=config["objective"]["target_delimiter"]
    )
    tokenization_qualification = _qualify_tokenization(
        tokenization_audit(examples), config
    )
    if not tokenization_qualification["pass"]:
        raise RuntimeError(f"H28 tokenization qualification failed: {tokenization_qualification}")

    selection = shuffled_selection(
        int(config["optimization"]["source_rows"]),
        int(config["optimization"]["shuffled_selected_rows"]),
        int(config["optimization"]["shuffle_seed"]),
    )
    selection_qualification = _qualify_selection(selection, config)
    if not selection_qualification["pass"]:
        raise RuntimeError(f"H28 selection qualification failed: {selection_qualification}")

    target_manifest = _load_yaml(PROJECT_ROOT / config["evaluation"]["target_source"])
    canonical_target = float(
        _resolve_target(target_manifest, config["evaluation"]["target_key"])
    )
    if canonical_target != float(config["evaluation"]["target_accuracy_pct"]):
        raise RuntimeError("H28 target does not match the canonical manifest")

    temporary_context: tempfile.TemporaryDirectory[str] | None = None
    if official:
        incomplete_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        training_checkpoint = incomplete_checkpoint
    else:
        temporary_context = tempfile.TemporaryDirectory(prefix="mlx-h28-smoke-")
        training_checkpoint = Path(temporary_context.name) / "adapter"

    trainable_audit, training, checkpoint_qualification = _train_and_save_adapter(
        config=config,
        base=base,
        examples=examples,
        selected_indices=selection["selected_indices"],
        tokenizer=tokenizer,
        checkpoint=training_checkpoint,
        smoke=args.smoke,
    )
    audit, sample_text, evaluation_record, reload_qualification = _evaluate_adapter(
        config=config,
        base=base,
        checkpoint=training_checkpoint,
        task_manager=task_manager,
        target_accuracy_pct=canonical_target,
        smoke=args.smoke,
    )

    if official:
        training_checkpoint.rename(checkpoint)
        checkpoint_qualification["path"] = config["run"]["checkpoint"]
        reload_qualification["checkpoint_path"] = config["run"]["checkpoint"]
    elif temporary_context is not None:
        temporary_context.cleanup()

    report = {
        "run_id": config["run"]["id"] if official else "smoke_h28",
        "hypothesis": config["run"]["hypothesis"],
        "classification": (
            config["classification"] if official else "runtime_smoke_not_an_experiment"
        ),
        "validation_eligible": official,
        "git_commit": _git_commit(),
        "protocol": config,
        "preflight": preflight,
        "runtime_qualification": runtime_qualification,
        "harness_qualification": harness_qualification,
        "model_qualification": model_qualification,
        "train_dataset_qualification": train_dataset_qualification,
        "validation_dataset_qualification": validation_dataset_qualification,
        "task_qualification": task_qualification,
        "tokenization_qualification": tokenization_qualification,
        "selection_qualification": selection_qualification,
        "trainable_parameter_audit": trainable_audit,
        "training": training,
        "checkpoint": checkpoint_qualification,
        "reloaded_adapter_qualification": reload_qualification,
        "evaluation": {
            "audit": audit,
            "sample_log": {
                **evaluation_record["sample_log"],
                "path": config["run"]["samples_output"] if official else None,
            },
            "harness_results": evaluation_record["results"],
        },
        "runtime": {
            "wall_time_seconds": time.perf_counter() - started,
            "torch_version": str(torch.__version__),
            "transformers_version": transformers.__version__,
            "datasets_version": datasets.__version__,
            "peft_version": peft.__version__,
        },
        "summary": {
            "accuracy_pct": audit["accuracy_pct"],
            "correct_count": audit["correct_count"],
            "sample_count": audit["sample_count"],
            "paper_target_accuracy_pct": canonical_target,
            "relative_error": audit["relative_error"],
            "pass": audit["pass"] if official else None,
            "validation_eligible": official,
        },
    }
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    if official:
        output.parent.mkdir(parents=True, exist_ok=True)
        samples_output.parent.mkdir(parents=True, exist_ok=True)
        samples_output.write_text(sample_text, encoding="utf-8")
        output.write_text(
            json.dumps(
                report,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                default=handle_non_serializable,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0 if args.smoke or audit["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
