#!/usr/bin/env python3
"""Train and evaluate the frozen H20 Llama2 chunk-FFT LoRA reconstruction."""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import peft
import pyarrow.parquet as pq
import torch
import transformers
import yaml
from peft import get_peft_model
from torch.nn import functional
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import get_cosine_schedule_with_warmup

from mlxsim.llama_fft import (
    audit_trainable_parameters,
    install_compressed_attention,
    make_lora_config,
)
from mlxsim.llama_perplexity import (
    audit_perplexity,
    complete_window_ranges,
    qualify_model_files,
    sha256_file,
    window_accounting,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/training/llama2_fft_lora_s075_v1.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
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


def _load_text(path: Path, separator: str) -> tuple[str, int]:
    rows = pq.read_table(path, columns=["text"]).column("text").to_pylist()
    return separator.join(str(row) for row in rows), len(rows)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _evaluate(
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    ranges: list[tuple[int, int]],
    *,
    device: torch.device,
    target: float,
    gate: float,
    label: str,
) -> dict[str, Any]:
    model.eval()
    total_nll = 0.0
    predicted_tokens = 0
    with torch.inference_mode():
        for index, (start, end) in enumerate(ranges, start=1):
            input_ids = token_ids[:, start:end].to(device)
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
                print(f"{label}: evaluated {index}/{len(ranges)} windows", file=sys.stderr, flush=True)
    return audit_perplexity(
        total_nll=total_nll,
        predicted_tokens=predicted_tokens,
        target=target,
        relative_error_gate=gate,
    )


def _adapter_hashes(checkpoint: Path) -> dict[str, Any]:
    hashes: dict[str, Any] = {}
    for path in sorted(checkpoint.iterdir()):
        if path.is_file():
            hashes[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return hashes


def main() -> int:
    args = _parse_args()
    config = _load_yaml(args.config)
    base = _load_yaml(PROJECT_ROOT / config["base_evaluation_config"])
    official = not args.smoke
    output = args.output or PROJECT_ROOT / config["run"]["output"]
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    checkpoint = PROJECT_ROOT / config["run"]["checkpoint"]
    if official and (output.exists() or checkpoint.exists()):
        raise SystemExit(f"refusing to overwrite H20 output/checkpoint: {output}, {checkpoint}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for H20")
    if transformers.__version__ != base["runtime"]["transformers_version"]:
        raise SystemExit("Transformers version does not match the H19 base protocol")
    if peft.__version__ != config["lora"]["peft_version"]:
        raise SystemExit(
            f"PEFT version mismatch: expected {config['lora']['peft_version']}, found {peft.__version__}"
        )

    started = time.perf_counter()
    seed = int(config["optimization"]["seed"])
    _set_seed(seed)
    model_qualification = qualify_model_files(base["model"])
    if not model_qualification["pass"]:
        raise RuntimeError("base model qualification failed")

    train_path = PROJECT_ROOT / config["training_data"]["path"]
    test_path = PROJECT_ROOT / base["dataset"]["path"]
    train_hash = sha256_file(train_path)
    test_hash = sha256_file(test_path)
    if train_hash != config["training_data"]["sha256"] or test_hash != base["dataset"]["sha256"]:
        raise RuntimeError("training or test dataset hash mismatch")
    train_text, train_rows = _load_text(train_path, config["training_data"]["row_separator"])
    test_text, test_rows = _load_text(test_path, base["dataset"]["row_separator"])
    if train_rows != int(config["training_data"]["expected_rows"]):
        raise RuntimeError(f"training row-count mismatch: {train_rows}")
    if test_rows != int(base["dataset"]["expected_rows"]):
        raise RuntimeError(f"test row-count mismatch: {test_rows}")

    model_path = PROJECT_ROOT / base["model"]["path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, local_files_only=True)
    train_ids = tokenizer(train_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    test_ids = tokenizer(test_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    if int(train_ids.shape[1]) != int(config["training_data"]["expected_token_count"]):
        raise RuntimeError(f"training token-count mismatch: {train_ids.shape[1]}")
    if int(test_ids.shape[1]) != int(base["dataset"]["expected_token_count"]):
        raise RuntimeError(f"test token-count mismatch: {test_ids.shape[1]}")

    sequence_length = int(config["training_data"]["sequence_length"])
    selected_windows = int(config["training_data"]["selected_complete_windows"])
    train_ranges = complete_window_ranges(int(train_ids.shape[1]), sequence_length)[:selected_windows]
    test_accounting = window_accounting(int(test_ids.shape[1]), sequence_length)
    test_ranges = complete_window_ranges(int(test_ids.shape[1]), sequence_length)
    if args.smoke:
        train_ranges = train_ranges[:2]
        test_ranges = test_ranges[:1]

    device = torch.device(config["optimization"]["device"])
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
    structured = config["structured_attention"]
    installed_layers = install_compressed_attention(
        model,
        layer_indices=list(structured["modified_layer_indices"]),
        chunk_length=int(structured["chunk_length"]),
        compression_ratio=float(structured["compression_ratio"]),
    )
    model = get_peft_model(model, make_lora_config(config["lora"]))
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    trainable_audit = audit_trainable_parameters(
        model,
        expected_layers=list(config["lora"]["layers_to_transform"]),
        maximum_fraction=float(config["lora"]["expected_trainable_fraction_max"]),
    )
    if not trainable_audit["pass"]:
        raise RuntimeError(f"LoRA trainable-parameter audit failed: {trainable_audit}")

    target_manifest = _load_yaml(PROJECT_ROOT / config["evaluation"]["target_source"])
    canonical_target = float(
        target_manifest["fig15_quality"]["llm_generation"]["llama2_wikitext2_1k"][
            "perplexity"
        ][1]
    )
    if canonical_target != float(config["evaluation"]["target_perplexity"]):
        raise RuntimeError("H20 target does not match the canonical target manifest")
    gate = float(config["evaluation"]["relative_error_gate"])
    pre_training = _evaluate(
        model,
        test_ids,
        test_ranges,
        device=device,
        target=canonical_target,
        gate=gate,
        label="pre-training" if official else "smoke-pre",
    )

    optimizer_cfg = config["optimization"]
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(optimizer_cfg["learning_rate"]),
        weight_decay=float(optimizer_cfg["weight_decay"]),
    )
    gradient_accumulation = (
        int(optimizer_cfg["gradient_accumulation_steps"]) if official else len(train_ranges)
    )
    optimizer_steps = math.ceil(len(train_ranges) / gradient_accumulation)
    if official and optimizer_steps != int(optimizer_cfg["optimizer_steps"]):
        raise RuntimeError(f"optimizer-step mismatch: expected 64, got {optimizer_steps}")
    warmup_steps = math.ceil(float(optimizer_cfg["warmup_ratio"]) * optimizer_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=optimizer_steps,
    )
    permutation = torch.randperm(len(train_ranges), generator=torch.Generator().manual_seed(seed))
    model.train()
    optimizer.zero_grad(set_to_none=True)
    training_losses: list[float] = []
    completed_optimizer_steps = 0
    for micro_step, permutation_index in enumerate(permutation.tolist(), start=1):
        start, end = train_ranges[permutation_index]
        input_ids = train_ids[:, start:end].to(device)
        logits = model(input_ids=input_ids, use_cache=False).logits[:, :-1, :].float()
        labels = input_ids[:, 1:]
        loss = functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="mean",
        )
        training_losses.append(float(loss.detach()))
        (loss / gradient_accumulation).backward()
        if micro_step % gradient_accumulation == 0 or micro_step == len(train_ranges):
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters, float(optimizer_cfg["max_grad_norm"])
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            completed_optimizer_steps += 1
        if micro_step % 16 == 0 or micro_step == len(train_ranges):
            print(
                f"trained {micro_step}/{len(train_ranges)} windows; "
                f"optimizer steps {completed_optimizer_steps}/{optimizer_steps}; "
                f"loss {training_losses[-1]:.4f}",
                file=sys.stderr,
                flush=True,
            )

    post_training = _evaluate(
        model,
        test_ids,
        test_ranges,
        device=device,
        target=canonical_target,
        gate=gate,
        label="post-training" if official else "smoke-post",
    )
    checkpoint_hashes: dict[str, Any] = {}
    if official:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint, safe_serialization=True)
        checkpoint_hashes = _adapter_hashes(checkpoint)

    report = {
        "run_id": config["run"]["id"] if official else "smoke_h20",
        "hypothesis": config["run"]["hypothesis"],
        "classification": (
            config["classification"] if official else "runtime_smoke_not_an_experiment"
        ),
        "validation_eligible": False,
        "full_mlx_bar_reproduced": False,
        "git_commit": _git_commit(),
        "protocol": config,
        "model_qualification": model_qualification,
        "datasets": {
            "train": {
                "path": config["training_data"]["path"],
                "sha256": train_hash,
                "rows": train_rows,
                "tokens": int(train_ids.shape[1]),
            },
            "test": {
                "path": base["dataset"]["path"],
                "sha256": test_hash,
                "rows": test_rows,
                "tokens": int(test_ids.shape[1]),
                "full_window_accounting": test_accounting,
            },
        },
        "installed_compressed_layers": installed_layers,
        "trainable_parameter_audit": trainable_audit,
        "training": {
            "executed_windows": len(train_ranges),
            "gradient_accumulation_steps": gradient_accumulation,
            "optimizer_steps": completed_optimizer_steps,
            "warmup_steps": warmup_steps,
            "initial_loss": training_losses[0],
            "final_loss": training_losses[-1],
            "mean_loss": sum(training_losses) / len(training_losses),
        },
        "pre_training_evaluation": pre_training,
        "post_training_evaluation": post_training,
        "checkpoint": {
            "path": str(checkpoint.relative_to(PROJECT_ROOT)) if official else None,
            "files": checkpoint_hashes,
        },
        "runtime": {
            "wall_time_seconds": time.perf_counter() - started,
            "device": str(device),
            "cuda_device": torch.cuda.get_device_name(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "peft_version": peft.__version__,
        },
        "summary": {
            "post_training_perplexity": post_training["perplexity"],
            "post_training_relative_error": post_training["relative_error"],
            "numerical_gate_pass": post_training["pass"],
            "full_mlx_bar_reproduced": False,
        },
    }
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    if official:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if args.smoke or post_training["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
