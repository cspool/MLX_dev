"""Pairwise WinoGrande training helpers for the H28 dense LoRA run."""

from __future__ import annotations

import hashlib
import json
import struct
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import get_peft_model_state_dict
from safetensors.torch import load_file
from torch.nn import functional

from mlxsim.llama_perplexity import sha256_file


@dataclass(frozen=True)
class EncodedChoice:
    token_ids: tuple[int, ...]
    context_tokens: int


@dataclass(frozen=True)
class EncodedExample:
    choices: tuple[EncodedChoice, EncodedChoice]
    target: int


def partial_scoring_strings(
    row: Mapping[str, Any], *, target_delimiter: str = " "
) -> tuple[tuple[str, str], tuple[str, str]]:
    sentence = str(row["sentence"])
    split = sentence.index("_")
    continuation = target_delimiter + sentence[split + 1 :].strip()
    prefix = sentence[:split]
    return (
        (prefix + str(row["option1"]), continuation),
        (prefix + str(row["option2"]), continuation),
    )


def encode_choice(tokenizer: Any, *, context: str, continuation: str) -> EncodedChoice:
    """Match lm-eval HFLM's causal context/continuation boundary rule."""

    trailing_spaces = len(context) - len(context.rstrip())
    if trailing_spaces:
        continuation = context[-trailing_spaces:] + continuation
        context = context[:-trailing_spaces]
    whole_ids = tuple(int(token) for token in tokenizer.encode(context + continuation))
    context_ids = tuple(int(token) for token in tokenizer.encode(context))
    continuation_ids = whole_ids[len(context_ids) :]
    if not context_ids or not continuation_ids:
        raise ValueError("context and continuation must each contain at least one token")
    return EncodedChoice(token_ids=whole_ids, context_tokens=len(context_ids))


def encode_example(
    row: Mapping[str, Any], tokenizer: Any, *, target_delimiter: str = " "
) -> EncodedExample:
    pairs = partial_scoring_strings(row, target_delimiter=target_delimiter)
    choices = tuple(
        encode_choice(tokenizer, context=context, continuation=continuation)
        for context, continuation in pairs
    )
    answer = str(row["answer"])
    if answer not in {"1", "2"}:
        raise ValueError(f"unexpected WinoGrande answer: {answer!r}")
    return EncodedExample(choices=choices, target=int(answer) - 1)  # type: ignore[arg-type]


def encode_examples(
    rows: Sequence[Mapping[str, Any]], tokenizer: Any, *, target_delimiter: str = " "
) -> list[EncodedExample]:
    return [
        encode_example(row, tokenizer, target_delimiter=target_delimiter) for row in rows
    ]


def tokenization_audit(examples: Sequence[EncodedExample]) -> dict[str, Any]:
    if not examples:
        raise ValueError("examples cannot be empty")
    digest = hashlib.sha256()
    lengths: list[int] = []
    context_lengths: list[int] = []
    continuation_lengths: list[int] = []
    for example in examples:
        for choice in example.choices:
            length = len(choice.token_ids)
            digest.update(struct.pack(">H", length))
            digest.update(struct.pack(">H", choice.context_tokens))
            for token in choice.token_ids:
                digest.update(struct.pack(">I", token))
            lengths.append(length)
            context_lengths.append(choice.context_tokens)
            continuation_lengths.append(length - choice.context_tokens)
    return {
        "examples": len(examples),
        "requests": len(lengths),
        "min_sequence_tokens": min(lengths),
        "max_sequence_tokens": max(lengths),
        "max_context_tokens": max(context_lengths),
        "min_continuation_tokens": min(continuation_lengths),
        "max_continuation_tokens": max(continuation_lengths),
        "sha256": digest.hexdigest(),
        "first_request_ids": list(examples[0].choices[0].token_ids),
        "first_request_context_tokens": examples[0].choices[0].context_tokens,
        "last_request_ids": list(examples[-1].choices[-1].token_ids),
        "last_request_context_tokens": examples[-1].choices[-1].context_tokens,
    }


def shuffled_selection(total_rows: int, selected_rows: int, seed: int) -> dict[str, Any]:
    if not 0 < selected_rows <= total_rows:
        raise ValueError("selected_rows must be in (0, total_rows]")
    permutation = torch.randperm(
        total_rows, generator=torch.Generator().manual_seed(seed)
    ).tolist()
    selected = permutation[:selected_rows]
    digest = hashlib.sha256()
    for index in selected:
        digest.update(struct.pack(">I", index))
    return {
        "selected_indices": selected,
        "dropped_indices": permutation[selected_rows:],
        "selected_indices_sha256_uint32be": digest.hexdigest(),
    }


def collate_pairwise(
    examples: Sequence[EncodedExample], *, pad_token_id: int
) -> dict[str, torch.Tensor]:
    if not examples:
        raise ValueError("examples cannot be empty")
    choices = [choice for example in examples for choice in example.choices]
    maximum_length = max(len(choice.token_ids) for choice in choices)
    input_ids = torch.full(
        (len(choices), maximum_length), int(pad_token_id), dtype=torch.long
    )
    attention_mask = torch.zeros_like(input_ids)
    continuation_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for row, choice in enumerate(choices):
        length = len(choice.token_ids)
        input_ids[row, :length] = torch.tensor(choice.token_ids, dtype=torch.long)
        attention_mask[row, :length] = 1
        continuation_mask[row, choice.context_tokens : length] = True
    targets = torch.tensor([example.target for example in examples], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "continuation_mask": continuation_mask,
        "targets": targets,
    }


def continuation_choice_scores(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    continuation_mask: torch.Tensor,
) -> torch.Tensor:
    """Return summed continuation log likelihood for two choices per example."""

    if logits.ndim != 3 or input_ids.shape != logits.shape[:2]:
        raise ValueError("logits/input shape mismatch")
    if continuation_mask.shape != input_ids.shape:
        raise ValueError("continuation mask shape mismatch")
    if logits.shape[0] % 2:
        raise ValueError("pairwise batch must contain an even number of requests")
    shifted_logits = logits[:, :-1, :].float()
    shifted_labels = input_ids[:, 1:]
    shifted_mask = continuation_mask[:, 1:]
    if not bool(shifted_mask.any(dim=1).all()):
        raise ValueError("every request must score at least one continuation token")
    token_nll = functional.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1),
        reduction="none",
    ).reshape(shifted_labels.shape)
    request_scores = -(token_nll * shifted_mask).sum(dim=1)
    return request_scores.reshape(-1, 2)


def audit_lora_training_model(
    model: torch.nn.Module,
    *,
    expected_layers: Sequence[int],
    expected_modules: Sequence[str],
    expected_trainable_parameters: int,
    expected_trainable_tensors: int,
    expected_total_parameters: int,
    maximum_trainable_fraction: float,
) -> dict[str, Any]:
    named = list(model.named_parameters())
    trainable = [(name, parameter) for name, parameter in named if parameter.requires_grad]
    trainable_names = [name for name, _ in trainable]
    trainable_parameters = sum(parameter.numel() for _, parameter in trainable)
    total_parameters = sum(parameter.numel() for _, parameter in named)
    missing: list[str] = []
    for layer in expected_layers:
        for module in expected_modules:
            for factor in ("A", "B"):
                if not any(
                    f".layers.{layer}." in name
                    and f".{module}.lora_{factor}." in name
                    for name in trainable_names
                ):
                    missing.append(f"layer={layer},module={module},factor={factor}")
    unexpected = [
        name for name in trainable_names if ".lora_A." not in name and ".lora_B." not in name
    ]
    active = list(getattr(model, "active_adapters", []))
    fraction = trainable_parameters / total_parameters
    checks = {
        "trainable_parameters": trainable_parameters == expected_trainable_parameters,
        "trainable_tensors": len(trainable) == expected_trainable_tensors,
        "total_parameters": total_parameters == expected_total_parameters,
        "maximum_trainable_fraction": fraction <= maximum_trainable_fraction,
        "only_lora_trainable": not unexpected,
        "all_layer_module_factors_present": not missing,
        "default_adapter_active": active == ["default"],
    }
    return {
        "trainable_parameters": trainable_parameters,
        "trainable_tensors": len(trainable),
        "total_parameters": total_parameters,
        "trainable_fraction": fraction,
        "active_adapters": active,
        "unexpected_trainable_parameters": unexpected,
        "missing_layer_module_factors": missing,
        "checks": checks,
        "pass": all(checks.values()),
    }


def qualify_adapter_checkpoint(
    checkpoint: Path,
    *,
    expected_layers: Sequence[int],
    expected_modules: Sequence[str],
    expected_rank: int,
    expected_alpha: int,
    expected_dropout: float,
    expected_parameters: int,
    expected_tensors: int,
) -> dict[str, Any]:
    config_path = checkpoint / "adapter_config.json"
    weights_path = checkpoint / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        return {
            "path": str(checkpoint),
            "required_files_present": False,
            "pass": False,
        }
    config = json.loads(config_path.read_text(encoding="utf-8"))
    state = load_file(weights_path, device="cpu")
    missing: list[str] = []
    for layer in expected_layers:
        for module in expected_modules:
            for factor in ("A", "B"):
                if not any(
                    f".layers.{layer}." in name
                    and f".{module}.lora_{factor}.weight" in name
                    for name in state
                ):
                    missing.append(f"layer={layer},module={module},factor={factor}")
    parameters = sum(tensor.numel() for tensor in state.values())
    dtypes = dict(Counter(str(tensor.dtype) for tensor in state.values()))
    checks = {
        "rank": int(config.get("r", -1)) == expected_rank,
        "alpha": int(config.get("lora_alpha", -1)) == expected_alpha,
        "dropout": float(config.get("lora_dropout", -1.0)) == expected_dropout,
        "bias": config.get("bias") == "none",
        "task_type": config.get("task_type") == "CAUSAL_LM",
        "target_modules": set(config.get("target_modules", [])) == set(expected_modules),
        "layers_to_transform": config.get("layers_to_transform") == list(expected_layers),
        "tensor_count": len(state) == expected_tensors,
        "parameter_count": parameters == expected_parameters,
        "all_layer_module_factors_present": not missing,
    }
    file_hashes = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(checkpoint.iterdir())
        if path.is_file()
    }
    return {
        "path": str(checkpoint),
        "required_files_present": True,
        "adapter_config": config,
        "tensor_count": len(state),
        "parameter_count": parameters,
        "tensor_dtypes": dtypes,
        "missing_layer_module_factors": missing,
        "files": file_hashes,
        "checks": checks,
        "pass": all(checks.values()),
    }


def qualify_reloaded_adapter(
    model: torch.nn.Module,
    checkpoint: Path,
    *,
    expected_layers: Sequence[int],
    expected_modules: Sequence[str],
    expected_rank: int,
    expected_alpha: int,
    expected_dropout: float,
    expected_parameters: int,
    expected_tensors: int,
) -> dict[str, Any]:
    """Prove that PEFT reloaded every saved LoRA tensor and config field."""

    peft_configs = getattr(model, "peft_config", {})
    adapter_config = peft_configs.get("default")
    active = list(getattr(model, "active_adapters", []))
    loaded_state = get_peft_model_state_dict(model, adapter_name="default")
    disk_state = load_file(checkpoint / "adapter_model.safetensors", device="cpu")
    disk_keys = set(disk_state)
    loaded_keys = set(loaded_state)
    mismatched_values = [
        name
        for name in sorted(disk_keys & loaded_keys)
        if not torch.equal(disk_state[name], loaded_state[name].detach().cpu())
    ]
    missing: list[str] = []
    for layer in expected_layers:
        for module in expected_modules:
            for factor in ("A", "B"):
                if not any(
                    f".layers.{layer}." in name
                    and f".{module}.lora_{factor}.weight" in name
                    for name in loaded_state
                ):
                    missing.append(f"layer={layer},module={module},factor={factor}")

    parameters = sum(tensor.numel() for tensor in loaded_state.values())
    raw_task_type = getattr(adapter_config, "task_type", None)
    task_type = getattr(raw_task_type, "value", raw_task_type)
    checks = {
        "default_config_present": adapter_config is not None,
        "default_adapter_active": active == ["default"],
        "inference_mode": bool(getattr(adapter_config, "inference_mode", False)),
        "rank": getattr(adapter_config, "r", None) == expected_rank,
        "alpha": getattr(adapter_config, "lora_alpha", None) == expected_alpha,
        "dropout": getattr(adapter_config, "lora_dropout", None) == expected_dropout,
        "bias": getattr(adapter_config, "bias", None) == "none",
        "task_type": task_type == "CAUSAL_LM",
        "target_modules": set(getattr(adapter_config, "target_modules", []))
        == set(expected_modules),
        "layers_to_transform": getattr(adapter_config, "layers_to_transform", None)
        == list(expected_layers),
        "tensor_count": len(loaded_state) == expected_tensors,
        "parameter_count": parameters == expected_parameters,
        "all_layer_module_factors_present": not missing,
        "checkpoint_keys_exact": disk_keys == loaded_keys,
        "checkpoint_values_exact": not mismatched_values,
    }
    return {
        "checkpoint_path": str(checkpoint),
        "active_adapters": active,
        "tensor_count": len(loaded_state),
        "parameter_count": parameters,
        "missing_layer_module_factors": missing,
        "missing_checkpoint_keys": sorted(disk_keys - loaded_keys),
        "unexpected_loaded_keys": sorted(loaded_keys - disk_keys),
        "mismatched_checkpoint_values": mismatched_values,
        "checks": checks,
        "pass": all(checks.values()),
    }
