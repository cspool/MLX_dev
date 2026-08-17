#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import bitsandbytes as bnb
import google.protobuf
import modelscope
import torch
import transformers
from accelerate import __version__ as accelerate_version
from datasets import __version__ as datasets_version
from peft import LoraConfig, TaskType, get_peft_model
from peft import __version__ as peft_version
from torch import nn


class _TinyCausalModel(nn.Module):
    """Minimal PEFT-compatible module for an offline LoRA injection check."""

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(16, 16, bias=False)
        self.v_proj = nn.Linear(16, 16, bias=False)

    def forward(self, input_ids: torch.Tensor, **_: object) -> torch.Tensor:
        return self.q_proj(input_ids) + self.v_proj(input_ids)

    def prepare_inputs_for_generation(self, *args: object, **kwargs: object) -> dict[str, object]:
        return kwargs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="optional path for the machine-readable JSON report",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise SystemExit("expected two visible CUDA devices")

    check_device = torch.device("cuda:1")
    model = get_peft_model(
        _TinyCausalModel(),
        LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        ),
    ).to(device=check_device, dtype=torch.bfloat16)
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    lora_output = model(input_ids=torch.randn(2, 16, device=check_device, dtype=torch.bfloat16))
    lora_output.square().mean().backward()
    adapter_gradients = sum(
        parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad
    )

    nf4_layer = bnb.nn.Linear4bit(
        64,
        64,
        bias=False,
        compute_dtype=torch.bfloat16,
        quant_type="nf4",
    ).to(check_device)
    with torch.no_grad():
        nf4_output = nf4_layer(torch.randn(2, 64, device=check_device, dtype=torch.bfloat16))

    devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    report = {
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "transformers": transformers.__version__,
        "peft": peft_version,
        "protobuf": google.protobuf.__version__,
        "accelerate": accelerate_version,
        "datasets": datasets_version,
        "modelscope": modelscope.__version__,
        "bitsandbytes": bnb.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": devices,
        "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        "check_device": str(check_device),
        "lora_adapter_tensors_with_gradients": adapter_gradients,
        "lora_forward_dtype": str(lora_output.dtype),
        "lora_forward_finite": bool(torch.isfinite(lora_output).all()),
        "lora_trainable_parameters": trainable,
        "lora_total_parameters": total,
        "nf4_forward_dtype": str(nf4_output.dtype),
        "nf4_forward_finite": bool(torch.isfinite(nf4_output).all()),
        "nf4_quant_type": nf4_layer.weight.quant_type,
    }
    report_json = json.dumps(report, indent=2, sort_keys=True)
    print(report_json)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{report_json}\n", encoding="utf-8")
    if trainable <= 0 or trainable >= total:
        raise SystemExit("LoRA injection did not isolate trainable adapter parameters")
    if adapter_gradients == 0 or not bool(torch.isfinite(lora_output).all()):
        raise SystemExit("LoRA forward/backward check failed")
    if nf4_layer.weight.quant_type != "nf4" or not bool(torch.isfinite(nf4_output).all()):
        raise SystemExit("bitsandbytes NF4 forward check failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
