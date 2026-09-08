#!/usr/bin/env python3
"""Capture full-checkpoint Llama inference and operator gaps; not MLX execution."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import json
import os
import re
import sys
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from mlxsim.model_execution_inventory import (
    ModelExecutionInventory,
    require_model_performance_ready,
)
from mlxsim.model_tensor_observation import TensorObserver
from mlxsim.model_matrix_reference import MatrixReferenceMode
from mlxsim.model_numeric_reference import NumericReferenceMode

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run_reference(model, input_ids, new_tokens, eos_id, trace=None):
    records = []
    cache = None
    current = input_ids
    for step in range(new_tokens):
        phase = "prefill" if step == 0 else "decode"
        past = cache.get_seq_length() if cache is not None else 0
        scope = trace.step(step, phase) if trace else contextlib.nullcontext()
        with scope:
            output = model(
                input_ids=current, past_key_values=cache, use_cache=True, logits_to_keep=1
            )
            logits = output.logits[:, -1, :]
            if trace:
                trace.phase = "token_selection"
            selected = logits.argmax(dim=-1)
            current = selected.unsqueeze(1)
        # Observation copies occur outside dispatch tracing; no hidden debug
        # operations enter the compiler inventory and no per-op GPU sync occurs.
        token = int(selected[0])
        cache = output.past_key_values
        cache_states = []
        for index, layer in enumerate(cache.layers):
            state = {
                "layer_idx": index,
                "key_shape": list(layer.keys.shape),
                "value_shape": list(layer.values.shape),
            }
            if trace:
                state.update(key=trace.tensor(layer.keys), value=trace.tensor(layer.values))
                trace.boundaries.append(
                    {
                        "event": "cache_state",
                        "request_id": trace.request_id,
                        "forward_id": step,
                        "phase": phase,
                        **state,
                    }
                )
            cache_states.append(state)
        records.append(
            {
                "forward_id": step,
                "phase": phase,
                "q_len": input_ids.shape[1] if step == 0 else 1,
                "past_len": past,
                "kv_len": cache.get_seq_length(),
                "token_id": token,
                "logits": logits.detach().clone(),
                "cache_states": cache_states,
            }
        )
        if eos_id is not None and token == eos_id:
            break
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=ROOT / "third_party/models/llama2-7b-hf-msmirror"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--prompt", default="A short explanation of spatial computing is")
    parser.add_argument("--new-tokens", type=int, default=3)
    parser.add_argument("--capture-bindings", action="store_true")
    parser.add_argument("--observe-operators", type=Path, help="JSON array of output-only diagnostic operator IDs")
    parser.add_argument("--matrix-reference", choices=("framework", "kasc"), default="framework",
                        help="explicit matrix numeric-contract reference; does not replace historical framework checks")
    parser.add_argument("--float-reference", choices=("framework", "explicit"), default="framework",
                        help="explicit vector FP32 operation/reduction contract; atomic libm functions shared with C++")
    args = parser.parse_args()
    if args.float_reference == "explicit" and args.matrix_reference != "kasc":
        raise RuntimeError("explicit floating reference requires --matrix-reference kasc")
    model_path, output = args.model.resolve(), args.output.resolve()
    if output.exists():
        raise RuntimeError("use a new output directory to preserve prior reference evidence")
    if args.new_tokens < 2:
        raise RuntimeError("this prefill/decode capture requires at least two requested tokens")
    config = json.loads((model_path / "config.json").read_text())
    signature = {
        "model_type": "llama",
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "vocab_size": 32000,
    }
    if any(config.get(key) != value for key, value in signature.items()):
        raise RuntimeError(
            "this initial runner requires the full, unmodified Llama2-7B architecture"
        )
    output.mkdir(parents=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
    weight_index = json.loads((model_path / "model.safetensors.index.json").read_text())
    weights = sorted(set(weight_index["weight_map"].values()))
    files = [
        model_path / name
        for name in (
            *weights,
            "config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "tokenizer.model",
        )
    ]
    print("Fingerprinting the actual full-model weights and tokenizer", flush=True)
    fingerprints = {
        str(path): {"bytes": path.stat().st_size, "sha256": sha(path)} for path in files
    }
    input_stats = {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in files}
    sources = {
        str(path.relative_to(ROOT)): sha(path)
        for path in (
            Path(__file__).resolve(),
            ROOT / "src/mlxsim/model_execution_inventory.py",
            ROOT / "src/mlxsim/model_tensor_observation.py",
            ROOT / "src/mlxsim/model_matrix_reference.py",
            ROOT / "src/mlxsim/model_numeric_reference.py",
        )
    }
    print("Loading the full Llama2-7B checkpoint (no layer/width reduction)", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model, loading = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.float16,
        attn_implementation="eager",
        device_map={"": args.device},
        output_loading_info=True,
    )
    model.eval()
    for key in ("missing_keys", "mismatched_keys", "error_msgs"):
        if loading.get(key):
            raise RuntimeError(f"checkpoint load was incomplete: {key}={loading[key]}")
    parameter_names = set(dict(model.named_parameters()))
    checkpoint_names = set(weight_index["weight_map"])
    if parameter_names - checkpoint_names:
        raise RuntimeError("checkpoint does not bind every model parameter")
    ignored_buffers = sorted(checkpoint_names - parameter_names)
    allowed_buffer = re.compile(r"model\.layers\.\d+\.self_attn\.rotary_emb\.inv_freq")
    if any(
        not allowed_buffer.fullmatch(name)
        for name in ignored_buffers + list(loading.get("unexpected_keys", []))
    ):
        raise RuntimeError("unexpected checkpoint tensors beyond known nonpersistent RoPE buffers")
    if len(model.model.layers) != 32:
        raise RuntimeError("model depth changed during loading")
    input_ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"].to(args.device)
    print("Running independent uninstrumented prefill and greedy decode", flush=True)
    reference_type = NumericReferenceMode if args.float_reference == "explicit" else MatrixReferenceMode
    plain_matrix = reference_type() if args.matrix_reference == "kasc" else None
    observed_matrix = reference_type() if args.matrix_reference == "kasc" else None
    if plain_matrix:
        print("Explicit K-ascending FP32 matrix contract reference; NOT the default framework arithmetic", flush=True)
    with torch.inference_mode(), plain_matrix if plain_matrix is not None else contextlib.nullcontext():
        plain = run_reference(model, input_ids, args.new_tokens, tokenizer.eos_token_id)
    observer = TensorObserver(json.loads(args.observe_operators.read_text())) if args.observe_operators else None
    trace = ModelExecutionInventory(model, "llama2-public-dense-reference-0", args.capture_bindings, observer)
    if args.capture_bindings:
        trace.bind_input("input_ids", input_ids)
    print("Replaying the same complete model with metadata-only instrumentation", flush=True)
    with trace.attach(), torch.inference_mode(), observed_matrix if observed_matrix is not None else contextlib.nullcontext():
        observed = run_reference(model, input_ids, args.new_tokens, tokenizer.eos_token_id, trace)
    if len(plain) != len(observed) or len(observed) < 2:
        raise RuntimeError("forward-count mismatch or decode was not exercised")
    if plain_matrix is not None and (plain_matrix.calls != observed_matrix.calls or not sum(plain_matrix.calls.values())):
        raise RuntimeError("matrix numerical reference call coverage differs under instrumentation")
    if args.float_reference == "explicit" and (plain_matrix.float_calls != observed_matrix.float_calls or not sum(plain_matrix.float_calls.values())):
        raise RuntimeError("floating numeric reference coverage differs under instrumentation")
    checks = []
    for expected, actual in zip(plain, observed, strict=True):
        same = torch.equal(expected["logits"], actual["logits"])
        if not same or expected["token_id"] != actual["token_id"]:
            raise RuntimeError("instrumentation changed model logits or selected token")
        if actual["kv_len"] != actual["past_len"] + actual["q_len"] or any(
            state["key_shape"][-2] != actual["kv_len"]
            or state["value_shape"][-2] != actual["kv_len"]
            for state in actual["cache_states"]
        ):
            raise RuntimeError("cache growth does not match the actual prefill/decode step")
        layer_entries = [
            event
            for event in trace.boundaries
            if event["event"] == "enter"
            and event["forward_id"] == actual["forward_id"]
            and event["module_path"] in {f"model.layers.{layer}" for layer in range(32)}
        ]
        if len(layer_entries) != 32 or {event["layer_idx"] for event in layer_entries} != set(
            range(32)
        ):
            raise RuntimeError("trace did not execute every decoder layer")
        logits_path = output / f"logits-{actual['forward_id']}.f16.bin"
        logits_path.write_bytes(actual["logits"].cpu().numpy().tobytes())
        checks.append({key: value for key, value in actual.items() if key != "logits"})
        checks[-1].update(
            logits_bitwise_equal=True,
            logits_shape=list(actual["logits"].shape),
            logits_file=str(logits_path),
            logits_sha256=sha(logits_path),
            layer_entries=32,
        )
    report = trace.report()
    if observer is not None:
        observer.save(output / "diagnostics")
    report.update(
        model_identity={
            "family": "Llama2-7B",
            "variant": "public_dense_not_paper_hybrid",
            "path": str(model_path),
            "config": config,
            "files": fingerprints,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "parameter_tensors": len(parameter_names),
            "all_parameters_loaded": True,
            "ignored_nonpersistent_checkpoint_buffers": ignored_buffers,
            "model_source": inspect.getfile(type(model)),
            "model_source_sha256": sha(Path(inspect.getfile(type(model)))),
        },
        input={
            "prompt": args.prompt,
            "token_ids": input_ids.cpu().tolist(),
            "batch": 1,
            "requested_new_tokens": args.new_tokens,
            "sampling": "greedy_stop_at_eos",
        },
        runtime={
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "dtype": "float16",
            "attention": "eager",
            "device": args.device,
            "deterministic_algorithms": True,
            "tf32": False,
            "fp16_reduced_precision_reduction": False,
            "bf16_reduced_precision_reduction": False,
            "matrix_numeric_mode": "mlx-matrix-f32-kasc-v1" if plain_matrix else "framework_default",
            "matrix_reference_calls": plain_matrix.calls if plain_matrix else None,
            "float_numeric_mode": "mlx-vector-fp32-v1" if args.float_reference == "explicit" else "framework_default",
            "float_reference_calls": plain_matrix.float_calls if args.float_reference == "explicit" else None,
            "numeric_reference_provenance": plain_matrix.provenance() if args.float_reference == "explicit" else None,
        },
        reference_checks=checks,
        generated_text=tokenizer.decode([row["token_id"] for row in checks]),
        instrumentation_equivalence_passed=True,
        sources=sources,
    )
    try:
        require_model_performance_ready(report)
    except RuntimeError as error:
        report["performance_gate_reason"] = str(error)
    else:
        raise RuntimeError("reference-only evidence incorrectly passed the MLX performance gate")
    if any(sha(ROOT / path) != value for path, value in sources.items()):
        raise RuntimeError("capture source changed during reference execution")
    if any(
        (Path(path).stat().st_size, Path(path).stat().st_mtime_ns) != value
        for path, value in input_stats.items()
    ):
        raise RuntimeError("model/tokenizer files changed during reference execution")
    (output / "inventory.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    summary = {
        key: report[key]
        for key in (
            "classification",
            "coverage",
            "reference_checks",
            "generated_text",
            "instrumentation_equivalence_passed",
            "mlx_model_execution",
            "inference_performance_eligible",
            "performance_gate_reason",
        )
    }
    summary["routes"] = report["routes"]
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"MODEL_REFERENCE_AND_INVENTORY_PASS (not MLX execution) {output / 'summary.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
