"""Capture complete local dense BERT QA references; never certify MLX execution."""

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import sys

import torch
import transformers
from safetensors.torch import load_file
from transformers import BertConfig, BertForQuestionAnswering, BertTokenizerFast

from mlxsim.model_execution_inventory import ModelExecutionInventory, require_model_performance_ready
from mlxsim.model_tensor_compiler import validate_event

ROOT = Path(__file__).resolve().parents[1]
SIGNATURE = {
    "model_type": "bert", "architectures": ["BertForQuestionAnswering"],
    "num_hidden_layers": 12, "hidden_size": 768, "intermediate_size": 3072,
    "num_attention_heads": 12, "vocab_size": 30522, "max_position_embeddings": 512,
    "type_vocab_size": 2, "hidden_act": "gelu",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_signature(config):
    require(all(config.get(key) == value for key, value in SIGNATURE.items()),
            "requires the complete dense 12-layer BERT-base QA architecture")


def f32_bits_equal(left, right):
    return (left.dtype == right.dtype == torch.float32 and left.shape == right.shape
            and torch.equal(left.contiguous().view(torch.int32), right.contiguous().view(torch.int32)))


def canonical_state(state, expected):
    """Only bijective LayerNorm gamma/beta names change, never tensor values."""
    values, mapping = {}, {}
    for source, tensor in state.items():
        target = re.sub(r"(?<=LayerNorm\.)gamma$", "weight", source)
        target = re.sub(r"(?<=LayerNorm\.)beta$", "bias", target)
        require(target not in values, "checkpoint name migration collision")
        require(target in expected, f"unexpected checkpoint tensor: {source}")
        require(isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float32,
                f"checkpoint must retain F32 precision: {source}")
        require(tensor.shape == expected[target].shape, f"checkpoint shape mismatch: {source}")
        require(expected[target].dtype == torch.float32, "model constructor changed checkpoint precision")
        require(torch.isfinite(tensor).all().item(), f"nonfinite checkpoint tensor: {source}")
        values[target], mapping[target] = tensor, source
    require(set(values) == set(expected), "checkpoint does not bind every model state tensor")
    return values, mapping


def validate_cases(cases):
    require(isinstance(cases, list) and len(cases) >= 2, "at least two QA input cases required")
    names = set()
    for case in cases:
        require(isinstance(case, dict) and set(case) == {"name", "question", "context", "pad_to"},
                "QA case requires name/question/context/pad_to only")
        require(all(isinstance(case[k], str) and case[k].strip() for k in ("name", "question", "context")),
                "QA names/question/context must be nonempty strings")
        require(case["name"] not in names, "duplicate QA case name")
        require(type(case["pad_to"]) is int and 0 <= case["pad_to"] <= 512,
                "QA pad_to must be an integer in [0,512]")
        names.add(case["name"])
    return cases


def choose_span(start, end, context_mask, max_tokens=30):
    """Reference postprocessing only; first lexicographic span wins score ties."""
    require(len(start) == len(end) == len(context_mask) and len(start) > 0,
            "QA span shape mismatch")
    require(type(max_tokens) is int and max_tokens > 0, "invalid maximum answer length")
    require(all(math.isfinite(v) for v in (*start, *end)),
            "nonfinite QA logits")
    best = None
    for first in range(len(start)):
        if not context_mask[first]:
            continue
        for last in range(first, min(len(start), first + max_tokens)):
            if not context_mask[last]:
                break
            score = start[first] + end[last]
            if best is None or score > best[0]:
                best = (score, first, last)
    require(best is not None, "QA input has no context tokens")
    return {"start": best[1], "end": best[2], "score_f64": best[0]}


def layer_coverage(trace, forward):
    layers = [e["layer_idx"] for e in trace.boundaries if e["event"] == "enter"
              and e["forward_id"] == forward
              and re.fullmatch(r"bert\.encoder\.layer\.\d+", e["module_path"])]
    require(sorted(layers) == list(range(12)), "trace must execute every full BERT layer exactly once")
    return layers


def run(args):
    model_path, output = args.model.resolve(), args.output.resolve()
    require(not output.exists(), "use a fresh reference directory")
    require(args.threads > 0, "reference threads must be positive")
    input_files = [model_path / name for name in
                   ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")]
    input_files.append(args.cases.resolve())
    inputs = {str(p): {"bytes": p.stat().st_size, "sha256": sha(p)} for p in input_files}
    cases = validate_cases(json.loads(args.cases.read_text()))
    config = json.loads((model_path / "config.json").read_text())
    validate_signature(config)
    source_paths = [Path(__file__).resolve(), ROOT / "src/mlxsim/model_execution_inventory.py",
                    ROOT / "src/mlxsim/model_tensor_compiler.py"]
    sources = {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    output.mkdir(parents=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False
    model_config = BertConfig.from_dict(config)
    model_config._attn_implementation = "eager"
    model = BertForQuestionAnswering(model_config)
    state, name_map = canonical_state(load_file(str(model_path / "model.safetensors")), model.state_dict())
    require(len(state) == 199 and sum(t.numel() for t in state.values()) == 108893186,
            "full baseline checkpoint tensor/parameter count changed")
    loaded = model.load_state_dict(state, strict=True)
    require(not loaded.missing_keys and not loaded.unexpected_keys, "non-strict checkpoint load")
    require(all(f32_bits_equal(model.state_dict()[key], value) for key, value in state.items()),
            "loaded parameter bytes differ from the original F32 checkpoint")
    del state
    model.eval().to(args.device)
    tokenizer = BertTokenizerFast.from_pretrained(model_path, local_files_only=True)
    framework_sources = {str(Path(inspect.getfile(type(m))).resolve()) for m in model.modules()}
    framework_sources.add(str(Path(inspect.getfile(type(tokenizer))).resolve()))
    framework = {p: sha(p) for p in sorted(framework_sources)}
    trace = ModelExecutionInventory(model, "bert-qa-baseline-dense-reference", capture_bindings=True)
    checks = []
    for forward, case in enumerate(cases):
        encoded = tokenizer(case["question"], case["context"], truncation=False,
                            padding="max_length" if case["pad_to"] else False,
                            max_length=case["pad_to"] or None, return_tensors="pt",
                            return_offsets_mapping=True)
        offsets = encoded.pop("offset_mapping")[0].tolist()
        context_mask = [seq == 1 for seq in encoded.sequence_ids(0)]
        require(encoded["input_ids"].shape[1] <= 512, "input exceeds architecture position capacity")
        if case["pad_to"]:
            require(encoded["input_ids"].shape[1] == case["pad_to"], "pad_to cannot truncate the real input")
        batch = {k: v.to(args.device) for k, v in encoded.items()}
        for key, value in batch.items():
            trace.bind_input(f"case{forward}:{key}", value)
        with torch.inference_mode():
            plain = model(**batch)
        print(f"BERT_QA_REFERENCE case={case['name']} tokens={len(context_mask)} full_layers=12", flush=True)
        with trace.attach(), torch.inference_mode(), trace.step(forward, "qa_forward"):
            actual = model(**batch)
        layer_coverage(trace, forward)
        values = {}
        for name in ("start_logits", "end_logits"):
            tensor = getattr(actual, name)
            require(tensor.dtype == torch.float32 and tensor.shape == batch["input_ids"].shape,
                    "QA output shape or precision changed")
            require(torch.isfinite(tensor).all().item() and f32_bits_equal(tensor, getattr(plain, name)),
                    "instrumentation changed QA logits or produced nonfinite values")
            file = output / f"case-{forward}-{name}.f32.bin"
            file.write_bytes(tensor.cpu().contiguous().numpy().tobytes())
            values[name] = {**trace.tensor(tensor), "file": str(file), "sha256": sha(file),
                            "bytes": file.stat().st_size, "instrumentation_bitwise_equal": True}
        first, last = actual.start_logits[0].tolist(), actual.end_logits[0].tolist()
        span = choose_span(first, last, context_mask)
        span["text"] = case["context"][offsets[span["start"]][0]:offsets[span["end"]][1]]
        plain_span = choose_span(plain.start_logits[0].tolist(), plain.end_logits[0].tolist(), context_mask)
        require(all(span[k] == v for k, v in plain_span.items()), "QA span selection changed under tracing")
        checks.append({"forward_id": forward, **case, "input_tensors": {k: trace.tensor(v) for k, v in batch.items()},
                       "input_values": {k: v.cpu().tolist() for k, v in batch.items()},
                       "context_mask": context_mask, "offsets": offsets, "outputs": values,
                       "layers_executed": 12, "span": span,
                       "span_selection": "reference_f64_start_plus_end_context_only_max30_first_lexicographic_tie",
                       "postprocessing_compiled": False, "dataset_accuracy_evaluated": False})
    report = trace.report()
    parameter_names = set(dict(model.named_parameters()))
    for forward in range(len(cases)):
        identifiers = set()
        def collect(value):
            if isinstance(value, dict):
                if "tensor_id" in value:
                    identifiers.add(value["tensor_id"])
                else:
                    for child in value.values():
                        collect(child)
            elif isinstance(value, list):
                for child in value:
                    collect(child)
        events = [e for e in report["operations"] if e["forward_id"] == forward]
        for event in events:
            collect([event["inputs"], event["kwargs"]])
        used = {name for identifier in identifiers for name in report["tensors"][identifier]["parameter_or_buffer_names"]}
        require(parameter_names <= used, "forward did not consume every checkpoint parameter")
        checks[forward]["parameter_tensors_consumed"] = len(parameter_names)
        checks[forward]["operator_calls"] = len(events)
    review = []
    for event in report["operations"]:
        row = {"source_operator_id": event["operator_id"], "operator": event["operator"]}
        try:
            row["candidate_native_kind"] = validate_event(event)
            row["status"] = "name_attributes_accepted_only_not_compiled_or_executed"
        except (ValueError, KeyError, TypeError) as error:
            row.update(status="rejected", reason=str(error))
        review.append(row)
    report.update(
        model_identity={"family": "BERT-base-QA", "variant": "local_dense_baseline_not_paper_hybrid",
                        "path": str(model_path), "config": config, "parameters": 108893186,
                        "parameter_tensors": 199, "all_parameters_loaded": True,
                        "checkpoint_name_by_model_parameter": name_map, "checkpoint_values_changed": False},
        reference_checks=checks, compiler_preflight=review, files=inputs, sources=sources,
        framework_sources=framework, instrumentation_equivalence_passed=True,
        author_hybrid_identity_verified=False, full_model_execution_verified=False,
        full_reference_model_executed=True, runtime={"python": sys.version, "torch": torch.__version__,
            "transformers": transformers.__version__, "device": args.device, "threads": args.threads,
            "dtype": "torch.float32", "attention": "eager", "deterministic_algorithms": True,
            "tf32": False, "custom_model_code_executed": False, "training_pickle_loaded": False})
    try:
        require_model_performance_ready(report)
    except RuntimeError as error:
        report["performance_gate_reason"] = str(error)
    else:
        raise RuntimeError("reference-only capture incorrectly passed the MLX performance gate")
    require(all(sha(p) == row["sha256"] for p, row in inputs.items()), "model/input files changed during capture")
    require(all(sha(ROOT / p) == h for p, h in sources.items()), "capture source changed during execution")
    require(all(sha(p) == h for p, h in framework.items()), "framework source changed during execution")
    (output / "inventory.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    summary = {k: report[k] for k in ("classification", "model_identity", "coverage", "reference_checks", "routes",
               "instrumentation_equivalence_passed", "mlx_model_execution", "inference_performance_eligible",
               "performance_gate_reason", "full_reference_model_executed", "full_model_execution_verified")}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"BERT_QA_REFERENCE_AND_INVENTORY_PASS (not MLX execution) {output / 'summary.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    output_existed = args.output.exists()
    try:
        run(args)
    except Exception as error:
        if not output_existed and args.output.exists() and not (args.output / "inventory.json").exists():
            failure = args.output / "failure.json"
            if not failure.exists():
                failure.write_text(json.dumps({"classification": "failed_reference_capture_not_mlx_execution",
                    "error": type(error).__name__ + ": " + str(error), "inference_performance_eligible": False}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
