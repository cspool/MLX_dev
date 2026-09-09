"""Same-checkpoint full GPU inference measurements; no layer extrapolation."""
import argparse
import contextlib
import ctypes
import csv
import inspect
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import threading
import time

import numpy as np
import torch
import transformers
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, BertConfig, BertForQuestionAnswering, BertTokenizerFast

from mlxsim.model_execution_inventory import ModelExecutionInventory
from scripts.capture_mlx_bert_qa_reference import canonical_state, f32_bits_equal, validate_signature
from scripts.mlx_system_attempt import record
from scripts.run_mlx_tensor_semantics import sha, ROOT

ATOL = 5e-3
RTOL = 5e-3
TELEMETRY_FIELDS = ("index", "uuid", "name", "pstate", "clocks.current.sm", "clocks.current.memory",
                    "utilization.gpu", "temperature.gpu", "power.draw", "power.limit", "clocks_throttle_reasons.active")


def require(condition, message):
    if not condition: raise RuntimeError(message)


def distribution(samples):
    require(bool(samples) and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in samples), "invalid measured durations")
    values = np.asarray(samples, dtype=np.float64)
    return dict(samples=len(samples), min_ms=float(values.min()), median_ms=float(np.median(values)),
                p10_ms=float(np.quantile(values, .1)), p90_ms=float(np.quantile(values, .9)),
                max_ms=float(values.max()), mean_ms=float(values.mean()), stddev_ms=float(values.std()))


def smi(device, fields):
    r = subprocess.run(["nvidia-smi", "-i", str(device), "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, check=True, timeout=10)
    rows = list(csv.reader(io.StringIO(r.stdout)))
    require(len(rows) == 1 and len(rows[0]) == len(fields), "GPU telemetry identity/field count differs")
    return dict(zip(fields, (v.strip() for v in rows[0]), strict=True))


class Telemetry:
    def __init__(self, device):
        self.device = device; self.stop_event = threading.Event(); self.samples = []; self.errors = []
        self.phase = "setup"; self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop_event.is_set():
            before = time.monotonic_ns(); phase = self.phase
            try:
                values = smi(self.device, TELEMETRY_FIELDS)
                self.samples.append(dict(begin_monotonic_ns=before, end_monotonic_ns=time.monotonic_ns(), phase=phase, values=values))
            except Exception as error: self.errors.append(str(error))
            self.stop_event.wait(.1)

    def __enter__(self): self.thread.start(); return self

    def __exit__(self, *unused): self.stop_event.set(); self.thread.join(timeout=12)


def fingerprints(inventory):
    bound = inventory.get("files", inventory["model_identity"].get("files", {}))
    require(bool(bound), "GPU benchmark requires fixed checkpoint/tokenizer inputs")
    for path, expected in bound.items():
        require(Path(path).stat().st_size == expected["bytes"] and sha(Path(path)) == expected["sha256"], "bound model input changed: " + path)
    return {path: expected["sha256"] for path, expected in bound.items()}


def checked_load(inventory, device):
    identity = inventory["model_identity"]; path = Path(identity["path"])
    if identity["family"] == "Llama2-7B":
        require(identity["variant"] == "public_dense_not_paper_hybrid" and identity["parameters"] == 6738415616, "expected complete dense Llama2")
        model, loading = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, trust_remote_code=False,
            dtype=torch.float16, attn_implementation="eager", device_map={"": device}, output_loading_info=True)
        require(all(not loading.get(k) for k in ("missing_keys", "mismatched_keys", "error_msgs")), "Llama2 checkpoint did not load completely")
        parameters = dict(model.named_parameters())
        index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
        omitted = set(index) - set(parameters)
        require(omitted == {f"model.layers.{i}.self_attn.rotary_emb.inv_freq" for i in range(32)}
                and not set(parameters) - set(index) and set(loading.get("unexpected_keys", [])) <= omitted, "unexpected Llama2 parameter omissions")
        require(len(parameters) == 291 and sum(v.numel() for v in parameters.values()) == 6738415616 and len(model.model.layers) == 32, "Llama2 model was reduced")
        # Copy each actually loaded GPU parameter back for byte equality, outside timing.
        for shard in sorted(set(index.values())):
            with safe_open(path / shard, framework="pt", device="cpu") as source:
                for name, value in parameters.items():
                    if index[name] != shard: continue
                    expected = source.get_tensor(name)
                    require(value.dtype == expected.dtype == torch.float16 and value.shape == expected.shape
                            and torch.equal(value.detach().cpu().view(torch.int16), expected.view(torch.int16)), "loaded Llama2 parameter bytes differ: " + name)
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        ids = tokenizer(inventory["input"]["prompt"], return_tensors="pt")["input_ids"]
        require(ids.tolist() == inventory["input"]["token_ids"] and list(ids.shape) == [1, 8]
                and inventory["input"]["requested_new_tokens"] == 3, "Llama2 benchmark input changed")
        inputs = ids.to(device)
        return model.eval(), inputs, tokenizer
    require(identity["family"] == "BERT-base-QA" and identity["variant"] == "local_dense_baseline_not_paper_hybrid", "unregistered GPU benchmark model")
    config = json.loads((path / "config.json").read_text()); validate_signature(config)
    cfg = BertConfig.from_dict(config); cfg._attn_implementation = "eager"
    model = BertForQuestionAnswering(cfg)
    state, mapping = canonical_state(load_file(str(path / "model.safetensors")), model.state_dict())
    require(len(state) == 199 and sum(t.numel() for t in state.values()) == 108893186 and mapping == identity["checkpoint_name_by_model_parameter"], "BERT complete parameter identity changed")
    model.load_state_dict(state, strict=True); model.eval().to(device)
    require(all(f32_bits_equal(model.state_dict()[k].cpu(), value) for k, value in state.items()), "actual GPU BERT weights differ from checkpoint")
    tokenizer = BertTokenizerFast.from_pretrained(path, local_files_only=True)
    batches = []
    for check in inventory["reference_checks"]:
        encoded = tokenizer(check["question"], check["context"], truncation=False, padding="max_length" if check["pad_to"] else False,
                            max_length=check["pad_to"] or None, return_tensors="pt", return_offsets_mapping=True)
        offsets = encoded.pop("offset_mapping")[0].tolist()
        require(offsets == check["offsets"] and [s == 1 for s in encoded.sequence_ids(0)] == check["context_mask"]
                and {k: v.tolist() for k, v in encoded.items()} == check["input_values"], "BERT tokenizer inputs differ")
        batches.append({k: v.to(device) for k, v in encoded.items()})
    require([b["input_ids"].shape[1] for b in batches] == [28, 64, 64], "BERT case coverage changed")
    return model, batches, tokenizer


def graph(model, inputs, inventory, tokenizer, events=None, trace=None):
    rows = []
    llama = inventory["model_identity"]["family"] == "Llama2-7B"
    cache = None; current = inputs
    for i in range(3):
        scope = trace.step(i, "prefill" if i == 0 else "decode") if trace and llama else trace.step(i, "qa_forward") if trace else contextlib.nullcontext()
        if events: events[i][0].record()
        with scope:
            if llama:
                output = model(input_ids=current, past_key_values=cache, use_cache=True, logits_to_keep=1)
                logits = output.logits[:, -1, :]
                if trace: trace.phase = "token_selection"
                selected = logits.argmax(-1); current = selected.unsqueeze(1)
                cache = output.past_key_values
            else:
                output = model(**inputs[i])
        if events: events[i][1].record()
        if llama:
            token = int(selected[0])  # Same greedy EOS decision, never a reference token.
            rows.append(dict(logits=logits, token=token, kv_length=cache.get_seq_length()))
            if tokenizer.eos_token_id is not None and token == tokenizer.eos_token_id: break
        else: rows.append(dict(start_logits=output.start_logits, end_logits=output.end_logits))
    return rows


class Span(ctypes.Structure):
    _fields_ = [("start", ctypes.c_size_t), ("end", ctypes.c_size_t), ("score", ctypes.c_double), ("candidates", ctypes.c_uint64)]


def span_function(out):
    source = ROOT / "simulator_ext/control_model/qa_span.c"; lib = out / "qa-postprocess.so"
    subprocess.run(["cc", "-O2", "-std=c11", "-ffp-contract=off", "-fPIC", "-shared", str(source), "-o", str(lib)], check=True, timeout=30)
    handle = ctypes.CDLL(str(lib)); fn = handle.mlx_qa_select_span
    fn.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(Span)]
    return fn, lib


def task_results(rows, inventory, choose):
    if inventory["model_identity"]["family"] == "Llama2-7B":
        return [dict(token=r["token"], kv_length=r["kv_length"]) for r in rows]
    results = []
    for row, check in zip(rows, inventory["reference_checks"], strict=True):
        a = row["start_logits"].detach().cpu().contiguous().numpy().reshape(-1)
        b = row["end_logits"].detach().cpu().contiguous().numpy().reshape(-1)
        mask = np.asarray(check["context_mask"], dtype=np.uint8); selected = Span()
        status = choose(a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), len(a), 30, ctypes.byref(selected))
        require(status == 0, "GPU actual QA logits failed C postprocessing")
        first = check["offsets"][selected.start][0]; last = check["offsets"][selected.end][1]
        results.append(dict(start=selected.start, end=selected.end, score_f64=selected.score, text=check["context"][first:last]))
    return results


def check_outputs(rows, inventory, tasks, out):
    require(len(rows) == len(inventory["reference_checks"]) == 3, "GPU did not execute the complete requested inference")
    llama = inventory["model_identity"]["family"] == "Llama2-7B"; comparisons = []
    for i, (row, check, task) in enumerate(zip(rows, inventory["reference_checks"], tasks, strict=True)):
        if llama: require(task["token"] == check["token_id"] and task["kv_length"] == check["kv_len"], "GPU task tokens or cache growth differ")
        else: require(all(task[k] == check["span"][k] for k in ("start", "end", "text")), "GPU QA answer differs")
        for role in (("logits",) if llama else ("start_logits", "end_logits")):
            spec = dict(file=check["logits_file"], sha256=check["logits_sha256"], shape=check["logits_shape"]) if llama else check["outputs"][role]
            require(sha(Path(spec["file"])) == spec["sha256"] and list(row[role].shape) == spec["shape"], "GPU reference bytes/shape changed")
            value = row[role].detach().cpu().contiguous().numpy()
            reference = np.fromfile(spec["file"], dtype="<f2" if llama else "<f4").reshape(value.shape)
            require(np.isfinite(value).all() and np.isfinite(reference).all(), "GPU outputs must be finite")
            error = np.abs(value.astype(np.float64) - reference.astype(np.float64))
            failure = error > ATOL + RTOL * np.abs(reference.astype(np.float64))
            require(not failure.any(), "GPU benchmark correctness exceeds bound tolerance")
            file = out / f"forward-{i}-{role}.{'f16' if llama else 'f32'}.bin"; file.write_bytes(value.tobytes())
            comparisons.append(dict(forward_id=i, role=role, file=str(file), shape=list(value.shape), dtype=str(value.dtype),
                actual_sha256=sha(file), reference_sha256=spec["sha256"], bitwise_equal=value.tobytes() == reference.tobytes(),
                max_abs_error=float(error.max()), failed_elements=int(failure.sum()), atol=ATOL, rtol=RTOL))
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=1); parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=50)
    args = parser.parse_args(); out = args.output.resolve()
    require(not out.exists() and args.warmup >= 5 and args.repetitions >= 20, "fresh directory and adequate warmup/repeats required")
    out.mkdir(parents=True); inventory = json.loads(args.inventory.read_text())
    os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"; os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(2); torch.manual_seed(42); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False; torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.cuda.set_device(args.device); device = f"cuda:{args.device}"
    identity = smi(args.device, ("index", "uuid", "name", "driver_version", "memory.total", "compute_cap", "power.limit"))
    require(identity["name"] == "NVIDIA GeForce RTX 4090" and identity["compute_cap"] == "8.9", "GPU benchmark hardware identity differs")
    environment = subprocess.run(["nvidia-smi", "-i", str(args.device), "-q", "-d", "PIDS"], capture_output=True, text=True, check=True, timeout=10)
    (out / "gpu-processes-before.log").write_text(environment.stdout)
    files = fingerprints(inventory); files[str(args.inventory.resolve())] = sha(args.inventory)
    sources = {str(p.resolve()): sha(p) for p in (Path(__file__), ROOT / "scripts/capture_mlx_bert_qa_reference.py",
        ROOT / "src/mlxsim/model_execution_inventory.py", ROOT / "simulator_ext/control_model/qa_span.c", ROOT / "simulator_ext/control_model/qa_span.h")}
    print("GPU_BENCHMARK loading complete " + inventory["model_identity"]["family"], flush=True)
    model, inputs, tokenizer = checked_load(inventory, device)
    for module in model.modules():
        path = Path(inspect.getfile(type(module))).resolve(); sources[str(path)] = sha(path)
    framework_files = (Path(inspect.getfile(transformers.masking_utils)), Path(inspect.getfile(torch.cuda.Event)), Path(inspect.getfile(torch)))
    for path in framework_files: sources[str(path.resolve())] = sha(path)
    weights = {name: (value._version, value.data_ptr(), tuple(value.shape), str(value.dtype)) for name, value in model.named_parameters()}
    choose, library = span_function(out); files[str(library)] = sha(library)
    policy = dict(classification="same_input_full_gpu_hardware_benchmark_not_mlx_simulation", model_family=inventory["model_identity"]["family"],
                  model_variant=inventory["model_identity"]["variant"], device=identity, warmup=args.warmup, repetitions=args.repetitions,
                  input_inventory_sha256=sha(args.inventory), loaded_parameters_bitwise_checked=True, atol=ATOL, rtol=RTOL,
                  attention="eager", tf32=False, fp16_reduced_precision_reduction=False, fresh_cache_each_repetition=True,
                  weights_and_inputs_resident=True, clock_locking=False, power_limit_changed=False,
                  scope="resident_model_with_separate_readback_postprocessing_not_checkpoint_or_tokenization_loading",
                  cuda_event_scope="current_stream_interval_including_possible_host_launch_gaps_not_kernel_duration_sum",
                  software=dict(torch=torch.__version__, transformers=transformers.__version__, cuda=torch.version.cuda))
    record(out / "protocol.json", policy); record(out / "provenance.json", dict(sources=sources, inputs=files))
    with torch.inference_mode():
        plain = graph(model, inputs, inventory, tokenizer)
        task = task_results(plain, inventory, choose)
        comparisons = check_outputs(plain, inventory, task, out)
        trace = ModelExecutionInventory(model, "gpu-benchmark-correctness-only")
        with trace.attach(): observed = graph(model, inputs, inventory, tokenizer, trace=trace)
        for a, b in zip(plain, observed, strict=True):
            for key, value in a.items():
                require(torch.equal(value, b[key]) if isinstance(value, torch.Tensor) else value == b[key], "GPU instrumentation changed outputs")
        for forward in range(3):
            pattern = r"model\.layers\.\d+" if inventory["model_identity"]["family"] == "Llama2-7B" else r"bert\.encoder\.layer\.\d+"
            layers = [b["module_path"] for b in trace.boundaries if b["event"] == "enter" and b["forward_id"] == forward and re.fullmatch(pattern, b["module_path"])]
            expected = 32 if inventory["model_identity"]["family"] == "Llama2-7B" else 12
            require(len(layers) == len(set(layers)) == expected, "GPU did not execute every complete layer")
        record(out / "correctness.json", dict(passed=True, tasks=task, comparison=comparisons, source_calls=len(trace.operations),
                                               instrumentation_bitwise_equal=True, all_layers_verified=True))
        record(out / "gpu-correctness-inventory.json", trace.report())
        baseline_outputs = plain
        del trace, observed
        per_step = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(3)]
        begin = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        samples = []
        with Telemetry(args.device) as telemetry:
            for index in range(args.warmup + args.repetitions):
                telemetry.phase = "warmup" if index < args.warmup else "measurement"
                torch.cuda.synchronize(args.device); wall_begin = time.monotonic_ns(); begin.record()
                outputs = graph(model, inputs, inventory, tokenizer, events=per_step)
                end.record(); end.synchronize(); wall_graph_end = time.monotonic_ns()
                actual_task = task_results(outputs, inventory, choose); wall_task_end = time.monotonic_ns()
                require(actual_task == task, "GPU task result changed during measurement")
                # Compare the actual repeated logits outside all reported timing
                # intervals; no result is injected into the next repetition.
                for actual, expected in zip(outputs, baseline_outputs, strict=True):
                    for role in (("logits",) if "logits" in actual else ("start_logits", "end_logits")):
                        require(torch.equal(actual[role], expected[role]), "GPU logits changed between repetitions")
                if index >= args.warmup:
                    samples.append(dict(iteration=index - args.warmup, begin_monotonic_ns=wall_begin, end_monotonic_ns=wall_task_end,
                        cuda_stream_interval_ms=begin.elapsed_time(end), synchronized_graph_wall_ms=(wall_graph_end-wall_begin)/1e6,
                        task_wall_ms=(wall_task_end-wall_begin)/1e6, readback_postprocessing_wall_ms=(wall_task_end-wall_graph_end)/1e6,
                        forward_cuda_stream_ms=[a.elapsed_time(b) for a,b in per_step]))
            telemetry.phase = "finished"
        (out / "final").mkdir()
        final_comparison = check_outputs(outputs, inventory, actual_task, out / "final")
    require(sum(s["phase"] == "measurement" for s in telemetry.samples) >= 2 and not telemetry.errors, "GPU frequency measurement telemetry unavailable")
    require(all(value["values"]["uuid"] == identity["uuid"] for value in telemetry.samples), "telemetry GPU differs")
    require(weights == {name: (v._version, v.data_ptr(), tuple(v.shape), str(v.dtype)) for name, v in model.named_parameters()}, "GPU model weights changed during inference")
    require(all(sha(Path(p)) == h for p,h in {**files, **sources}.items()), "GPU benchmark sources or inputs changed")
    # CPU graph execution threads were two; five long CPU simulator runs may coexist.
    record(out / "telemetry.json", dict(samples=telemetry.samples, errors=telemetry.errors, sampling_period_seconds=.1, locking=False))
    summary = {name: distribution([row[name] for row in samples]) for name in ("cuda_stream_interval_ms", "synchronized_graph_wall_ms", "task_wall_ms")}
    summary["per_forward_cuda_stream_ms"] = [distribution([row["forward_cuda_stream_ms"][i] for row in samples]) for i in range(3)]
    result = dict(**policy, correctness_passed=True, final_comparison=final_comparison, samples=samples, statistics=summary,
                  telemetry_samples=len(telemetry.samples), telemetry_sha256=sha(out / "telemetry.json"),
                  protocol_sha256=sha(out / "protocol.json"), correctness_sha256=sha(out / "correctness.json"), provenance_sha256=sha(out / "provenance.json"),
                  mlx_system_verified=False, gpu_simulator_result=False, cold_loading_measured=False, runtime_cpu_threads=2)
    record(out / "result.json", result)
    print(json.dumps(dict(result=str(out / "result.json"), statistics=summary)), flush=True)


if __name__ == "__main__": main()
