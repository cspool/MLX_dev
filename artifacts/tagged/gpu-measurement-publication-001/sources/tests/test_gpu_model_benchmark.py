"""Benchmark scope/statistics tests; fixtures are not model performance results."""
import copy
import math
from types import SimpleNamespace

import pytest
import torch

from scripts.benchmark_mlx_gpu_models import distribution, graph
from scripts.audit_mlx_gpu_benchmark import audit_samples


@pytest.mark.parametrize("values", [[], [0], [-1], [math.inf], [math.nan], [True]])
def test_invalid_timing_samples_are_rejected(values):
    with pytest.raises(RuntimeError): distribution(values)


def timing_fixture():
    samples = [dict(iteration=i, begin_monotonic_ns=i * 10000000, end_monotonic_ns=i * 10000000 + 4000000,
                    cuda_stream_interval_ms=3., synchronized_graph_wall_ms=3.5, task_wall_ms=4., readback_postprocessing_wall_ms=.5,
                    forward_cuda_stream_ms=[1., 1., 1.]) for i in range(20)]
    report = dict(samples=samples, warmup=5, repetitions=20, device=dict(uuid="GPU-test"), statistics={})
    for key in ("cuda_stream_interval_ms", "synchronized_graph_wall_ms", "task_wall_ms"):
        report["statistics"][key] = distribution([r[key] for r in samples])
    report["statistics"]["per_forward_cuda_stream_ms"] = [distribution([1.] * 20)] * 3
    telemetry = dict(errors=[], locking=False, samples=[dict(begin_monotonic_ns=i * 10000000, end_monotonic_ns=i * 10000000 + 3000000,
        phase="measurement", values={"uuid": "GPU-test", "clocks.current.sm": "2500", "clocks.current.memory": "10000", "clocks_throttle_reasons.active": "0x0"}) for i in (0, 10)])
    return report, telemetry


def test_scoped_raw_measurements_and_telemetry_are_audited():
    report, telemetry = timing_fixture()
    assert audit_samples(report, telemetry)["overlapping_clock_samples"] == 2


@pytest.mark.parametrize("damage", ["median", "missing", "backwards", "scope", "nan", "steps", "gpu", "telemetry_time", "telemetry_gap", "locked"])
def test_corrupt_measurement_or_scope_cannot_pass(damage):
    report, telemetry = timing_fixture()
    if damage == "median": report["statistics"]["task_wall_ms"]["median_ms"] = 1
    if damage == "missing": report["samples"].pop()
    if damage == "backwards": report["samples"][1]["begin_monotonic_ns"] = 0
    if damage == "scope": report["samples"][0]["task_wall_ms"] = 10
    if damage == "nan": report["samples"][0]["cuda_stream_interval_ms"] = math.nan
    if damage == "steps": report["samples"][0]["forward_cuda_stream_ms"] = [10., 10., 10.]
    if damage == "gpu": telemetry["samples"][0]["values"]["uuid"] = "wrong"
    if damage == "telemetry_time": telemetry["samples"][0]["end_monotonic_ns"] = -1
    if damage == "telemetry_gap": telemetry["samples"] = []
    if damage == "locked": telemetry["locking"] = True
    with pytest.raises(RuntimeError): audit_samples(report, telemetry)


def test_generation_benchmark_uses_actual_tokens_and_fresh_cache():
    class Cache:
        def __init__(self, length): self.length = length
        def get_seq_length(self): return self.length
    class Model:
        def __init__(self): self.seen = []
        def __call__(self, input_ids, past_key_values, use_cache, logits_to_keep):
            self.seen.append((input_ids.tolist(), None if past_key_values is None else past_key_values.length))
            token = 1 if past_key_values is None else (int(input_ids[0, 0]) + 1) % 4
            logits = torch.full((1, 1, 4), -1.); logits[0, 0, token] = 1
            return SimpleNamespace(logits=logits, past_key_values=Cache(input_ids.shape[1] + (past_key_values.length if past_key_values else 0)))
    model = Model(); inventory = dict(model_identity=dict(family="Llama2-7B"))
    first = graph(model, torch.tensor([[0, 0]]), inventory, SimpleNamespace(eos_token_id=None))
    second = graph(model, torch.tensor([[0, 0]]), inventory, SimpleNamespace(eos_token_id=None))
    assert [r["token"] for r in first] == [1, 2, 3] == [r["token"] for r in second]
    assert model.seen == [([[0, 0]], None), ([[1]], 2), ([[2]], 3)] * 2
    stopped = graph(model, torch.tensor([[0, 0]]), inventory, SimpleNamespace(eos_token_id=1))
    assert len(stopped) == 1
