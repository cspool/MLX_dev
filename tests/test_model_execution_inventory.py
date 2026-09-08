import pytest
import torch

from mlxsim.model_execution_inventory import (
    ModelExecutionInventory,
    candidate_route,
    require_model_performance_ready,
    require_model_suite_performance_ready,
)


class SmallModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)])

    def forward(self, value):
        for layer in self.layers:
            value = torch.nn.functional.silu(layer(value))
        return value


def test_inventory_preserves_output_and_joins_model_layers_and_operators():
    torch.manual_seed(42)
    model = SmallModel().eval()
    value = torch.ones(2, 4)
    with torch.inference_mode():
        expected = model(value)
    trace = ModelExecutionInventory(model, "unit-0")
    with trace.attach(), torch.inference_mode(), trace.step(0, "prefill"):
        actual = model(value)
    assert torch.equal(actual, expected)
    report = trace.report()
    assert report["operations"]
    assert all(
        event["request_id"] == "unit-0" and event["forward_id"] == 0
        for event in report["operations"]
    )
    assert {
        event["layer_idx"] for event in report["operations"] if event["layer_idx"] is not None
    } == {0, 1}
    assert sum(event["event"] == "enter" for event in trace.boundaries) == sum(
        event["event"] == "exit" for event in trace.boundaries
    )
    assert any(meta["parameter_or_buffer_names"] for meta in report["tensors"].values())
    assert all(
        not module._forward_hooks and not module._forward_pre_hooks for module in model.modules()
    )
    assert report["coverage"]["missing_calls"] == len(report["operations"])
    with pytest.raises(RuntimeError, match="blocked"):
        require_model_performance_ready(report)


@pytest.mark.parametrize(
    "operator",
    [
        "aten.linear.default",
        "aten.softmax.int",
        "aten.rsqrt.default",
        "aten.embedding.default",
        "aten.where.ScalarOther",
        "aten.dropout.default",
    ],
)
def test_primitive_or_candidate_name_is_not_implemented_model_lowering(operator):
    route = candidate_route(operator)
    assert route["candidate_family"] != "unclassified"
    assert route["status"] == "missing_lowering"
    assert route["implemented_entry"] is None


def test_performance_gate_rejects_reference_only_and_missing_models():
    with pytest.raises(RuntimeError, match="blocked"):
        require_model_performance_ready(
            {
                "instrumentation_equivalence_passed": True,
                "coverage": {
                    "observed_calls": 1,
                    "compiled_and_executed_calls": 1,
                    "missing_calls": 0,
                },
            }
        )
    with pytest.raises(RuntimeError, match="required models"):
        require_model_suite_performance_ready({}, ["llama2", "internlm2"])
    with pytest.raises(RuntimeError, match="nonempty"):
        require_model_suite_performance_ready({}, [])
