"""Real-run operator inventory; never claims model compilation or MLX execution."""

from __future__ import annotations

import contextlib
import math
import re
import weakref
from collections import Counter

import torch
from torch.utils._python_dispatch import TorchDispatchMode


def candidate_route(operator):
    """Planning hints, deliberately separate from an implemented lowering."""
    name = operator.split(".")[1] if operator.startswith("aten.") else operator
    if name in {"mm", "bmm", "matmul", "addmm", "linear"}:
        family = "dense_gemm_tiled"
        gap = "tensor tiling, weight streaming and accumulation precision are not implemented in tagged lowering"
    elif (
        name in {"_softmax", "softmax", "_safe_softmax", "scaled_dot_product_attention"}
        or "attention" in name
    ):
        family = "attention_or_softmax_composite"
        gap = "full-shape attention/reduction, mask and FP32 normalization lowering is missing"
    elif name in {"embedding", "gather", "index_select", "index"}:
        family = "indexed_memory_access"
        gap = "tensor/index address generation and global-memory binding are missing"
    elif "fft" in name:
        family = "fft_cmp_or_fft_primitive"
        gap = "actual complex dtype, normalization, layout and full tensor FFT lowering must be implemented"
    elif name in {
        "mean",
        "sum",
        "amax",
        "max",
        "argmax",
        "native_layer_norm",
        "layer_norm",
        "rsqrt",
        "sqrt",
    }:
        family = "normalization_or_reduction"
        gap = "full tensor reduction and required precision/reciprocal-root modes are missing"
    elif name in {
        "view",
        "_unsafe_view",
        "reshape",
        "transpose",
        "t",
        "permute",
        "slice",
        "select",
        "unsqueeze",
        "squeeze",
        "expand",
        "as_strided",
        "detach",
        "detach_",
    }:
        family = "layout_alias_or_address_lowering"
        gap = "no tensor-view/alias lowering; metadata views cannot be assumed to make required data movement free"
    elif name in {"cat", "copy_", "clone", "_to_copy", "to", "contiguous", "index_copy_"}:
        family = "copy_cast_or_kv_cache_update"
        gap = "tensor lifetime, cast and KV-cache memory-update lowering is missing"
    elif name in {"arange", "full", "zeros", "ones", "empty", "lift_fresh", "scalar_tensor"}:
        family = "constant_or_index_generation"
        gap = "constant binding/index-generation lowering is missing; shape knowledge is not tensor data"
    elif name in {
        "le",
        "lt",
        "ge",
        "gt",
        "eq",
        "ne",
        "where",
        "masked_fill",
        "bitwise_and",
        "bitwise_not",
        "any",
        "all",
    }:
        family = "mask_compare_select"
        gap = "Boolean/index semantics, broadcasting and mask materialization lowering is missing"
    elif name == "dropout":
        family = "inference_dropout_identity"
        gap = "identity elimination requires a checked inference/training=false guard and alias mapping"
    elif name in {
        "add",
        "sub",
        "mul",
        "div",
        "exp",
        "pow",
        "neg",
        "silu",
        "gelu",
        "sigmoid",
        "sin",
        "cos",
    }:
        family = "elementwise_or_composite_arithmetic"
        gap = "vector ISA primitives exist, but ATen shape/broadcast/dtype-to-MLX lowering is not implemented"
    else:
        family = "unclassified"
        gap = "operator must receive an explicit semantic and backend implementation review"
    return {
        "candidate_family": family,
        "status": "missing_lowering",
        "implemented_entry": None,
        "gap": gap,
    }


class ModelExecutionInventory(TorchDispatchMode):
    def __init__(self, model, request_id="reference-0", capture_bindings=False, observer=None):
        super().__init__()
        self.request_id = request_id
        self.forward_id = -1
        self.phase = "outside_forward"
        self.stack = []
        self.operations = []
        self.boundaries = []
        self.tensors = {}
        self.references = {}
        self.counter = 0
        self.names = {}
        self.capture_bindings = capture_bindings
        self.produced = set()
        self.external_snapshots = {}
        self.bindings = {}
        self.observer = observer
        for name, tensor in (*model.named_parameters(), *model.named_buffers()):
            self.names.setdefault(id(tensor), (weakref.ref(tensor), []))[1].append(name)
        self.model = model
        if capture_bindings:
            for name, value in model.named_buffers():
                if value.numel() * value.element_size() > 1024 * 1024:
                    raise RuntimeError(f"large model buffer requires an explicit binding: {name}")
                identifier = self.tensor(value)["tensor_id"]
                self.bindings[identifier] = {
                    "kind": "buffer",
                    "name": name,
                    "values": self.small(value.detach().cpu().tolist()),
                }

    def bind_input(self, name, value):
        if not self.capture_bindings:
            raise RuntimeError("binding capture was not enabled")
        if value.numel() > 65536:
            raise RuntimeError("large input requires a binary asset binding")
        identifier = self.tensor(value)["tensor_id"]
        self.bindings[identifier] = {
            "kind": "input",
            "name": name,
            "values": self.small(value.detach().cpu().tolist()),
        }
        return identifier

    def capture_external_inputs(self, values):
        if isinstance(values, torch.Tensor):
            identifier = self.tensor(values)["tensor_id"]
            named = self.tensors[identifier]["parameter_or_buffer_names"]
            if identifier not in self.produced and identifier not in self.bindings and not named:
                if values.numel() > 16:
                    raise RuntimeError(
                        f"unbound nonparameter tensor {identifier} requires explicit input binding"
                    )
                if identifier not in self.external_snapshots:
                    # Keep only tiny values on the original device. Copy to host
                    # once after the run, never synchronizing each intercepted op.
                    with torch._C._DisableTorchDispatch():
                        self.external_snapshots[identifier] = values.detach().clone()
        elif isinstance(values, (list, tuple)):
            for value in values:
                self.capture_external_inputs(value)
        elif isinstance(values, dict):
            for value in values.values():
                self.capture_external_inputs(value)

    def tensor(self, value):
        key = id(value)
        existing = self.references.get(key)
        if existing is not None and existing[0]() is value:
            return {
                "tensor_id": existing[1],
                "shape": list(value.shape),
                "stride": list(value.stride()),
                "dtype": str(value.dtype),
            }
        identifier = f"t{self.counter}"
        self.counter += 1
        self.references[key] = (weakref.ref(value), identifier)
        metadata = {
            "shape": list(value.shape),
            "stride": list(value.stride()),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "parameter_or_buffer_names": (
                self.names[key][1] if key in self.names and self.names[key][0]() is value else []
            ),
        }
        self.tensors[identifier] = metadata
        return {
            "tensor_id": identifier,
            "shape": list(value.shape),
            "stride": list(value.stride()),
            "dtype": str(value.dtype),
        }

    def small(self, value):
        if isinstance(value, torch.Tensor):
            return self.tensor(value)
        if isinstance(value, (tuple, list)):
            return [self.small(item) for item in value]
        if isinstance(value, dict):
            return {str(key): self.small(item) for key, item in value.items()}
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else {"float_literal": str(value)}
        if isinstance(value, (torch.dtype, torch.device, torch.layout, torch.memory_format)):
            return str(value)
        return {"python_type": f"{type(value).__module__}.{type(value).__name__}"}

    def context(self):
        path = self.stack[-1] if self.stack else "<generation-control>"
        layer = re.search(r"(?:layers|layer|blocks)\.(\d+)(?:\.|$)", path)
        return {
            "request_id": self.request_id,
            "forward_id": self.forward_id,
            "phase": self.phase,
            "module_path": path,
            "layer_idx": int(layer[1]) if layer else None,
        }

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        inputs = self.small(args)
        keyword_inputs = self.small(kwargs)
        if self.capture_bindings:
            self.capture_external_inputs((args, kwargs))
        result = func(*args, **kwargs)
        outputs = self.small(result)

        def mark(value):
            if isinstance(value, dict):
                if "tensor_id" in value:
                    self.produced.add(value["tensor_id"])
                else:
                    for child in value.values():
                        mark(child)
            elif isinstance(value, list):
                for child in value:
                    mark(child)

        mark(outputs)
        self.operations.append(
            {
                "operator_id": len(self.operations),
                "operator": str(func),
                "schema": str(func._schema),
                "mutable": func._schema.is_mutable,
                **self.context(),
                "inputs": inputs,
                "kwargs": keyword_inputs,
                "outputs": outputs,
            }
        )
        if self.observer is not None:
            # Diagnostic observations never become executable input bindings.
            # The callback must preserve the returned tensor and stay bounded.
            with torch._C._DisableTorchDispatch():
                self.observer(self.operations[-1], result)
        return result

    @contextlib.contextmanager
    def attach(self):
        handles = []
        try:
            for name, module in self.model.named_modules():
                path = name or "<model>"

                def before(module, args, kwargs, path=path):
                    self.stack.append(path)
                    self.boundaries.append(
                        {
                            "event": "enter",
                            **self.context(),
                            "inputs": self.small(args),
                            "kwargs": self.small(kwargs),
                        }
                    )

                def after(module, args, kwargs, output, path=path):
                    if not self.stack or self.stack[-1] != path:
                        raise RuntimeError("model hook nesting changed")
                    self.boundaries.append(
                        {"event": "exit", **self.context(), "outputs": self.small(output)}
                    )
                    self.stack.pop()

                handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
                handles.append(
                    module.register_forward_hook(after, with_kwargs=True, always_call=True)
                )
            yield self
        finally:
            for handle in handles:
                handle.remove()
            if self.stack:
                raise RuntimeError("unclosed model trace boundary")

    @contextlib.contextmanager
    def step(self, forward_id, phase):
        self.forward_id, self.phase = forward_id, phase
        with self:
            yield

    def report(self):
        bindings = dict(self.bindings)
        for identifier, value in self.external_snapshots.items():
            bindings[identifier] = {
                "kind": "captured_small_constant",
                "values": self.small(value.cpu().tolist()),
            }
        counts = Counter(event["operator"] for event in self.operations)
        routes = {
            operator: {**candidate_route(operator), "dynamic_calls": count}
            for operator, count in sorted(counts.items())
        }
        return {
            "classification": "real_model_operator_inventory_not_mlx_execution",
            "request_id": self.request_id,
            "operations": self.operations,
            "boundaries": self.boundaries,
            "tensors": self.tensors,
            "bindings": bindings,
            "routes": routes,
            "coverage": {
                "observed_calls": len(self.operations),
                "operator_kinds": len(counts),
                "compiled_and_executed_calls": 0,
                "missing_calls": len(self.operations),
            },
            "mlx_model_execution": "not_executed",
            "inference_performance_eligible": False,
            "observed_tensor_dtypes": dict(
                Counter(tensor["dtype"] for tensor in self.tensors.values())
            ),
        }


def require_model_performance_ready(report):
    coverage = report.get("coverage", {})
    if (
        report.get("mlx_model_execution") != "verified_end_to_end"
        or report.get("mlx_system_verified") is not True
        or report.get("mlx_hardware_mapping_complete") is not True
        or not report.get("numerical_comparison", {}).get("passed")
        or coverage.get("observed_calls", 0) == 0
        or coverage.get("compiled_and_executed_calls", 0) != coverage.get("observed_calls")
        or coverage.get("missing_calls", 1) != 0
        or report.get("fallback_tensor_compute_calls", 1) != 0
    ):
        raise RuntimeError(
            "model inference performance is blocked until complete MLX execution and numerical validation"
        )


def require_model_suite_performance_ready(reports, required_models):
    if not required_models:
        raise RuntimeError("required model list must be nonempty")
    missing = sorted(set(required_models) - set(reports))
    if missing:
        raise RuntimeError(f"required models have no end-to-end evidence: {missing}")
    for model in required_models:
        require_model_performance_ready(reports[model])
