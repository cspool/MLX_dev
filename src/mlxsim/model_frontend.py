"""Automatic PyTorch FX and ONNX frontends for canonical MLX workload graphs."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper
from torch import nn
from torch.fx import GraphModule, Node, Tracer
from torch.fx.passes.shape_prop import ShapeProp

from mlxsim.schema import KernelProfile, StageSpec, Workload

SUPPORTED_KINDS = {
    "rmsnorm",
    "bsmm",
    "fft_cmp",
    "attention",
    "elementwise",
}


class RMSNorm(nn.Module):
    mlx_kind = "rmsnorm"

    def __init__(self, dimension: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + 1.0e-5)
        return value * scale * self.weight


class StructuredLinear(nn.Module):
    mlx_kind = "bsmm"

    def __init__(self, dimension: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dimension, dimension))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.matmul(value, self.weight)


class FFTCompression(nn.Module):
    mlx_kind = "fft_cmp"

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.fft.fft(value, dim=1, norm="ortho").real


class StructuredAttention(nn.Module):
    mlx_kind = "attention"

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(value, value.transpose(-1, -2)) / math.sqrt(value.shape[-1])
        return torch.matmul(torch.softmax(scores, dim=-1), value)


class StructuredSilu(nn.Module):
    mlx_kind = "elementwise"

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(value)


class AutoStructuredBlock(nn.Module):
    """A real executable module used to exercise the automatic frontends."""

    def __init__(self, dimension: int):
        super().__init__()
        self.norm = RMSNorm(dimension)
        self.qkv = StructuredLinear(dimension)
        self.fft_cmp = FFTCompression()
        self.attention = StructuredAttention()
        self.output = StructuredLinear(dimension)
        self.silu = StructuredSilu()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.norm(value)
        value = self.qkv(value)
        value = self.fft_cmp(value)
        value = self.attention(value)
        value = self.output(value)
        return self.silu(value)


class MlxTracer(Tracer):
    def is_leaf_module(self, module: nn.Module, qualified_name: str) -> bool:
        return hasattr(module, "mlx_kind") or super().is_leaf_module(module, qualified_name)


def _node_dependencies(node: Node, legal_names: set[str]) -> list[str]:
    result: list[str] = []

    def visit(value: Any) -> Any:
        if isinstance(value, Node) and value.name in legal_names:
            result.append(value.name)
        return value

    torch.fx.map_arg((node.args, node.kwargs), visit)
    return list(dict.fromkeys(result))


def import_fx_graph(
    *, input_shape: tuple[int, ...], hidden_dimension: int, seed: int
) -> tuple[dict[str, Any], GraphModule]:
    torch.manual_seed(seed)
    module = AutoStructuredBlock(hidden_dimension).eval()
    tracer = MlxTracer()
    graph_module = GraphModule(module, tracer.trace(module))
    sample = torch.randn(input_shape, dtype=torch.float32)
    ShapeProp(graph_module).propagate(sample)
    legal_nodes = [node for node in graph_module.graph.nodes if node.op == "call_module"]
    legal_names = {node.name for node in legal_nodes}
    nodes: list[dict[str, Any]] = []
    for node in legal_nodes:
        submodule = graph_module.get_submodule(str(node.target))
        kind = str(submodule.mlx_kind)
        tensor_meta = node.meta.get("tensor_meta")
        if tensor_meta is None:
            raise ValueError(f"FX ShapeProp did not produce tensor metadata for {node.name}")
        nodes.append(
            {
                "id": node.name,
                "kind": kind,
                "depends_on": _node_dependencies(node, legal_names),
                "shape": [int(value) for value in tensor_meta.shape],
                "dtype": str(tensor_meta.dtype).removeprefix("torch."),
                "source": {
                    "frontend": "pytorch_fx",
                    "node_name": node.name,
                    "node_op": node.op,
                    "target": str(node.target),
                    "module_type": type(submodule).__name__,
                },
            }
        )
    return {
        "schema_version": 1,
        "frontend": "pytorch_fx",
        "input_shape": list(input_shape),
        "nodes": nodes,
    }, graph_module


def build_onnx_model(
    *, input_shape: tuple[int, ...], hidden_dimension: int, seed: int
) -> onnx.ModelProto:
    rng = np.random.default_rng(seed)
    shape = list(input_shape)
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)
    output_info = helper.make_tensor_value_info("silu_out", TensorProto.FLOAT, shape)
    value_infos = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
        for name in ("norm_out", "qkv_out", "fft_out", "attention_out", "output_out")
    ]
    scale = numpy_helper.from_array(np.ones(hidden_dimension, dtype=np.float32), "norm_scale")
    bias = numpy_helper.from_array(np.zeros(hidden_dimension, dtype=np.float32), "norm_bias")
    qkv_weight = numpy_helper.from_array(
        rng.normal(0.0, 0.02, (hidden_dimension, hidden_dimension)).astype(np.float32),
        "qkv_weight",
    )
    output_weight = numpy_helper.from_array(
        rng.normal(0.0, 0.02, (hidden_dimension, hidden_dimension)).astype(np.float32),
        "output_weight",
    )
    nodes = [
        helper.make_node(
            "LayerNormalization",
            ["input", "norm_scale", "norm_bias"],
            ["norm_out"],
            name="norm",
            axis=-1,
            epsilon=1.0e-5,
        ),
        helper.make_node("MatMul", ["norm_out", "qkv_weight"], ["qkv_out"], name="qkv"),
        helper.make_node(
            "FFTCompression", ["qkv_out"], ["fft_out"], name="fft_cmp", domain="mlx"
        ),
        helper.make_node(
            "StructuredAttention",
            ["fft_out"],
            ["attention_out"],
            name="attention",
            domain="mlx",
        ),
        helper.make_node(
            "MatMul", ["attention_out", "output_weight"], ["output_out"], name="output"
        ),
        helper.make_node("Silu", ["output_out"], ["silu_out"], name="silu", domain="mlx"),
    ]
    graph = helper.make_graph(
        nodes,
        "auto_structured_block",
        [input_info],
        [output_info],
        initializer=[scale, bias, qkv_weight, output_weight],
        value_info=value_infos,
    )
    model = helper.make_model(
        graph,
        producer_name="mlx-paper-repro",
        opset_imports=[helper.make_opsetid("", 20), helper.make_opsetid("mlx", 1)],
    )
    onnx.checker.check_model(model)
    return model


def _onnx_shape_map(model: onnx.ModelProto) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    values = [*model.graph.input, *model.graph.value_info, *model.graph.output]
    for value in values:
        dimensions = value.type.tensor_type.shape.dim
        result[value.name] = [int(item.dim_value) for item in dimensions]
    return result


def import_onnx_graph(model: onnx.ModelProto) -> dict[str, Any]:
    kind_map = {
        "LayerNormalization": "rmsnorm",
        "MatMul": "bsmm",
        "FFTCompression": "fft_cmp",
        "StructuredAttention": "attention",
        "Silu": "elementwise",
    }
    initializer_names = {item.name for item in model.graph.initializer}
    producer = {
        output: node.name for node in model.graph.node for output in node.output
    }
    shapes = _onnx_shape_map(model)
    nodes: list[dict[str, Any]] = []
    for node in model.graph.node:
        if node.op_type not in kind_map:
            raise ValueError(f"unsupported ONNX operator: {node.op_type}")
        dependencies = [
            producer[name]
            for name in node.input
            if name not in initializer_names and name in producer
        ]
        output_name = node.output[0]
        nodes.append(
            {
                "id": node.name,
                "kind": kind_map[node.op_type],
                "depends_on": dependencies,
                "shape": shapes[output_name],
                "dtype": "float32",
                "source": {
                    "frontend": "onnx",
                    "node_name": node.name,
                    "node_op": node.op_type,
                    "domain": node.domain,
                    "output_name": output_name,
                },
            }
        )
    return {
        "schema_version": 1,
        "frontend": "onnx",
        "input_shape": shapes[model.graph.input[0].name],
        "nodes": nodes,
    }


def canonical_signature(graph: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": node["id"],
            "kind": node["kind"],
            "depends_on": node["depends_on"],
            "shape": node["shape"],
            "dtype": node["dtype"],
        }
        for node in graph["nodes"]
    ]


def plan_graph(graph: Mapping[str, Any], planning: Mapping[str, Any]) -> dict[str, Any]:
    nodes = list(graph["nodes"])
    mesh_x, mesh_y = (int(value) for value in planning["mesh"])
    register_count = int(planning["register_count"])
    register_banks = int(planning["register_banks"])
    alignment = int(planning["dma_alignment_bytes"])
    spm_bytes = int(planning["spm_bytes"])
    tensor_bytes = max(
        math.prod(int(value) for value in node["shape"]) * 4 for node in nodes
    )
    aligned_tensor_bytes = math.ceil(tensor_bytes / alignment) * alignment
    planned_nodes: list[dict[str, Any]] = []
    for index, node in enumerate(nodes):
        register = index % register_count
        address = (index % 2) * aligned_tensor_bytes
        planned_nodes.append(
            {
                **node,
                "cdc_id": 0,
                "tag": index + 1,
                "pe": [index % mesh_x, (index // mesh_x) % mesh_y],
                "output_register": register,
                "register_bank": register % register_banks,
                "spm_address": address,
                "dma_bytes": math.prod(int(value) for value in node["shape"]) * 4,
                "memory_live_interval": [index, index + 1],
            }
        )
    return {
        "frontend": graph["frontend"],
        "mesh": [mesh_x, mesh_y],
        "simd_width": int(planning["simd_width"]),
        "spm_bytes": spm_bytes,
        "aligned_tensor_bytes": aligned_tensor_bytes,
        "peak_spm_bytes": 2 * aligned_tensor_bytes,
        "cdcs": [{"id": 0, "nodes": [node["id"] for node in nodes]}],
        "nodes": planned_nodes,
    }


def profile_for_node(node: Mapping[str, Any], model_contract: Mapping[str, Any]) -> dict[str, Any]:
    shape = [int(value) for value in node["shape"]]
    elements = math.prod(shape)
    n = int(model_contract["sequence_length"])
    d = int(model_contract["hidden_dimension"])
    kind = str(node["kind"])
    if kind == "rmsnorm":
        operations, resource = 4 * elements, "compute_frsqrt"
    elif kind == "bsmm":
        operations, resource = 2 * n * d * d * 0.75, "compute_fma"
    elif kind == "fft_cmp":
        operations, resource = 10 * elements * int(math.log2(n)), "compute_fma"
    elif kind == "attention":
        operations, resource = 4 * n * n * d, "compute_fma"
    elif kind == "elementwise":
        operations, resource = 4 * elements, "compute_fexp"
    else:
        raise ValueError(f"unsupported canonical kind: {kind}")
    stage = StageSpec(
        tag=int(node["tag"]),
        name=str(node["id"]),
        compute_resource=resource,
        operations=float(operations),
        load_bytes=float(elements * 4),
        store_bytes=float(elements * 4),
        route_distance=1,
        kernel_class=kind,
    )
    profile = KernelProfile(
        operations=float(operations),
        offchip_bytes=float(elements * 8),
        output_elements=float(elements),
        stages=(stage,),
        metadata={
            "stage_count": 1,
            "source_frontend": node["source"]["frontend"],
            "source_node": node["source"]["node_name"],
            "cdc_id": node["cdc_id"],
            "pe": node["pe"],
            "output_register": node["output_register"],
            "spm_address": node["spm_address"],
        },
    )
    workload = Workload(
        kernel="bsmm",
        n=n,
        d=d,
        batch=int(model_contract["batch"]),
        block_size=int(model_contract["block_size"]),
        name=f"{node['id']}-{node['source']['frontend']}",
    )
    return {"workload": workload.to_dict(), "profile": asdict(profile)}


def graph_digest(graph: Mapping[str, Any]) -> str:
    payload = json.dumps(canonical_signature(graph), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def save_onnx(model: onnx.ModelProto, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)
    payload = path.read_bytes()
    return {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


__all__ = [
    "AutoStructuredBlock",
    "build_onnx_model",
    "canonical_signature",
    "graph_digest",
    "import_fx_graph",
    "import_onnx_graph",
    "plan_graph",
    "profile_for_node",
    "save_onnx",
]
