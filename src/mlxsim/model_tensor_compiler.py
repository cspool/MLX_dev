"""Lower a captured complete model to typed native tensor semantics.

This is the functional compilation stage. Hardware/ISA scheduling and system
memory integration remain separate, mandatory gates; no performance is certified.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from mlxsim.model_matrix_program import matrix_program
from mlxsim.model_vector_program import FLOAT_KINDS, vector_program
from mlxsim.model_dtype_lowering import lower_softmax_input_cast
from mlxsim.model_memory_program import Planner
from mlxsim.model_control_program import control_program
from mlxsim.model_value_outputs import value_outputs

ROUTES = {
    "aten.embedding.default": "embedding",
    "aten.linear.default": "linear",
    "aten.matmul.default": "matmul",
    "aten.mm.default": "matmul",
    "aten.arange.default": "arange",
    "aten.add.Tensor": "add",
    "aten.mul.Tensor": "mul",
    "aten.le.Tensor": "le",
    "aten.ge.Scalar": "ge",
    "aten.__and__.Tensor": "bitwise_and",
    "aten.bitwise_and.Tensor": "bitwise_and",
    "aten.all.default": "all",
    "aten.is_nonzero.default": "guard",
    "aten._local_scalar_dense.default": "guard",
    "aten.where.ScalarOther": "where",
    "aten.pow.Tensor_Scalar": "pow",
    "aten.rsqrt.default": "rsqrt",
    "aten.mean.dim": "mean",
    "aten.silu.default": "silu",
    "aten.cos.default": "cos",
    "aten.sin.default": "sin",
    "aten.neg.default": "neg",
    "aten.softmax.int": "softmax",
    "aten.argmax.default": "argmax",
    "aten.cat.default": "cat",
    "aten.contiguous.default": "contiguous",
    "aten.to.dtype": "cast",
    "aten._to_copy.default": "cast",
    "aten.to.dtype_layout": "cast_device",
    "aten.to.device": "cast_device",
    "aten.view.default": "reshape",
    "aten.reshape.default": "reshape",
    "aten.transpose.int": "transpose",
    "aten.unsqueeze.default": "unsqueeze",
    "aten.squeeze.dim": "squeeze",
    "aten.split.Tensor": "split",
    "aten.index.Tensor": "advanced_index",
    "aten.new_ones.default": "new_ones",
    "aten.expand.default": "expand",
    "aten.slice.Tensor": "slice",
    "aten.select.int": "select",
    "aten.detach_.default": "alias",
    "aten.lift_fresh.default": "alias",
    "aten.dropout.default": "dropout_inference",
}
DTYPES = {
    "torch.float16": "f16",
    "torch.float32": "f32",
    "torch.int64": "i64",
    "torch.bool": "bool",
}

# Only the recorded overloads/attributes with implemented inference semantics
# are accepted. A familiar ATen name is not permission to ignore its attributes.
ARITY = {
    "split": (2, 3),
    "squeeze": (2, 2), "advanced_index": (2, 2), "new_ones": (2, 2),
    "ge": (2, 2), "bitwise_and": (2, 2), "all": (1, 1), "guard": (1, 1),
    "embedding": (2, 5), "linear": (2, 3), "matmul": (2, 2), "arange": (1, 1),
    "add": (2, 2), "mul": (2, 2), "le": (2, 2), "where": (3, 3), "pow": (2, 2),
    "rsqrt": (1, 1), "mean": (2, 3), "silu": (1, 1), "cos": (1, 1), "sin": (1, 1),
    "neg": (1, 1), "softmax": (2, 3), "argmax": (1, 3), "cat": (1, 2),
    "contiguous": (1, 1), "cast": (2, 4), "cast_device": (3, 5),
    "reshape": (2, 2), "transpose": (3, 3), "unsqueeze": (2, 2), "expand": (2, 3),
    "slice": (1, 5), "select": (3, 3), "alias": (1, 1), "dropout_inference": (3, 3),
}
KWARGS = {
    "new_ones": {"dtype", "layout", "device", "pin_memory"},
    "arange": {"dtype", "layout", "device", "pin_memory"}, "add": {"alpha"},
    "mean": {"dtype"}, "cast": {"memory_format"}, "cast_device": {"memory_format"},
    "contiguous": {"memory_format"},
}


def validate_event(event):
    operator = event["operator"]
    if operator not in ROUTES:
        raise ValueError(f"missing native semantic lowering: {operator}")
    kind = ROUTES[operator]
    args, kwargs = event["inputs"], event["kwargs"]
    copy_overload = operator == "aten._to_copy.default"
    layout_overload = operator == "aten.to.dtype_layout"
    first, last = (1, 1) if copy_overload or layout_overload else ARITY[kind]
    if not isinstance(args, list) or not first <= len(args) <= last:
        raise ValueError(f"unsupported argument count for {operator}")
    allowed_kwargs = {"dtype", "layout", "device", "pin_memory", "non_blocking", "memory_format"} if copy_overload else KWARGS.get(kind, set())
    if layout_overload:
        allowed_kwargs = {"dtype", "layout", "device", "pin_memory", "non_blocking", "copy", "memory_format"}
    if set(kwargs) - allowed_kwargs:
        raise ValueError(f"unsupported attributes for {operator}: {kwargs}")
    if event.get("mutable") and operator != "aten.detach_.default":
        raise ValueError(f"mutable operation requires alias/write tracking: {operator}")
    if kind in {"bitwise_and", "all", "guard"}:
        inputs = args if kind == "bitwise_and" else args[:1]
        if any(not isinstance(arg, dict) or arg.get("dtype") != "torch.bool" for arg in inputs):
            raise ValueError("Boolean control lowering requires Boolean tensor inputs")
        output = event["outputs"]
        if kind == "guard":
            if type(output) is not bool or math.prod(args[0]["shape"]) != 1:
                raise ValueError("control-flow guard requires one Boolean element and scalar Boolean result")
        elif not isinstance(output, dict) or output.get("dtype") != "torch.bool" or (kind == "all" and output.get("shape") != []):
            raise ValueError("Boolean control output contract mismatch")
    if kwargs.get("memory_format") not in (None, "torch.contiguous_format", "torch.preserve_format"):
        raise ValueError("unsupported tensor memory format")
    if kind == "contiguous" and kwargs.get("memory_format") == "torch.preserve_format":
        raise ValueError("contiguous requires contiguous memory format")
    if kwargs.get("layout") not in {None, "torch.strided"}:
        raise ValueError("only strided tensor layout is implemented")
    if kind == "new_ones" or layout_overload:
        if kwargs.get("pin_memory") is not None and type(kwargs["pin_memory"]) is not bool:
            raise ValueError("pin_memory must be Boolean or None")
        if kwargs.get("pin_memory") not in (None, False):
            raise ValueError("pinned-memory allocation is not registered")
        if kwargs.get("dtype") not in (None, event["outputs"]["dtype"]):
            raise ValueError("dtype disagrees with the actual output contract")
        for flag in ("copy", "non_blocking"):
            if flag in kwargs and type(kwargs[flag]) is not bool:
                raise ValueError("conversion flags must be Boolean")
        if kwargs.get("dtype") is None and args[0]["dtype"] != event["outputs"]["dtype"]:
            raise ValueError("implicit dtype cannot change the input precision")
    if kind == "dropout_inference" and args[2] is not False:
        raise ValueError("dropout elimination requires an explicit inference guard")
    if kind == "argmax" and (len(args) < 2 or args[1] is None):
        raise ValueError("whole-tensor argmax needs an explicit lowering")
    if kind == "arange" and (type(args[0]) is not int or args[0] < 0):
        raise ValueError("arange requires a nonnegative integer end")
    if operator == "aten.mm.default" and any(len(arg.get("shape", [])) != 2 for arg in args):
        raise ValueError("mm lowering requires two rank-2 tensors")
    if copy_overload and kwargs.get("dtype") not in (None, event["outputs"]["dtype"]):
        raise ValueError("copy dtype disagrees with its output contract")
    if operator == "aten.mean.dim" and args[0]["dtype"] == "torch.float32" and kwargs.get("dtype") == "torch.float16":
        raise ValueError("narrowing mean.dtype requires a separately registered precision contract")
    return kind


def safetensors_header(path):
    with path.open("rb") as stream:
        size = int.from_bytes(stream.read(8), "little")
        if size < 2 or size > 100 * 1024 * 1024:
            raise ValueError(f"invalid safetensors header: {path}")
        header = json.loads(stream.read(size))
    return header, 8 + size


def compile_inventory(inventory, *, matrix_backend="blas", schedule_options=None, vector_backend="functional", vector_schedule_options=None, memory_backend="functional", memory_schedule_options=None, control_backend="functional", control_schedule_options=None):
    if matrix_backend not in {"blas", "microcode", "scheduled"}:
        raise ValueError("unknown matrix compilation backend")
    if schedule_options is not None and (matrix_backend != "scheduled" or not isinstance(schedule_options, dict)):
        raise ValueError("schedule options require the scheduled matrix backend")
    if vector_backend not in {"functional", "microcode", "scheduled"}:
        raise ValueError("unknown vector compilation backend")
    if vector_schedule_options is not None and (vector_backend != "scheduled" or not isinstance(vector_schedule_options, dict)):
        raise ValueError("vector timing options require the scheduled vector backend")
    if memory_backend not in {"functional", "planned", "scheduled"}:
        raise ValueError("unknown memory compilation backend")
    if memory_schedule_options is not None and (memory_backend != "scheduled" or not isinstance(memory_schedule_options, dict)):
        raise ValueError("memory timing options require the scheduled memory backend")
    if control_backend not in {"functional", "rv64_leaf", "scheduled"}:
        raise ValueError("unknown controller compilation backend")
    if control_schedule_options is not None and (control_backend != "scheduled" or not isinstance(control_schedule_options, dict)):
        raise ValueError("controller timing options require the scheduled controller backend")
    if inventory.get("classification") != "real_model_operator_inventory_not_mlx_execution":
        raise ValueError("expected a real model execution inventory")
    tensors = inventory["tensors"]
    model = inventory["model_identity"]
    path = Path(model["path"])
    headers = {}
    index_file = path / "model.safetensors.index.json"
    if index_file.is_file():
        index = json.loads(index_file.read_text())["weight_map"]
    else:
        single = path / "model.safetensors"
        headers[single] = safetensors_header(single)
        index = {name: single.name for name in headers[single][0] if name != "__metadata__"}
    if not isinstance(index, dict) or any(not isinstance(k,str) or not isinstance(v,str) for k,v in index.items()):
        raise ValueError("checkpoint index must map tensor names to shard files")
    for shard in set(index.values()):
        target = path / shard
        if Path(shard).is_absolute() or not target.resolve().is_relative_to(path.resolve()) or not target.is_file():
            raise ValueError("checkpoint shard is missing or outside the model directory")
    name_map = model.get("checkpoint_name_by_model_parameter")
    if name_map is not None:
        if not isinstance(name_map, dict) or any(not isinstance(k,str) or not isinstance(v,str) for k,v in name_map.items()):
            raise ValueError("checkpoint parameter name mapping must contain strings")
        if len(set(name_map.values())) != len(name_map) or set(name_map.values()) != set(index):
            raise ValueError("checkpoint parameter name mapping must be a complete bijection")
    fingerprints = dict(inventory.get("files", {}))
    for file, info in model.get("files", {}).items():
        if file in fingerprints and fingerprints[file] != info:
            raise ValueError("conflicting checkpoint file fingerprints")
        fingerprints[file] = info
    assets, nodes, routes, outputs, current = {}, [], [], [], {}
    guards = []
    memory_planner = Planner()

    def metadata(source):
        meta = tensors[source]
        if meta["dtype"] not in DTYPES:
            raise ValueError(f"unsupported tensor dtype {meta['dtype']}: {source}")
        return {"dtype": DTYPES[meta["dtype"]], "shape": meta["shape"]}

    def external(source):
        identifier = f"asset:{source}"
        spec = metadata(source)
        binding = inventory.get("bindings", {}).get(source)
        if binding is not None:
            if binding.get("kind") not in {"input", "buffer", "captured_small_constant"}:
                raise ValueError("diagnostic/intermediate tensors are not executable input bindings")
            spec.update(kind="literal", values=binding["values"], origin=binding["kind"])
        elif math.prod(spec["shape"]) == 0:
            spec.update(kind="literal", values=[], origin="empty_tensor")
        else:
            names = tensors[source]["parameter_or_buffer_names"]
            name = next((name for name in names if name in (name_map if name_map is not None else index)), None)
            if name is None:
                raise ValueError(
                    f"unbound tensor {source}: no parameter, input or constant binding"
                )
            checkpoint_name = name_map[name] if name_map is not None else name
            file = path / index[checkpoint_name]
            if file not in headers:
                headers[file] = safetensors_header(file)
            header, data_start = headers[file]
            if checkpoint_name not in header:
                raise ValueError(f"checkpoint index names a missing tensor: {checkpoint_name}")
            weight = header[checkpoint_name]
            dtypes = {"F16": "f16", "F32": "f32", "I64": "i64", "BOOL": "bool"}
            if weight["shape"] != spec["shape"] or dtypes.get(weight["dtype"]) != spec["dtype"]:
                raise ValueError(f"runtime/checkpoint tensor mismatch for {name}")
            first, end = weight["data_offsets"]
            expected = (
                math.prod(spec["shape"]) * {"f16": 2, "f32": 4, "i64": 8, "bool": 1}[spec["dtype"]]
            )
            if type(first) is not int or type(end) is not int or first < 0 or end - first != expected or data_start + end > file.stat().st_size:
                raise ValueError(f"out-of-bounds tensor data for {name}")
            fingerprint = fingerprints.get(str(file))
            if not isinstance(fingerprint, dict) or not isinstance(fingerprint.get("sha256"), str) or len(fingerprint["sha256"]) != 64:
                raise ValueError(f"checkpoint file has no bound fingerprint: {file}")
            if any(c not in "0123456789abcdef" for c in fingerprint["sha256"]):
                raise ValueError(f"checkpoint fingerprint is not canonical SHA256: {file}")
            if "bytes" in fingerprint and fingerprint["bytes"] != file.stat().st_size:
                raise ValueError(f"checkpoint file size differs from its binding: {file}")
            spec.update(
                kind="mapped_file",
                path=str(file),
                byte_offset=data_start + first,
                bytes=expected,
                parameter_name=checkpoint_name,
                file_sha256=fingerprint["sha256"],
            )
            if name_map is not None:
                spec["model_parameter_name"] = name
        assets[identifier] = spec
        memory_planner.add_asset(identifier, spec)
        current[source] = identifier
        return identifier

    def resolve(value):
        if isinstance(value, dict):
            if "tensor_id" in value:
                source = value["tensor_id"]
                return {"value": current[source] if source in current else external(source)}
            return {key: resolve(item) for key, item in value.items()}
        if isinstance(value, list):
            return [resolve(item) for item in value]
        return value

    checks = {check["forward_id"]: check for check in inventory["reference_checks"]}
    for position, event in enumerate(inventory["operations"]):
        operator = event["operator"]
        kind = validate_event(event)
        if event["operator_id"] != position:
            raise ValueError("operator IDs must be unique and sequential")
        guarded = kind == "guard"
        if kind in {"ge", "bitwise_and", "all", "guard"} and control_backend == "functional":
            raise ValueError("Boolean/guard control requires an explicit RV64 leaf or scheduled backend")
        if kind in {"advanced_index", "new_ones", "squeeze", "split"} and memory_backend == "functional":
            raise ValueError("index/generation/squeeze requires an explicit planned or scheduled memory backend")
        split = kind == "split"
        if split:
            if not isinstance(event["outputs"], list) or not event["outputs"] or any(not isinstance(o, dict) or "tensor_id" not in o for o in event["outputs"]):
                raise ValueError("split requires all tensor outputs")
            if len({o["tensor_id"] for o in event["outputs"]}) != len(event["outputs"]):
                raise ValueError("split output tensor identities are duplicated")
        if not guarded and not split and (not isinstance(event["outputs"], dict) or "tensor_id" not in event["outputs"]):
            raise ValueError(f"multi/non-tensor output needs an explicit lowering: {operator}")
        args, kwargs = resolve(event["inputs"]), resolve(event["kwargs"])
        if operator == "aten._to_copy.default":
            # Preserve the mandatory fresh allocation of _to_copy, even when
            # dtype is unchanged. Device placement is still a pending system
            # lowering, just as for the existing functional to.device entry.
            args = [args[0], event["outputs"]["dtype"], kwargs.get("non_blocking", False), True]
            kwargs = {key: value for key, value in kwargs.items() if key == "memory_format"}
        if operator == "aten.to.dtype_layout":
            device = kwargs.get("device") or tensors[event["inputs"][0]["tensor_id"]]["device"]
            args = [args[0], device, event["outputs"]["dtype"], kwargs.get("non_blocking", False), kwargs.get("copy", False)]
            kwargs = {key: value for key, value in kwargs.items() if key == "memory_format"}
        source_output = None if guarded or split else event["outputs"]["tensor_id"]
        identifier = f"v{event['operator_id']}"
        spec = {"dtype": "bool", "shape": []} if guarded else metadata(event["inputs"][0]["tensor_id"]) if split else metadata(source_output)
        if guarded:
            args.append(event["outputs"])
        elif not split:
            spec.update(shape=event["outputs"]["shape"], dtype=DTYPES[event["outputs"]["dtype"]])
        node = {
            "id": identifier,
            "kind": kind,
            "args": args,
            "kwargs": kwargs,
            "output": spec,
            "source_operator": operator,
            "source_operator_id": event["operator_id"],
            "forward_id": event["forward_id"],
            "phase": event["phase"],
            "module_path": event["module_path"],
            "layer_idx": event["layer_idx"],
        }
        if guards:
            node["control_dependencies"] = [{"value": guard} for guard in guards]
        if split:
            node["split_outputs"] = [{"id": f"{identifier}:{index}", "output": {"dtype": DTYPES[out["dtype"]], "shape": out["shape"]}}
                                     for index, out in enumerate(event["outputs"])]
        if matrix_backend != "blas" and kind in {"linear", "matmul"}:
            input_type = DTYPES[event["inputs"][0]["dtype"]]
            node["matrix_program"] = matrix_program(input_type, spec["dtype"], kind == "linear" and len(args) > 2 and args[2] is not None)
        if vector_backend != "functional" and kind in FLOAT_KINDS and spec["dtype"] in {"f16", "f32"}:
            count = 2 if kind in {"add", "mul"} else 1
            input_types = [DTYPES[arg["dtype"]] if isinstance(arg, dict) and "tensor_id" in arg else "f32" for arg in event["inputs"][:count]]
            if any(value not in {"f16", "f32"} for value in input_types):
                raise ValueError("floating vector microcode cannot silently narrow integer tensor operands")
            if kind == "pow" and args[1] != 2:
                raise ValueError("vector pow lowering currently requires exponent 2")
            width = event["inputs"][0]["shape"][-1] if kind in {"mean", "softmax"} else None
            node["vector_program"] = vector_program(kind, input_types, spec["dtype"], width=width, alpha=event["kwargs"].get("alpha", 1))
            lower_softmax_input_cast(node)
        same_device = True
        if kind == "cast_device":
            source_meta = tensors[event["inputs"][0]["tensor_id"]]
            target_meta = tensors[event["outputs"]["tensor_id"]]
            same_device = source_meta.get("device") == target_meta.get("device")
        memory_planner.register(node, planned=memory_backend in {"planned", "scheduled"}, same_device=same_device)
        if control_backend in {"rv64_leaf", "scheduled"} and (kind in {"arange", "le", "argmax", "ge", "bitwise_and", "all", "guard"} or kind in {"add", "mul"} and spec["dtype"] == "i64"):
            if kind in {"arange", "add", "mul"}:
                input_type = "i64"
                if spec["dtype"] != "i64":
                    raise ValueError("controller generator/arithmetic requires int64 output")
            else:
                input_type = DTYPES[event["inputs"][0]["dtype"]]
                if input_type == "bool" and kind in {"le", "ge", "bitwise_and", "all", "guard"}:
                    input_type = "i64"
            node["control_program"] = control_program(kind, input_type)
        nodes.append(node)
        if guarded:
            guards.append(identifier)
        elif split:
            for index, out in enumerate(event["outputs"]):
                current[out["tensor_id"]] = f"{identifier}:{index}"
        else:
            current[source_output] = identifier
        routes.append(
            {
                "source_operator_id": event["operator_id"],
                "native_entry": f"tensor_model::{node['kind']}",
                "status": "compiled_native_semantics",
                "mlx_hardware_mapping": "pending",
                "matrix_microcode_entry": ("mlx::matrix_schedule::Simulator" if matrix_backend == "scheduled" else "mlx::tensor_model::execute_matrix_program") if "matrix_program" in node else None,
                "matrix_microcode_profile": node.get("matrix_program", {}).get("profile"),
                "vector_microcode_entry": ("mlx::vector_schedule::Simulator" if vector_backend == "scheduled" else "mlx::vector_model::execute") if "vector_program" in node else None,
                "memory_plan_entry": ("mlx::memory_model::Simulator" if memory_backend == "scheduled" else "mlx::memory_model::execute") if "memory_program" in node else None,
                "control_entry": ("mlx::control_schedule::Simulator" if control_backend == "scheduled" else "mlx::control_model::execute") if "control_program" in node else None,
            }
        )
        if node["kind"] == "argmax" and node["phase"] == "token_selection":
            outputs.append(
                {"forward_id": node["forward_id"], "logits": args[0]["value"], "token": identifier}
            )
    if {output["forward_id"] for output in outputs} != set(checks):
        raise ValueError("not every generation step has its own computed logits/token output")
    # Record last use of each SSA value. Aliases share storage, but never reuse
    # an SSA definition; in particular decode inputs depend on computed argmax.
    last_use = {}

    def visit(value, at):
        if isinstance(value, dict):
            if "value" in value:
                last_use[value["value"]] = at
            else:
                for child in value.values():
                    visit(child, at)
        elif isinstance(value, list):
            for child in value:
                visit(child, at)

    for position, node in enumerate(nodes):
        visit(node["args"], position)
        visit(node["kwargs"], position)
        visit(node.get("control_dependencies", []), position)
    for output in outputs:
        last_use[output["logits"]] = last_use[output["token"]] = len(nodes)
    release_at = {}
    for position, node in enumerate(nodes):
        last_use.setdefault(node["id"], position)
        for identifier, _ in value_outputs(node):
            last_use.setdefault(identifier, position)
    for value, position in last_use.items():
        release_at.setdefault(position, []).append(value)
    for position, node in enumerate(nodes):
        node["release"] = release_at.get(position, [])
    program = {
        "schema": "mlx_tensor_semantics_v1",
        "assets": assets,
        "nodes": nodes,
        "outputs": outputs,
        "model": {key: model[key] for key in ("family", "variant", "parameters")},
        "input_contract": inventory["input"],
        "timing_mode": "unmodeled",
        "matrix_backend": matrix_backend,
        "vector_backend": vector_backend,
        "matrix_schedule_options": {"trace": False, **(schedule_options or {})} if matrix_backend == "scheduled" else None,
        "mlx_system_verified": False,
        "performance_eligible": False,
    }
    if any(node["kind"] == "split" for node in nodes):
        program["schema"] = "mlx_tensor_semantics_v2"
        program["value_contract"] = "tuple_view_outputs_v1"
    if memory_backend in {"planned", "scheduled"}:
        program["memory_backend"] = memory_backend
    if memory_backend == "scheduled":
        program["memory_schedule_options"] = {"trace": False, **(memory_schedule_options or {})}
    if control_backend in {"rv64_leaf", "scheduled"}:
        program["control_backend"] = control_backend
    if control_backend == "scheduled":
        program["control_schedule_options"] = {"trace": False, **(control_schedule_options or {})}
    if vector_backend == "scheduled":
        program["vector_schedule_options"] = {"trace": False, **(vector_schedule_options or {})}
    digest = hashlib.sha256(json.dumps(program, sort_keys=True).encode()).hexdigest()
    return program, {
        "canonical_program_sha256": digest,
        "source_calls": len(nodes),
        "routes": routes,
        "semantic_lowering_complete": True,
        "mlx_hardware_mapping_complete": False,
        "required_modes": [
            "global tensor memory",
            "FP16 inputs with FP32 GEMM accumulation",
            "FP32 elementwise/reduction",
            "integer/Boolean indices",
            "KV cache lifetime",
        ],
    }
