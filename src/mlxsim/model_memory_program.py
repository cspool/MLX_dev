"""Checked affine layouts and bounded data-movement programs.

These are accelerator/runtime memory plans, not a claim of Rocket execution or
Chipyard DMA timing. View elimination and actual transfers remain distinct.
"""

import copy
import math

KINDS = {"embedding", "where", "cat", "cast", "cast_device", "contiguous", "reshape",
         "transpose", "slice", "select", "unsqueeze", "expand", "alias", "dropout_inference"}
TYPE = {"f16": 0, "f32": 1, "i64": 2, "bool": 3}
BYTES = {"f16": 2, "f32": 4, "i64": 8, "bool": 1}
# Transfer-controller operations, separate from the PE arithmetic opcode space.
LOAD, CONVERT, STORE, LOAD_INDEX, LOAD_PREDICATE = range(1, 6)


def strides(shape):
    result, size = [], 1
    for extent in reversed(shape):
        result.append(size)
        size *= max(extent, 1)
    return result[::-1]


def contiguous(layout):
    if math.prod(layout["shape"]) == 0:
        return True
    size = 1
    for extent, step in zip(layout["shape"][::-1], layout["strides"][::-1], strict=True):
        if extent > 1 and step != size:
            return False
        size *= max(extent, 1)
    return True


def dense(layout):
    size = 1
    for step, extent in sorted((step, extent) for step, extent in zip(layout["strides"], layout["shape"], strict=True) if extent > 1):
        if step != size:
            return False
        size *= extent
    return True


def dimension(value, rank, insertion=False):
    limit = rank + int(insertion)
    if value < 0:
        value += limit
    if not 0 <= value < limit:
        raise ValueError("memory layout axis out of bounds")
    return value


def reshape_shape(requested, elements):
    result = list(requested)
    if result.count(-1) > 1 or any(value < -1 for value in result):
        raise ValueError("invalid inferred reshape dimensions")
    known = math.prod(value for value in result if value != -1)
    if -1 in result:
        if not known or elements % known:
            raise ValueError("ambiguous or incompatible inferred reshape")
        result[result.index(-1)] = elements // known
    if math.prod(result) != elements:
        raise ValueError("reshape changes element count")
    return result


def reshape_strides(old, requested):
    """Match new dimensions to contiguous subspaces, including strided ones."""
    if not old["shape"]:
        return strides(requested)
    if math.prod(old["shape"]) == 0:
        return list(old["strides"]) if old["shape"] == requested else strides(requested)
    result, remaining = [0] * len(requested), len(requested) - 1
    base, consumed, packed = old["strides"][-1], 1, 1
    for axis in range(len(old["shape"]) - 1, -1, -1):
        consumed *= old["shape"][axis]
        boundary = axis == 0 or (old["shape"][axis - 1] != 1 and old["strides"][axis - 1] != consumed * base)
        if boundary:
            while remaining >= 0 and (packed < consumed or requested[remaining] == 1):
                result[remaining] = packed * base
                packed *= requested[remaining]
                remaining -= 1
            if packed != consumed:
                return None
            if axis:
                base, consumed, packed = old["strides"][axis - 1], 1, 1
    return result if remaining == -1 else None


def make_layout(dtype, shape, root):
    return {"dtype": dtype, "shape": list(shape), "strides": strides(shape), "offset": 0,
            "root": root, "storage_elements": math.prod(shape)}


class Planner:
    def __init__(self):
        self.layouts = {}

    def add_asset(self, identifier, spec):
        self.layouts[identifier] = make_layout(spec["dtype"], spec["shape"], identifier)

    def register(self, node, *, planned=False, same_device=True):
        kind, args = node["kind"], node["args"]
        output = make_layout(node["output"]["dtype"], node["output"]["shape"], node["id"])
        if not planned or kind not in KINDS:
            self.layouts[node["id"]] = output
            return
        references = []
        def find(value):
            if isinstance(value, dict) and "value" in value:
                references.append(value["value"])
            elif isinstance(value, list):
                for child in value:
                    find(child)
        find(args)
        inputs = {name: copy.deepcopy(self.layouts[name]) for name in references}
        source = self.layouts[args[0]["value"]] if isinstance(args[0], dict) and "value" in args[0] else None
        mode, selector, reason = "transfer", "linear", "materialization"
        if kind in {"alias", "dropout_inference", "transpose", "slice", "select", "unsqueeze", "expand", "reshape"}:
            result = copy.deepcopy(source)
            if kind == "transpose":
                a, b = [dimension(value, len(result["shape"])) for value in args[1:]]
                for key in ("shape", "strides"):
                    result[key][a], result[key][b] = result[key][b], result[key][a]
            elif kind == "unsqueeze":
                axis = dimension(args[1], len(result["shape"]), True)
                step = result["shape"][axis] * result["strides"][axis] if axis < len(result["shape"]) else 1
                result["shape"].insert(axis, 1); result["strides"].insert(axis, step)
            elif kind == "select":
                axis = dimension(args[1], len(result["shape"])); index = args[2]
                if index < 0:
                    index += result["shape"][axis]
                if not 0 <= index < result["shape"][axis]:
                    raise ValueError("select index out of bounds")
                result["offset"] += index * result["strides"][axis]
                del result["shape"][axis]; del result["strides"][axis]
            elif kind == "slice":
                axis = dimension(args[1] if len(args) > 1 else 0, len(result["shape"]))
                size = result["shape"][axis]
                begin, end, step = slice(args[2] if len(args) > 2 else None, args[3] if len(args) > 3 else None, args[4] if len(args) > 4 else 1).indices(size)
                if step <= 0:
                    raise ValueError("only positive memory slice steps are supported")
                result["offset"] += begin * result["strides"][axis]
                result["shape"][axis] = max(0, (end - begin + step - 1) // step)
                result["strides"][axis] *= step
            elif kind == "expand":
                requested = list(args[1]); extra = len(requested) - len(result["shape"])
                if extra < 0:
                    raise ValueError("expand cannot reduce rank")
                result["shape"] = [1] * extra + result["shape"]
                result["strides"] = [0] * extra + result["strides"]
                for i, extent in enumerate(requested):
                    if extent == -1:
                        if i < extra:
                            raise ValueError("new expand axis cannot be inferred")
                        extent = result["shape"][i]
                    if extent < 0 or result["shape"][i] not in (1, extent):
                        raise ValueError("invalid expand extent")
                    if extent != result["shape"][i]:
                        result["strides"][i] = 0
                    result["shape"][i] = extent
            elif kind == "reshape":
                requested = reshape_shape(args[1], math.prod(source["shape"]))
                new_steps = reshape_strides(source, requested)
                if new_steps is None:
                    if node["source_operator"] == "aten.view.default":
                        raise ValueError("view requires incompatible physical layout; no implicit copy is allowed")
                    result = None
                    reason = "reshape_requires_copy"
                else:
                    result["shape"], result["strides"] = requested, new_steps
            if result is not None:
                output, mode, reason = result, "view", "checked_affine_view"
        elif kind in {"cast", "cast_device", "contiguous"}:
            copy_index = 3 if kind == "cast" else 4
            force = kind != "contiguous" and len(args) > copy_index and args[copy_index]
            fmt = node["kwargs"].get("memory_format")
            wants_contiguous = kind == "contiguous" or fmt == "torch.contiguous_format"
            if output["dtype"] == source["dtype"] and not force and same_device and (not wants_contiguous or contiguous(source)):
                output, mode, reason = copy.deepcopy(source), "view", "checked_identity_conversion"
            elif not wants_contiguous and dense(source):
                output["strides"] = list(source["strides"])
                reason = "copy_preserves_dense_layout"
            if not same_device:
                reason = "device_transfer_requires_materialization"
        elif kind == "cat":
            selector = "concat"
            selected = [self.layouts[arg["value"]] for arg in args[0] if self.layouts[arg["value"]]["shape"] != [0]]
            if selected:
                axis = dimension(args[1] if len(args) > 1 else 0, len(selected[0]["shape"]))
                expected = list(selected[0]["shape"]); expected[axis] = 0
                for layout in selected:
                    if layout["dtype"] != output["dtype"] or len(layout["shape"]) != len(expected) or any(a != b for i, (a, b) in enumerate(zip(layout["shape"], selected[0]["shape"], strict=True)) if i != axis):
                        raise ValueError("concat layout/dtype mismatch")
                    expected[axis] += layout["shape"][axis]
                if expected != output["shape"]:
                    raise ValueError("concat output shape mismatch")
        elif kind == "embedding":
            selector = "indexed_rows"
            ids = self.layouts[args[1]["value"]]
            if len(source["shape"]) != 2 or ids["dtype"] != "i64" or output["shape"] != ids["shape"] + [source["shape"][1]] or output["dtype"] != source["dtype"]:
                raise ValueError("embedding memory contract mismatch")
        elif kind == "where":
            selector = "predicate_select"
            if source["dtype"] != "bool":
                raise ValueError("where requires a Boolean predicate")
            expected = [1] * max(len(layout["shape"]) for layout in inputs.values())
            for layout in inputs.values():
                shift = len(expected) - len(layout["shape"])
                for i, extent in enumerate(layout["shape"]):
                    at = shift + i
                    if expected[at] == 1:
                        expected[at] = extent
                    elif extent not in (1, expected[at]):
                        raise ValueError("where operand broadcast mismatch")
            if expected != output["shape"]:
                raise ValueError("where output shape mismatch")
        if output["dtype"] != node["output"]["dtype"] or output["shape"] != node["output"]["shape"]:
            raise ValueError(f"compiled {kind} layout disagrees with source output")
        prefix = [LOAD_INDEX] if selector == "indexed_rows" else [LOAD_PREDICATE] if selector == "predicate_select" else []
        words = [] if mode == "view" else prefix + [LOAD, CONVERT | TYPE[output["dtype"]] << 8, STORE]
        node["memory_program"] = {"profile": "mlx-memory-plan-v1", "kind": kind, "mode": mode,
                                  "selector": selector, "reason": reason, "input_layouts": inputs,
                                  "output_layout": output, "words": words, "register_count": 4,
                                  "register_bytes": 8, "staging_bytes": 128, "chunk_bytes": 64,
                                  "same_reference_device": same_device, "physical_dma_verified": False}
        self.layouts[node["id"]] = copy.deepcopy(output)
