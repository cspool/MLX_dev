"""Compile bounded, closed producer/pointwise-consumer pairs.

All source operators retain their existing numerical lowering. Unsupported
pairs remain whole-source dependencies; this is not a general CDC compiler.
"""
import copy
import math
from .model_value_outputs import value_outputs, require_value_contract


def references(value):
    if isinstance(value, dict):
        if "value" in value:
            yield value["value"]
        else:
            for child in value.values():
                yield from references(child)
    elif isinstance(value, list):
        for child in value:
            yield from references(child)


def mapping(node):
    shape = node["output"]["shape"]
    elements = math.prod(shape)
    if elements <= 0:
        raise ValueError("empty pipeline output")
    if "matrix_program" in node:
        if len(shape) < 2 or node["kind"] not in {"linear", "matmul"}:
            raise ValueError("unregistered matrix pipeline")
        linear = node["kind"] == "linear"
        m, n = (math.prod(shape[:-1]) if linear else shape[-2]), shape[-1]
        batches = 1 if linear else math.prod(shape[:-2])
        result = {"kind": "matrix", "elements": elements, "m": m, "n": n, "batches": batches}
        blocks = batches * ((m + 1) // 2) * ((n + 15) // 16)
    elif "vector_program" in node:
        if node["kind"] in {"mean", "softmax"}:
            width = shape[-1] if node["kind"] == "softmax" else 1
            result = {"kind": "reduction", "elements": elements, "row_width": width}
            blocks = elements // width
        else:
            result = {"kind": "vector", "elements": elements}
            blocks = (elements + 15) // 16
    else:
        raise ValueError("producer has no array lowering")
    return result, blocks


def resources(node):
    if "matrix_program" in node:
        p = node["matrix_program"]
        result = {"rf": p["rf_vectors_used"], "spm": (p["spm_bytes_used"] + 63) // 64,
                  "rom": sum(len(p[k]) for k in ("prologue", "body", "epilogue"))}
    else:
        p = node["vector_program"]
        result = {"rf": p["rf_vectors_used"], "spm": (p["spm_bytes_used"] + 63) // 64, "rom": len(p["rom"])}
    if not 0 < result["rf"] <= 16 or not 0 < result["spm"] <= 128 or not 0 < result["rom"] <= 32:
        raise ValueError("invalid_pipeline_resource_requirement")
    return result


def compile_block_pipelines(program, *, event_slots=32):
    if type(event_slots) is not int or not 1 <= event_slots <= 32:
        raise ValueError("completion window requires 1..32 event slots")
    result = copy.deepcopy(program); nodes = result["nodes"]
    require_value_contract(program)
    guards = []
    for node in nodes:
        expected = [{"value": identifier} for identifier in guards]
        if node.get("control_dependencies", []) != expected:
            raise ValueError("missing or foreign control-flow guard dependency")
        if node["kind"] == "guard":
            guards.append(node["id"])
    consumers = {identifier: set() for n in nodes for identifier, _ in value_outputs(n)}
    for index, node in enumerate(nodes):
        for dep in set(references(node["args"])) | set(references(node.get("kwargs", {}))) | set(references(node.get("control_dependencies", []))):
            if dep in consumers:
                consumers[dep].add(index)
    mo, vo = result.get("matrix_schedule_options", {}), result.get("vector_schedule_options", {})
    contexts = min(mo.get("contexts", 2), vo.get("contexts", 2))
    pes = mo.get("rows", 4) * mo.get("columns", 4)
    pairs, used, rejected = [], set(), []
    for producer, node in enumerate(nodes):
        if producer in used or node["kind"] == "split" or len(consumers[node["id"]]) != 1 or not any(k in node for k in ("matrix_program", "vector_program")):
            continue
        consumer = next(iter(consumers[node["id"]])); target = nodes[consumer]
        reason = None
        if consumer in used or producer >= consumer or node["source_operator_id"] >= target["source_operator_id"]:
            reason = "not_a_disjoint_forward_pair"
        elif "vector_program" not in target or target["kind"] in {"mean", "softmax"}:
            reason = "consumer_is_not_pointwise_vector"
        elif node["output"]["shape"] != target["output"]["shape"]:
            reason = "pending_input_requires_shape_remapping"
        elif not any(isinstance(arg, dict) and arg.get("value") == node["id"] for arg in target["args"]):
            reason = "pending_input_is_not_a_direct_tensor_operand"
        elif contexts < 2:
            reason = "producer_consumer_need_two_contexts_per_pe"
        if reason is None:
            try:
                layout, blocks = mapping(node)
                p, c = resources(node), resources(target)
                if p["rf"] + c["rf"] > 16 or p["spm"] + c["spm"] > 128 or p["rom"] + c["rom"] > 32:
                    reason = "joint_rf_spm_or_rom_capacity"
            except ValueError as error:
                reason = str(error)
        if reason:
            rejected.append({"producer": producer, "consumer": consumer, "reason": reason})
            continue
        pairs.append({"producer": producer, "consumer": consumer, "producer_source": node["source_operator_id"],
                      "consumer_source": target["source_operator_id"], "value": node["id"], "mapping": layout,
                      "producer_blocks": blocks, "consumer_blocks": (layout["elements"] + 15) // 16,
                      "producer_resources": p, "consumer_resources": c, "producer_context_limit": pes,
                      "consumer_context_limit": min(pes, (128 - p["spm"]) // c["spm"])})
        used.update((producer, consumer))
    result["block_pipeline_plan"] = {"profile": "mlx-bounded-pair-events-v1", "event_slots": event_slots,
                                      "pairs": pairs, "rejected_pairs": rejected,
                                      "unselected_sources": "existing_whole_source_execution_not_numeric_fallback"}
    return result
