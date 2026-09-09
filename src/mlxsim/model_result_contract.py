"""Typed QA results; tokenizer metadata is input, reference answers are not."""
from .model_value_outputs import value_outputs

QA_CONTRACT = "mlx-qa-result-v1"
QA_ROLES = ("start_logits", "end_logits", "context_mask", "offsets_utf8")
QA_SELECTION = "reference_f64_start_plus_end_context_only_max30_first_lexicographic_tie"


def result_roles(program):
    return QA_ROLES if program.get("output_contract") == QA_CONTRACT else ("logits", "token")


def compile_qa_outputs(inventory, current, assets, nodes):
    """No reference values, file paths, scores or answers enter this descriptor."""
    result = []
    for check in inventory["reference_checks"]:
        forward = check["forward_id"]
        outputs = check.get("outputs")
        if not isinstance(outputs, dict) or set(outputs) != {"start_logits", "end_logits"}:
            raise ValueError("QA requires both named computed outputs")
        mask, offsets, context = check.get("context_mask"), check.get("offsets"), check.get("context")
        if (not isinstance(mask, list) or not mask or not all(type(x) is bool for x in mask)
                or not any(mask) or not isinstance(context, str) or not isinstance(offsets, list)
                or len(offsets) != len(mask) or check.get("span_selection") != QA_SELECTION):
            raise ValueError("invalid QA tokenizer/selection input contract")
        n = len(mask)
        for role in ("input_ids", "token_type_ids", "attention_mask"):
            tensor = check["input_tensors"][role]
            binding = inventory.get("bindings", {}).get(tensor["tensor_id"], {})
            if (tensor["dtype"] != "torch.int64" or tensor["shape"] != [1, n]
                    or binding.get("kind") != "input" or binding.get("values") != check["input_values"][role]
                    or len(binding["values"]) != 1 or len(binding["values"][0]) != n
                    or any(type(v) is not int for v in binding["values"][0])):
                raise ValueError("QA input metadata differs from actual bound input")
        attention = check["input_values"]["attention_mask"][0]
        segments = check["input_values"]["token_type_ids"][0]
        if any(a not in (0, 1) or t not in (0, 1) or (m and (a != 1 or t != 1))
               for a, t, m in zip(attention, segments, mask, strict=True)):
            raise ValueError("QA context includes padding/question tokens")
        prefix = [0]
        for char in context:
            prefix.append(prefix[-1] + len(char.encode("utf-8")))
        byte_offsets = []
        previous = (0, 0)
        for eligible, pair in zip(mask, offsets, strict=True):
            if not isinstance(pair, list) or len(pair) != 2 or any(type(x) is not int or x < 0 for x in pair):
                raise ValueError("invalid QA tokenizer offsets")
            if eligible:
                a, b = pair
                if not (a < b <= len(context) and a >= previous[0] and b >= previous[1]):
                    raise ValueError("invalid QA context offsets/order")
                previous = (a, b)
                byte_offsets.append([prefix[a], prefix[b]])
            else:
                byte_offsets.append([0, 0])
        row = {"forward_id": forward, "context_utf8": context, "max_answer_tokens": 30}
        for role in ("start_logits", "end_logits"):
            source = outputs[role]
            if source.get("dtype") != "torch.float32" or source.get("shape") != [1, n] or source["tensor_id"] not in current:
                raise ValueError("QA result requires a computed F32 [1, sequence] tensor")
            row[role] = current[source["tensor_id"]]
        for role, dtype, shape, values in (("context_mask", "bool", [n], mask), ("offsets_utf8", "i64", [n, 2], byte_offsets)):
            name = f"qa-input:{forward}:{role}"
            if name in assets:
                raise ValueError("duplicate QA metadata asset")
            assets[name] = dict(kind="literal", dtype=dtype, shape=shape, values=values, origin="qa_tokenizer_input")
            row[role] = name
        result.append(row)
    return result


def validate_result_contract(program):
    qa = program.get("output_contract") == QA_CONTRACT
    if ("output_contract" in program and not qa) or (program.get("schema") == "mlx_tensor_semantics_v4") != qa:
        raise ValueError("unsupported/missing QA result contract")
    if not qa:
        return
    specs = {name: (spec, node) for node in program["nodes"] for name, spec in value_outputs(node)}
    forwards = set()
    if not program["outputs"]:
        raise ValueError("QA outputs are missing")
    for out in program["outputs"]:
        if set(out) != {"forward_id", "context_utf8", "max_answer_tokens", *QA_ROLES}:
            raise ValueError("noncanonical QA result descriptor")
        forward = out["forward_id"]
        if type(forward) is not int or not 0 <= forward <= 2147483647 or forward in forwards:
            raise ValueError("invalid/duplicate QA forward")
        forwards.add(forward)
        if out["max_answer_tokens"] != 30 or type(out["max_answer_tokens"]) is not int or not isinstance(out["context_utf8"], str):
            raise ValueError("invalid QA selection policy")
        n = None
        for role in ("start_logits", "end_logits"):
            if out[role] not in specs:
                raise ValueError("QA output must be computed, not an asset/internal owner")
            spec, node = specs[out[role]]
            if (spec["dtype"] != "f32" or len(spec["shape"]) != 2 or spec["shape"][0] != 1
                    or type(spec["shape"][1]) is not int or spec["shape"][1] <= 0 or node["forward_id"] != forward):
                raise ValueError("QA output shape/dtype/forward mismatch")
            if n is not None and n != spec["shape"][1]:
                raise ValueError("QA output lengths differ")
            n = spec["shape"][1]
        if out["start_logits"] == out["end_logits"]:
            raise ValueError("QA needs distinct named output bindings")
        for role, dtype, shape in (("context_mask", "bool", [n]), ("offsets_utf8", "i64", [n, 2])):
            asset = program["assets"].get(out[role], {})
            if (set(asset) != {"kind", "dtype", "shape", "values", "origin"} or asset.get("kind") != "literal"
                    or asset.get("origin") != "qa_tokenizer_input" or asset.get("dtype") != dtype or asset.get("shape") != shape):
                raise ValueError("QA metadata must be explicit tokenizer input assets")
        mask = program["assets"][out["context_mask"]]["values"]
        offsets = program["assets"][out["offsets_utf8"]]["values"]
        if not isinstance(mask, list) or len(mask) != n or any(type(v) is not bool for v in mask) or not any(mask) or not isinstance(offsets, list) or len(offsets) != n:
            raise ValueError("invalid QA mask/offset input values")
        text = out["context_utf8"].encode("utf-8")
        boundaries = {len(out["context_utf8"][:i].encode("utf-8")) for i in range(len(out["context_utf8"]) + 1)}
        previous = (0, 0)
        for flag, pair in zip(mask, offsets, strict=True):
            if not isinstance(pair, list) or len(pair) != 2 or any(type(x) is not int for x in pair):
                raise ValueError("invalid QA byte offsets")
            a, b = pair
            if not flag:
                if pair != [0, 0]: raise ValueError("masked QA offsets must be zero")
            elif not (a in boundaries and b in boundaries and a < b <= len(text) and a >= previous[0] and b >= previous[1]):
                raise ValueError("invalid QA UTF8 offsets/order")
            else:
                previous = (a, b)
        if any(out[role] in node.get("release", []) for node in program["nodes"] for role in QA_ROLES):
            raise ValueError("QA result released before readback")
    if forwards != {node["forward_id"] for node in program["nodes"]}:
        raise ValueError("QA does not cover every forward")


def verify_qa_result(program, spec, actual):
    """Audit actual saved logits and independently recompute the result, not target execution."""
    from pathlib import Path
    import numpy as np
    def require(condition, message):
        if not condition: raise RuntimeError(message)
    require(actual.get("output_contract") == QA_CONTRACT and actual.get("forward_id") == spec["forward_id"], "QA result identity differs")
    require(set(actual.get("outputs", {})) == {"start_logits", "end_logits"}, "QA named outputs missing")
    mask = program["assets"][spec["context_mask"]]["values"]
    n = len(mask)
    arrays = []
    files = []
    for role in ("start_logits", "end_logits"):
        row = actual["outputs"][role]
        require(row.get("value_id") == spec[role] and row.get("dtype") == "f32" and row.get("shape") == [1, n]
                and row.get("bytes") == n * 4, "QA computed output descriptor differs")
        path = Path(row["file"])
        require(path.is_file() and path.stat().st_size == n * 4, "QA computed output bytes missing")
        files.append(path.resolve())
        values = np.fromfile(path, dtype="<f4")
        require(bool(np.isfinite(values).all()), "QA computed logits contain nonfinite values")
        arrays.append(values)
    require(files[0] != files[1], "QA named outputs alias the same file")
    candidates = []
    for i in range(n):
        for j in range(i, min(n, i + 30)):
            if not mask[j]: break
            candidates.append((float(arrays[0][i]) + float(arrays[1][j]), i, j))
    require(bool(candidates), "QA has no eligible span")
    score, first, last = max(candidates, key=lambda item: item[0])
    offsets = program["assets"][spec["offsets_utf8"]]["values"]
    a, b = offsets[first][0], offsets[last][1]
    span = actual.get("span", {})
    require(span == dict(start=first, end=last, score_f64=score, start_byte=a, end_byte=b,
                         text=spec["context_utf8"].encode("utf-8")[a:b].decode("utf-8")), "QA answer was not derived from actual computed logits")
    post = actual.get("postprocessing", {})
    require(post == dict(entry="mlx_qa_select_span", backend="portable_c_host_execution_cpu_timing_unmodeled",
                         candidate_spans=len(candidates), max_answer_tokens=30, score_dtype="f64",
                         tie_policy="first_lexicographic", metadata_readback_verified=True,
                         actual_cpu_execution=False, target_cpu_cycles=None), "QA host execution/timing provenance differs")
    return 5 * n, 25 * n
