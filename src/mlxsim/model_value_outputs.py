"""Value definitions for scalar-output and explicitly registered tuple-view sources."""


def value_outputs(node):
    if node["kind"] != "split":
        if "split_outputs" in node:
            raise ValueError("tuple outputs are only registered for split")
        return [(node["id"], node["output"])]
    outputs = node.get("split_outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("split requires a nonempty output list")
    for index, output in enumerate(outputs):
        if set(output) != {"id", "output"} or output["id"] != f"{node['id']}:{index}":
            raise ValueError("split output identity is not canonical")
    return [(entry["id"], entry["output"]) for entry in outputs]


def require_value_contract(program):
    split = any(node["kind"] == "split" for node in program["nodes"])
    expected = "mlx_tensor_semantics_v2" if split else "mlx_tensor_semantics_v1"
    if program.get("schema", "mlx_tensor_semantics_v1") != expected or (split and program.get("value_contract") != "tuple_view_outputs_v1"):
        raise ValueError("unsupported/missing tuple-view program contract")
    for node in program["nodes"]:
        value_outputs(node)
    if not split:
        return
    owners = {node["id"] for node in program["nodes"] if node["kind"] == "split"}
    declared = set(program.get("assets", {}))
    def check_refs(value):
        if isinstance(value, dict):
            if "value" in value:
                if value["value"] in owners:
                    raise ValueError("internal split owner cannot be a tensor operand")
            else:
                for child in value.values(): check_refs(child)
        elif isinstance(value, list):
            for child in value: check_refs(child)
    for node in program["nodes"]:
        names = [node["id"]] + ([name for name,_ in value_outputs(node)] if node["kind"] == "split" else [])
        for name in names:
            if name in declared: raise ValueError("duplicate tuple-program value definition")
            declared.add(name)
        for field in ("args", "kwargs", "control_dependencies"): check_refs(node.get(field))
    for output in program.get("outputs", []):
        if any(output.get(role) in owners for role in ("logits", "token")):
            raise ValueError("internal split owner cannot be exported as a tensor")
