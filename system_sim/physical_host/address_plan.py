"""Replay checked allocation identities for descriptor compilation, not data."""
from .lowering import collect_layouts,BYTES
from mlxsim.model_value_outputs import value_outputs,require_value_contract
from mlxsim.model_result_contract import result_roles


def iter_bindings(program,life):
    require_value_contract(program)
    if not isinstance(life,dict) or not {"initial","events","source_nodes"}<=life.keys():raise ValueError("missing validated lifetime plan")
    layouts=collect_layouts(program);bindings={};ids={};initial=life["initial"]["allocations"]
    if len(initial)!=len(program["assets"]) or life["source_nodes"]!=len(program["nodes"]) or len(life["events"])!=len(program["nodes"]):raise ValueError("lifetime plan does not cover this model")
    for index,(name,allocation) in enumerate(zip(sorted(program["assets"]),initial,strict=True),1):
        layout=layouts[name]
        if allocation["id"]!=index or allocation["bytes"]!=layout["storage_elements"]*BYTES[layout["dtype"]] or allocation["writable"]:raise ValueError("initial bindings differ from lifetime plan")
        bindings[name]={key:allocation[key] for key in ("base","bytes","writable")};ids[name]=allocation["id"]
    def references(value):
        if isinstance(value,dict):
            if "value" in value:yield value["value"]
            else:
                for child in value.values():yield from references(child)
        elif isinstance(value,list):
            for child in value:yield from references(child)
    live=set(program["assets"])
    for node,event in zip(program["nodes"],life["events"],strict=True):
        if node["source_operator_id"]!=event["source_operator_id"] or node["kind"]!=event["kind"]:raise ValueError("lifetime operator ordering changed")
        if any(name not in live for field in ("args","kwargs","control_dependencies") for name in references(node.get(field,[]))):raise ValueError("descriptor references a released SSA value")
        layout=layouts[node["id"]];root=layout["root"];allocation=event["allocation"]
        if allocation["bytes"]!=layout["storage_elements"]*BYTES[layout["dtype"]]:raise ValueError("output allocation extent mismatch")
        binding={key:allocation[key] for key in ("base","bytes","writable")}
        if root in bindings:
            if bindings[root]!=binding or ids[root]!=allocation["id"]:raise ValueError("view changed its physical allocation")
        else:bindings[root]=binding;ids[root]=allocation["id"]
        definitions=[name for name,_ in value_outputs(node)]
        if node["kind"]=="split":
            values=event.get("value_allocations",{})
            if set(values)!=set(definitions) or any(values[name]!=allocation for name in definitions):raise ValueError("split lifetime lost an output or allocation identity")
        elif "value_allocations" in event:raise ValueError("unexpected tuple lifetime definitions")
        if any(name in live for name in definitions):raise ValueError("descriptor SSA was redefined")
        live.update(definitions)
        if node["kind"]=="split":live.add(node["id"]) # Ephemeral container release token, never a tensor binding.
        yield node,layouts,bindings
        for name in node["release"]:
            if name not in live:raise ValueError("descriptor plan has invalid release")
            live.remove(name)
        if node["kind"]=="split":live.discard(node["id"])
    if any(out[role] not in live for out in program["outputs"] for role in result_roles(program)):
        raise ValueError("system result was released before CPU observation")
