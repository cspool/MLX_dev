"""Bind full-model control routes to host ABI using checked lifetime addresses."""
import argparse
import hashlib
import json
from pathlib import Path

from system_sim.physical_host.lowering import collect_layouts,lower_control,BYTES

ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--program",type=Path,required=True);parser.add_argument("--lifetimes",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);args=parser.parse_args();out=args.output.resolve()
    if out.exists():raise RuntimeError("choose a fresh host lowering audit directory")
    out.mkdir(parents=True);inputs={str(p.resolve()):sha(p) for p in (args.program,args.lifetimes)}
    source_files=[Path(__file__).resolve(),ROOT/"system_sim/physical_host/lowering.py",ROOT/"system_sim/physical_host/control_runtime.h",ROOT/"src/mlxsim/model_control_program.py",ROOT/"src/mlxsim/model_memory_program.py"]
    sources={str(p.relative_to(ROOT)):sha(p) for p in source_files}
    program=json.loads(args.program.read_text());life=json.loads(args.lifetimes.read_text());layouts=collect_layouts(program);bindings={};ids={}
    initial=life["initial"]["allocations"]
    if len(initial)!=len(program["assets"]) or life["source_nodes"]!=len(program["nodes"]) or len(life["events"])!=len(program["nodes"]):raise RuntimeError("lifetime plan does not cover this model")
    for index,(name,allocation) in enumerate(zip(sorted(program["assets"]),initial,strict=True),1):
        layout=layouts[name]
        if allocation["id"]!=index or allocation["bytes"]!=layout["storage_elements"]*BYTES[layout["dtype"]] or allocation["writable"]:raise RuntimeError("initial model bindings differ from lifetime plan")
        bindings[name]={key:allocation[key] for key in ("base","bytes","writable")};ids[name]=allocation["id"]
    live=set(program["assets"]);routes=[]
    for node,event in zip(program["nodes"],life["events"],strict=True):
        if node["source_operator_id"]!=event["source_operator_id"] or node["kind"]!=event["kind"]:raise RuntimeError("lifetime operator ordering changed")
        def references(value):
            if isinstance(value,dict):
                if "value" in value:yield value["value"]
                else:
                    for child in value.values():yield from references(child)
            elif isinstance(value,list):
                for child in value:yield from references(child)
        if any(name not in live for name in references(node["args"])):raise RuntimeError("host lowering references a released SSA value")
        layout=layouts[node["id"]];root=layout["root"];allocation=event["allocation"]
        if allocation["bytes"]!=layout["storage_elements"]*BYTES[layout["dtype"]]:raise RuntimeError("host output byte binding mismatch")
        binding={key:allocation[key] for key in ("base","bytes","writable")}
        if root in bindings:
            if bindings[root]!=binding or ids[root]!=allocation["id"]:raise RuntimeError("host view changed its physical allocation")
        else:bindings[root]=binding;ids[root]=allocation["id"]
        live.add(node["id"])
        if "control_program" in node:
            blob,route=lower_control(node,layouts,bindings);file=out/f"control-{node['source_operator_id']}.bin";file.write_bytes(blob)
            route.update(command_sha256=sha(file),command_file=str(file),forward_id=node["forward_id"],layer_idx=node["layer_idx"]);routes.append(route)
        for name in node["release"]:
            if name not in live:raise RuntimeError("host lowering encountered an invalid release")
            live.remove(name)
    if len(routes)!=sum("control_program" in node for node in program["nodes"]):raise RuntimeError("host control route coverage mismatch")
    if sources!={str(p.relative_to(ROOT)):sha(p) for p in source_files} or any(sha(Path(p))!=digest for p,digest in inputs.items()):raise RuntimeError("host lowering inputs changed")
    report={"classification":"full_model_host_control_abi_lowering_not_model_execution","sources":sources,"inputs":inputs,"source_nodes":len(program["nodes"]),"host_control_routes":routes,
        "control_route_count":len(routes),"command_bytes_total":sum(r["command_bytes"] for r in routes),"actual_model_data_executed":False,"rocket_execution_verified":False,"mlx_system_verified":False,"inference_performance_eligible":False}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(f"HOST_CONTROL_ABI_LOWERING_VERIFIED {out/'report.json'}")


if __name__=="__main__":main()
