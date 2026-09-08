"""Join actual control/matrix/vector/memory wire audits without widening scope."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--program",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    for name in ("matrix","vector","memory","control"):parser.add_argument(f"--{name}-report",type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if out.exists():raise RuntimeError("choose a fresh joint lowering report")
    program=json.loads(args.program.read_text());program_path=str(args.program.resolve());program_hash=sha(args.program);reports={};inputs={program_path:program_hash};groups={};sources={}
    for name in ("matrix","vector","memory","control"):
        path=getattr(args,name+"_report");report=json.loads(path.read_text());reports[name]=report;inputs[str(path.resolve())]=sha(path)
        if report["inputs"].get(program_path)!=program_hash:raise RuntimeError("wire audits do not bind the same model program")
        for source,digest in report["sources"].items():
            if sha(ROOT/source)!=digest or source in sources and sources[source]!=digest:raise RuntimeError("wire audit sources changed or conflict")
            sources[source]=digest
        groups[name]={}
        for route in report["host_control_routes"] if name=="control" else report["routes"]:
            if sha(Path(route["command_file"]))!=route["command_sha256"]:raise RuntimeError("compiled wire command changed")
            groups[name].setdefault(route["source_operator_id"],[]).append(route)
    seen=set();rows=[];validated_bytes=executable_bytes=validated_commands=executable_commands=0;counts={name:0 for name in groups}
    for node in program["nodes"]:
        identifier=node["source_operator_id"]
        if identifier in seen:raise RuntimeError("model source operator IDs repeat")
        seen.add(identifier);kinds=[name for name in groups if name+"_program" in node]
        if len(kinds)!=1:raise RuntimeError("source operator lacks exactly one executable lowering family")
        kind=kinds[0];routes=groups[kind].pop(identifier,[]);expected=math.prod(node["output"]["shape"][:-2]) if node["kind"]=="matmul" else 1
        if len(routes)!=expected or any(r["kind"]!=node["kind"] for r in routes):raise RuntimeError("wire route count/kind differs from source")
        if kind=="matrix" and [r["output_batch"] for r in routes]!=list(range(expected)):raise RuntimeError("matrix wire batch route is incomplete")
        view=kind=="memory" and node["memory_program"]["mode"]=="view"
        bytes=sum(r["command_bytes"] for r in routes);validated_bytes+=bytes;validated_commands+=len(routes)
        if not view:executable_bytes+=bytes;executable_commands+=len(routes)
        counts[kind]+=1;rows.append({"source_operator_id":identifier,"kind":node["kind"],"family":kind,"forward_id":node["forward_id"],"layer_idx":node["layer_idx"],"path":"checked_layout_elision" if view else "host_control_abi" if kind=="control" else kind+"_wire",
            "validated_descriptor_count":len(routes),"execution_descriptor_required":not view,"command_sha256":[r["command_sha256"] for r in routes]})
    if any(group for group in groups.values()):raise RuntimeError("wire audit includes sources absent from model")
    result={"classification":"joint_full_model_source_to_wire_coverage_not_execution","inputs":inputs,"sources":sources,"source_calls":len(rows),"family_source_calls":counts,"routes":rows,
        "validated_descriptors":validated_commands,"validated_descriptor_bytes":validated_bytes,"nonview_execution_descriptors":executable_commands,"nonview_descriptor_bytes":executable_bytes,
        "all_source_calls_have_unique_route":True,"model_data_executed":False,"mlx_system_verified":False,"inference_performance_eligible":False}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+"\n");print(f"JOINT_MODEL_WIRE_COVERAGE_PASS {out}")


if __name__=="__main__":main()
