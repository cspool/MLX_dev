"""Compile a complete matrix/vector window without executing a simulator."""
import argparse
from collections import Counter
import json
from pathlib import Path

from mlxsim.model_matrix_events import matrix_events
from mlxsim.model_vector_events import vector_events
from mlxsim.model_memory_events import memory_events
from scripts.mlx_system_attempt import digest,record

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--family",choices=("matrix","vector","memory"),default="matrix")
    args=parser.parse_args();out=args.output.resolve()
    if out.exists():raise ValueError("choose a fresh matrix event compilation directory")
    fingerprint=digest(args.job);job=json.loads(args.job.read_text());program={"matrix":matrix_events,"vector":vector_events,"memory":memory_events}[args.family](job);counts=Counter()
    def visit(items,multiplier=1):
        for item in items:
            if "repeat" in item:visit(item["body"],multiplier*item["repeat"])
            else:
                counts[item["op"]+"_events"]+=multiplier
                counts[item["op"]+"_lanes"]+=multiplier*item.get("active_lanes",0)
                counts[item["op"]+"_bytes"]+=multiplier*item.get("bytes",0)
    for block in [*program["blocks"],*program.get("controllers",[])]:visit(block["events"])
    if args.family=="matrix":
        if counts["mul_lanes"]!=job["m"]*job["n"]*job["k"]:raise ValueError("matrix event lowering lost MAC work")
        output_bytes=job["m"]*job["n"]*(2 if job["program"]["output_dtype"]=="f16" else 4)
    else:
        import math
        width={"f16":2,"f32":4,"i64":8,"bool":1}[job["node"]["output"]["dtype"]]
        output_bytes=math.prod(job["node"]["output"]["shape"])*width
        if args.family=="memory" and job["node"]["memory_program"]["mode"]=="view":output_bytes=0
    if counts["dma_write_bytes"]!=output_bytes:raise ValueError("event lowering lost output traffic")
    if digest(args.job)!=fingerprint:raise ValueError("matrix job changed during compilation")
    out.mkdir(parents=True);record(out/"events.json",program)
    sources={name:digest(ROOT/name) for name in ("src/mlxsim/model_matrix_events.py","src/mlxsim/model_matrix_program.py","scripts/compile_mlx_matrix_events.py")}
    if args.family=="vector":
        for name in ("src/mlxsim/model_vector_events.py","src/mlxsim/model_vector_program.py","src/mlxsim/model_dtype_lowering.py"):sources[name]=digest(ROOT/name)
    if args.family=="memory":
        for name in ("src/mlxsim/model_memory_events.py","src/mlxsim/model_memory_program.py"):sources[name]=digest(ROOT/name)
    record(out/"manifest.json",dict(classification=f"complete_{args.family}_window_event_lowering_not_full_model_execution",family=args.family,sources=sources,
        input_job=str(args.job.resolve()),input_sha256=fingerprint,event_program_sha256=digest(out/"events.json"),blocks=len(program["blocks"]),
        declared_work=dict(counts),tensor_values_executed=False,full_model_lowering_complete=False,event_vs_end_to_end_error_available=False))
    print(f"{args.family.upper()}_WINDOW_EVENTS_COMPILED {out/'manifest.json'}")


if __name__=="__main__":main()
