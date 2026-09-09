"""Re-execute complete numerical matrix/vector windows and timing-only lowering."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from mlxsim.model_matrix_events import matrix_events
from mlxsim.model_vector_events import vector_events
from mlxsim.model_memory_events import memory_events
from scripts.mlx_system_attempt import digest,record
from scripts.run_mlx_tensor_semantics import ROOT,source_identity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("tests","native-binary","event-binary","output"):parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--family",choices=("matrix","vector","memory"),default="matrix")
    args=parser.parse_args();out=args.output.resolve();tests=args.tests.resolve()
    if out.exists():raise ValueError("choose a fresh matrix/event alignment directory")
    jobs=[p for p in (tests/"pytest").rglob("memory-event-job.json" if args.family=="memory" else "job.json") if not any(a.is_symlink() for a in p.parents)
          and json.loads(p.read_text()).get("schema")==("mlx_memory_event_window_job_v1" if args.family=="memory" else f"mlx_{args.family}_window_job_v1")]
    if len(jobs)!={"matrix":22,"vector":48,"memory":18}[args.family]:raise ValueError("native window comparison scope differs")
    sources=source_identity()
    for name in ("src/mlxsim/model_matrix_events.py","simulator_ext/event_schedule/event_schedule.cc","scripts/verify_mlx_matrix_event_alignment.py"):
        sources[name]=digest(ROOT/name)
    if args.family=="vector":sources["src/mlxsim/model_vector_events.py"]=digest(ROOT/"src/mlxsim/model_vector_events.py")
    if args.family=="memory":sources["src/mlxsim/model_memory_events.py"]=digest(ROOT/"src/mlxsim/model_memory_events.py")
    out.mkdir(parents=True);native=out/(args.family+"-window");event=out/"event-schedule"
    shutil.copy2(args.native_binary,native);shutil.copy2(args.event_binary,event)
    binaries={str(native):digest(native),str(event):digest(event)};cases=[]
    for i,job_file in enumerate(sorted(jobs)):
        job=json.loads(job_file.read_text());directory=out/f"case-{i:02d}";directory.mkdir()
        original_event=(job_file.parent if args.family=="memory" else job_file.parent.parent)/"event/program.json"
        lowered={"matrix":matrix_events,"vector":vector_events,"memory":memory_events}[args.family](job)
        if lowered!=json.loads(original_event.read_text()):raise ValueError("matrix lowering changed after tests")
        record(directory/"events.json",lowered)
        with (directory/"native.log").open("w") as log:
            native_job=job_file.parent/"native/job.json" if args.family=="memory" else job_file
            subprocess.run([str(native),str(native_job),str(directory/"native")],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
        with (directory/"event.log").open("w") as log:
            subprocess.run([str(event),str(directory/"events.json"),str(directory/"event.json")],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
        n=json.loads((directory/"native/result.json").read_text());e=json.loads((directory/"event.json").read_text())
        original_native=(job_file.parent/"native/out") if args.family=="memory" else (job_file.parent/"out")
        if n!=json.loads((original_native/"result.json").read_text()) or digest(directory/"native/output.bin")!=digest(original_native/"output.bin"):
            raise ValueError("native numerical execution changed")
        if e!=json.loads((original_event.parent/"result.json").read_text()):raise ValueError("event execution changed")
        if args.family=="memory":
            n=n["windows"][-1]
            if job.get("predicate_choices") is not None:
                choices=[bool(row["data"]) for row in n["trace"] if row["event"]=="response" and (job["node"]["memory_program"]["words"][row["pc"]]&255)==5]
                if choices!=job["predicate_choices"]:raise ValueError("predicate choice witness differs from actual execution data")
        read_bytes=n["numeric_instructions"]["global_read_bytes"] if args.family=="vector" else n["dma_read_bytes"]
        write_bytes=n["numeric_instructions"]["global_write_bytes"] if args.family=="vector" else n["dma_write_bytes"]
        event_cycles=e["block_intervals"][0]["window_cycles"] if args.family=="memory" else e["cycles"]
        event_read=sum(e["declared_work"].get(op+"_bytes",0) for op in (("dma_read","memory_index_read","memory_predicate_read") if args.family=="memory" else ("dma_read",)))
        if n["cycles"]!=event_cycles or read_bytes!=event_read or write_bytes!=e["declared_work"].get("dma_write_bytes",0):
            raise ValueError("matrix/event cycle or traffic mismatch")
        counters=[] if args.family=="memory" else [("spm_busy_cycles","spm_inflight_cycles"),("dma_busy_cycles","dma_inflight_cycles")]
        if args.family!="memory":counters+=([("vector_busy_pe_cycles","compute_inflight_pe_cycles"),("trans_busy_pe_cycles","sfu_inflight_pe_cycles"),("vector_sfu_overlap_pe_cycles","compute_sfu_inflight_overlap_pe_cycles"),("same_pe_context_overlap_cycles","same_pe_inflight_context_overlap_pe_cycles")] if args.family=="vector" else [("compute_busy_pe_cycles","compute_inflight_pe_cycles"),("compute_dma_overlap_pe_cycles","compute_dma_inflight_overlap_pe_cycles")])
        for a,b in counters:
            if n[a]!=e["integrated_usage"].get(b,0):raise ValueError("matrix/event occupancy mismatch")
        cases.append(dict(job=str(job_file),job_sha256=digest(job_file),native_job=str(native_job),native_job_sha256=digest(native_job),events_sha256=digest(directory/"events.json"),
            numerical_output_sha256=digest(directory/"native/output.bin"),native_result_sha256=digest(directory/"native/result.json"),
            event_result_sha256=digest(directory/"event.json"),cycles=n["cycles"],signed_cycle_error=0.0))
    if any(digest(ROOT/p)!=v for p,v in sources.items()) or any(digest(Path(p))!=v for p,v in binaries.items()):raise ValueError("alignment source/binary changed")
    record(out/"report.json",dict(classification=f"complete_{args.family}_window_alignment_not_full_model_error",family=args.family,sources=sources,binaries=binaries,cases=cases,
        all_window_cycles_and_work_equal=True,full_model_execution_verified=False,full_model_event_lowering_complete=False,model_performance_error_available=False))
    print(f"{args.family.upper()}_EVENT_ALIGNMENT_PASS cases={len(cases)}")


if __name__=="__main__":main()
