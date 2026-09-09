"""Re-execute complete numerical matrix windows and audit timing-only lowering."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from mlxsim.model_matrix_events import matrix_events
from scripts.mlx_system_attempt import digest,record
from scripts.run_mlx_tensor_semantics import ROOT,source_identity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("tests","native-binary","event-binary","output"):parser.add_argument("--"+key,type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();tests=args.tests.resolve()
    if out.exists():raise ValueError("choose a fresh matrix/event alignment directory")
    jobs=[p for p in (tests/"pytest").rglob("job.json") if not any(a.is_symlink() for a in p.parents)]
    if len(jobs)!=22:raise ValueError("native matrix comparison scope differs")
    sources=source_identity()
    for name in ("src/mlxsim/model_matrix_events.py","simulator_ext/event_schedule/event_schedule.cc","scripts/verify_mlx_matrix_event_alignment.py"):
        sources[name]=digest(ROOT/name)
    out.mkdir(parents=True);native=out/"matrix-window";event=out/"event-schedule"
    shutil.copy2(args.native_binary,native);shutil.copy2(args.event_binary,event)
    binaries={str(native):digest(native),str(event):digest(event)};cases=[]
    for i,job_file in enumerate(sorted(jobs)):
        job=json.loads(job_file.read_text());directory=out/f"case-{i:02d}";directory.mkdir()
        original_event=job_file.parent.parent/"event/program.json"
        lowered=matrix_events(job)
        if lowered!=json.loads(original_event.read_text()):raise ValueError("matrix lowering changed after tests")
        record(directory/"events.json",lowered)
        with (directory/"native.log").open("w") as log:
            subprocess.run([str(native),str(job_file),str(directory/"native")],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
        with (directory/"event.log").open("w") as log:
            subprocess.run([str(event),str(directory/"events.json"),str(directory/"event.json")],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
        n=json.loads((directory/"native/result.json").read_text());e=json.loads((directory/"event.json").read_text())
        if n!=json.loads((job_file.parent/"out/result.json").read_text()) or digest(directory/"native/output.bin")!=digest(job_file.parent/"out/output.bin"):
            raise ValueError("native numerical execution changed")
        if e!=json.loads((original_event.parent/"result.json").read_text()):raise ValueError("event execution changed")
        if n["cycles"]!=e["cycles"] or n["dma_read_bytes"]!=e["declared_work"].get("dma_read_bytes",0) or n["dma_write_bytes"]!=e["declared_work"]["dma_write_bytes"]:
            raise ValueError("matrix/event cycle or traffic mismatch")
        for a,b in (("compute_busy_pe_cycles","compute_inflight_pe_cycles"),("spm_busy_cycles","spm_inflight_cycles"),("dma_busy_cycles","dma_inflight_cycles"),("compute_dma_overlap_pe_cycles","compute_dma_inflight_overlap_pe_cycles")):
            if n[a]!=e["integrated_usage"].get(b,0):raise ValueError("matrix/event occupancy mismatch")
        cases.append(dict(job=str(job_file),job_sha256=digest(job_file),events_sha256=digest(directory/"events.json"),
            numerical_output_sha256=digest(directory/"native/output.bin"),native_result_sha256=digest(directory/"native/result.json"),
            event_result_sha256=digest(directory/"event.json"),cycles=n["cycles"],signed_cycle_error=0.0))
    if any(digest(ROOT/p)!=v for p,v in sources.items()) or any(digest(Path(p))!=v for p,v in binaries.items()):raise ValueError("alignment source/binary changed")
    record(out/"report.json",dict(classification="complete_matrix_window_alignment_not_full_model_error",sources=sources,binaries=binaries,cases=cases,
        all_window_cycles_and_work_equal=True,full_model_execution_verified=False,full_model_event_lowering_complete=False,model_performance_error_available=False))
    print(f"MATRIX_EVENT_ALIGNMENT_PASS cases={len(cases)}")


if __name__=="__main__":main()
