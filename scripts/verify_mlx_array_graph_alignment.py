"""Replay real dependent array graphs and source-lifecycle event execution."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from mlxsim.model_array_graph_events import array_graph_events,native_array_graph_job
from scripts.mlx_system_attempt import digest,record
from scripts.run_mlx_tensor_semantics import ROOT,source_identity
from scripts.verify_mlx_event_schedule import audit_trace,require


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("tests","native-binary","event-binary","output"):parser.add_argument("--"+key,type=Path,required=True)
    args=parser.parse_args();tests=args.tests.resolve();out=args.output.resolve();require(not out.exists(),"choose fresh graph alignment output")
    jobs=[p for p in (tests/"pytest").rglob("graph-job.json") if not any(a.is_symlink() for a in p.parents)];require(len(jobs)==8,"graph numerical coverage differs")
    sources=source_identity()
    for name in ("src/mlxsim/model_array_graph_events.py","src/mlxsim/model_array_group_events.py","src/mlxsim/model_matrix_events.py","src/mlxsim/model_vector_events.py",
                 "simulator_ext/event_schedule/event_schedule.cc","simulator_ext/event_alignment/main.cc","simulator_ext/event_alignment/CMakeLists.txt",
                 "scripts/verify_mlx_array_graph_alignment.py","scripts/verify_mlx_event_schedule.py"):
        sources[name]=digest(ROOT/name)
    out.mkdir(parents=True);native=out/"native-graph";event=out/"event-schedule";shutil.copy2(args.native_binary,native);shutil.copy2(args.event_binary,event)
    binaries={str(p):digest(p) for p in (native,event)};cases=[]
    for index,job in enumerate(sorted(jobs)):
        directory=out/f"case-{index:02d}";directory.mkdir();spec=json.loads(job.read_text());p=array_graph_events(spec);njob=native_array_graph_job(spec)
        require(p==json.loads((job.parent/"event/program.json").read_text()) and njob==json.loads((job.parent/"native-job.json").read_text()),"graph compilation changed")
        record(directory/"events.json",p);record(directory/"native-job.json",njob)
        for binary,command,log in ((native,[str(directory/"native-job.json"),str(directory/"native")],"native.log"),(event,[str(directory/"events.json"),str(directory/"event.json")],"event.log")):
            with (directory/log).open("w") as stream:subprocess.run([str(binary),*command],stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=120)
        n=json.loads((directory/"native/result.json").read_text());e=json.loads((directory/"event.json").read_text())
        require(n==json.loads((job.parent/"native/result.json").read_text()) and e==json.loads((job.parent/"event/result.json").read_text()),"native/event graph replay changed")
        require(n["cycles"]==e["cycles"] and n["source_frontend_peak"]==e["source_frontend_peak"],"graph cycles/frontend peak mismatch")
        require([{k:v for k,v in r.items() if k!="family"} for r in e["source_intervals"]]==n["source_intervals"],"source lifetime/batches differ")
        hashes=[]
        for i in range(len(spec["windows"])):
            sha=digest(directory/f"native/output-{i}.bin")
            require(sha==digest(job.parent/f"native/output-{i}.bin")==digest(job.parent/f"solo{i}/out/output.bin"),"real graph output differs from standalone numerical reference")
            hashes.append(sha)
        audit_trace(p,e)
        cases.append(dict(job=str(job),job_sha256=digest(job),source_count=len(spec["sources"]),window_count=len(spec["windows"]),cycles=n["cycles"],signed_cycle_error=0.0,
            numerical_output_sha256=hashes,native_result_sha256=digest(directory/"native/result.json"),event_result_sha256=digest(directory/"event.json")))
    require(all(digest(ROOT/p)==h for p,h in sources.items()) and all(digest(Path(p))==h for p,h in binaries.items()),"graph sources/binaries changed")
    record(out/"report.json",dict(classification="numerical_array_dependency_graphs_not_full_models",sources=sources,binaries=binaries,cases=cases,all_source_cycles_and_publication_equal=True,
        full_model_execution_verified=False,full_model_event_lowering_complete=False,model_performance_error_available=False))
    print("ARRAY_GRAPH_ALIGNMENT_PASS cases=8")


if __name__=="__main__":main()
