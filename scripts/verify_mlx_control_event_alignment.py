"""Re-execute complete RV64 descriptor control windows and independent event IR."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from mlxsim.model_control_events import control_events
from scripts.mlx_system_attempt import digest,record
from scripts.run_mlx_tensor_semantics import ROOT,source_identity
from scripts.verify_mlx_event_schedule import audit_trace,require


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("tests","native-binary","event-binary","output"):parser.add_argument("--"+key,type=Path,required=True)
    args=parser.parse_args();tests=args.tests.resolve();out=args.output.resolve()
    require(not out.exists(),"choose a fresh control alignment directory")
    jobs=[p for p in (tests/"pytest").rglob("control-event-job.json") if not any(a.is_symlink() for a in p.parents)]
    require(len(jobs)==28,"complete control alignment scope differs")
    sources=source_identity()
    for name in ("src/mlxsim/model_control_events.py","simulator_ext/event_schedule/event_schedule.cc","scripts/verify_mlx_control_event_alignment.py","scripts/verify_mlx_event_schedule.py","tests/control_external_memory.cc"):
        sources[name]=digest(ROOT/name)
    out.mkdir(parents=True);native=out/"control-window";event=out/"event-schedule"
    shutil.copy2(args.native_binary,native);shutil.copy2(args.event_binary,event)
    binaries={str(native):digest(native),str(event):digest(event)};cases=[]
    for i,job in enumerate(sorted(jobs)):
        directory=out/f"case-{i:02d}";directory.mkdir();request=json.loads(job.read_text());native_job=job.parent/"native/job.json"
        data=json.loads(native_job.read_text());require(not data["external"],"control timing endpoint differs")
        require(data["node"]==request["node"] and data.get("options",{})==request.get("options",{}),"control job binding differs")
        layouts={name:dict(shape=data.get("layouts",{}).get(name,{}).get("shape",a["shape"]),dtype=a["dtype"]) for name,a in data["assets"].items()}
        require(layouts==request["input_layouts"],"control event input layout differs")
        bound_files={}
        for name,asset in data["assets"].items():
            require(asset["kind"] in {"literal","mapped_file"},"unregistered control test asset")
            if asset["kind"]=="mapped_file":bound_files[asset["path"]]=digest(Path(asset["path"]))
        program=control_events(request)
        require(program==json.loads((job.parent/"event/program.json").read_text()),"control event compilation changed")
        record(directory/"events.json",program)
        for binary,command,log in ((native,[str(native_job),str(directory/"native")],"native.log"),(event,[str(directory/"events.json"),str(directory/"event.json")],"event.log")):
            with (directory/log).open("w") as stream:subprocess.run([str(binary),*command],stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=120)
        n=json.loads((directory/"native/result.json").read_text());e=json.loads((directory/"event.json").read_text())
        require(n==json.loads((job.parent/"native/out/result.json").read_text()) and digest(directory/"native/output.bin")==digest(job.parent/"native/out/output.bin"),"control numerical execution changed")
        require(e==json.loads((job.parent/"event/result.json").read_text()),"control event execution changed")
        require(n["functional_oracle_equal"] and n["done"] and not n["trace_truncated"],"control native correctness/trace incomplete")
        if request["node"]["kind"]=="argmax":
            selects=[r for r in n["trace"] if r["event"]=="instruction_issue" and r["phase"]=="select"]
            updates={(r["row"],r["column"]) for r in selects if r["pc"]==1}
            choices=[(r["row"],r["column"]) not in updates for r in selects if r["pc"]==0]
            require(choices==request["branch_taken"] and sum(choices)==n["branches_taken"],"argmax branch witness differs from actual PC path")
        if request["node"]["kind"]=="guard":
            value=bool(next(r["data"]&255 for r in n["trace"] if r["event"]=="response" and not r["write"]))
            require(value==request["guard_value"]==request["node"]["args"][1],"actual guard predicate mismatch")
        require(e["block_intervals"][0]["window_cycles"]==n["cycles"],"control cycle mismatch")
        for direction in ("read","write"):require(e["declared_work"].get("dma_"+direction+"_bytes",0)==n[direction+"_bytes"],"control traffic mismatch")
        require(sum(e["declared_work"].get("control_"+op+"_events",0) for op in ("alu","multiply","float","branch"))==n["instructions"],"control instruction work mismatch")
        audit_trace(program,e)
        require(all(digest(Path(path))==sha for path,sha in bound_files.items()),"control asset changed during replay")
        cases.append(dict(job=str(job),job_sha256=digest(job),native_job=str(native_job),native_job_sha256=digest(native_job),input_files=bound_files,
            numerical_output_sha256=digest(directory/"native/output.bin"),native_result_sha256=digest(directory/"native/result.json"),event_result_sha256=digest(directory/"event.json"),
            cycles=n["cycles"],signed_cycle_error=0.0))
    require(all(digest(ROOT/name)==sha for name,sha in sources.items()) and all(digest(Path(name))==sha for name,sha in binaries.items()),"control alignment sources/binaries changed")
    record(out/"report.json",dict(classification="complete_control_window_alignment_not_rocket_or_full_model",family="control",sources=sources,binaries=binaries,cases=cases,
        all_window_cycles_and_work_equal=True,full_model_execution_verified=False,full_model_event_lowering_complete=False,model_performance_error_available=False))
    print(f"CONTROL_EVENT_ALIGNMENT_PASS cases={len(cases)}")


if __name__=="__main__":main()
