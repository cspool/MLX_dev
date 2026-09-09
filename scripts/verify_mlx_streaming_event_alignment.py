"""Re-run actual numerical closed pairs and independently rebuild event inputs."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess

from mlxsim.model_streaming_pair_events import streaming_pair_events
from scripts.run_mlx_tensor_semantics import ROOT,source_identity
from scripts.mlx_system_attempt import digest,record,linked_libraries
from scripts.verify_mlx_event_schedule import require,audit_trace


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("tests","native-binary","event-binary","output"):parser.add_argument("--"+key,type=Path,required=True)
    args=parser.parse_args();tests=args.tests.resolve();out=args.output.resolve();require(not out.exists(),"choose a fresh streaming alignment directory")
    jobs=[p for p in (tests/"pytest").glob("*/native/program.json") if not any(a.is_symlink() for a in p.parents) and (p.parent.parent/"event/program.json").exists() and "streaming_pairs" in json.loads((p.parent.parent/"event/program.json").read_text())]
    require(len(jobs)==25,"streaming numerical coverage differs")
    sources=source_identity()
    for name in ("simulator_ext/event_schedule/event_schedule.cc","simulator_ext/event_schedule/pattern_store.h","simulator_ext/event_schedule/CMakeLists.txt","src/mlxsim/model_streaming_pair_events.py",
                 "src/mlxsim/model_event_windows.py","src/mlxsim/model_array_graph_events.py","src/mlxsim/model_array_group_events.py","src/mlxsim/model_lazy_event_patterns.py","src/mlxsim/model_compact_event_streams.py",
                 "scripts/verify_mlx_streaming_event_alignment.py","scripts/verify_mlx_event_schedule.py"):
        sources[name]=digest(ROOT/name)
    out.mkdir(parents=True);native=out/"native-ready-graph";event=out/"event-schedule";shutil.copy2(args.native_binary,native);shutil.copy2(args.event_binary,event)
    binaries={str(p):digest(p) for p in (native,event)};libraries={**linked_libraries(native),**linked_libraries(event)};cases=[]
    for index,job in enumerate(sorted(jobs)):
        directory=out/f"case-{index:02d}";directory.mkdir();program=json.loads(job.read_text());options=json.loads((job.parent/"options.json").read_text())
        require(all(a["kind"]=="literal" for a in program["assets"].values()),"streaming replay fixture has external unbound assets")
        p=streaming_pair_events(program,directory/"patterns",timed_templates=options["template_load_timing"],memory=options["memory"])
        old=json.loads((job.parent.parent/"event/program.json").read_text());projected=copy.deepcopy(p)
        for key,spec in projected["event_patterns"].items():
            require(digest(Path(old["event_patterns"][key]["path"]))==spec["sha256"],"original streaming pattern changed");spec["path"]=old["event_patterns"][key]["path"]
        require(projected==old,"streaming event compilation changed")
        record(directory/"program.json",program);record(directory/"options.json",options);record(directory/"events.json",p)
        for binary,command,log in ((native,[str(directory/"program.json"),str(directory/"options.json"),str(directory/"native")],"native.log"),(event,[str(directory/"events.json"),str(directory/"event.json")],"event.log")):
            with (directory/log).open("w") as stream:subprocess.run([str(binary),*command],stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=180)
        n=json.loads((directory/"native/result.json").read_text());e=json.loads((directory/"event.json").read_text());baseline=json.loads((job.parent/"out/result.json").read_text());normalized=copy.deepcopy(n);hashes=[]
        for new,prior in zip(normalized["outputs"],baseline["outputs"],strict=True):
            require(digest(Path(new["logits_file"]))==digest(Path(prior["logits_file"])),"streaming numerical output changed");hashes.append(digest(Path(new["logits_file"])));new["logits_file"]=prior["logits_file"]
        require(normalized==baseline and e==json.loads((job.parent.parent/"event/result.json").read_text()),"streaming runtime report changed")
        require(e["cycles"]==n["pipeline_groups"][0]["end_cycle"]-n["pipeline_groups"][0]["begin_cycle"],"streaming pair cycles differ")
        ng,eg=n["pipeline_groups"][0],e["pipeline_groups"][0]
        for field in ("blocks","frontier","admitted","completed","event_slots","peak_event_slots","finished","active","pending_visibility","out_of_order_done"):
            require(ng[field]==eg[field],"streaming event bank differs from numerical execution")
        audit_trace(p,e)
        cases.append(dict(program=str(job),program_sha256=digest(job),options_sha256=digest(job.parent/"options.json"),cycles=e["cycles"],producer_kind=program["nodes"][0]["kind"],
            event_slots=eg["event_slots"],output_sha256=hashes,native_result_sha256=digest(directory/"native/result.json"),event_result_sha256=digest(directory/"event.json"),signed_cycle_error=0.0))
    require(all(digest(ROOT/p)==h for p,h in sources.items()) and all(digest(Path(p))==h for p,h in {**binaries,**libraries}.items()),"streaming replay provenance changed")
    record(out/"report.json",dict(classification="actual_numerical_streaming_pair_alignment_not_full_model",sources=sources,binaries=binaries,runtime_libraries=libraries,cases=cases,
        all_pair_cycles_and_events_equal=True,full_model_execution_verified=False,model_performance_error_available=False))
    print("STREAMING_PAIR_ALIGNMENT_PASS cases=25")


if __name__=="__main__":main()
