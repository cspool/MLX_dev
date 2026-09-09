"""Independent numerical concurrent-window replay and source-order error audit."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from mlxsim.model_array_group_events import array_group_events,native_group_job
from scripts.mlx_system_attempt import digest,record
from scripts.run_mlx_tensor_semantics import ROOT,source_identity
from scripts.verify_mlx_event_schedule import audit_trace,require


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("tests","native-binary","event-binary","output"):parser.add_argument("--"+key,type=Path,required=True)
    args=parser.parse_args();tests=args.tests.resolve();out=args.output.resolve()
    require(not out.exists(),"choose a fresh array group audit")
    jobs=[p for p in (tests/"pytest").rglob("group-job.json") if not any(a.is_symlink() for a in p.parents)]
    require(len(jobs)==32,"array group alignment scope differs")
    sources=source_identity()
    for name in ("src/mlxsim/model_array_group_events.py","src/mlxsim/model_matrix_events.py","src/mlxsim/model_vector_events.py","simulator_ext/event_schedule/event_schedule.cc",
                 "simulator_ext/event_alignment/main.cc","simulator_ext/event_alignment/CMakeLists.txt","scripts/verify_mlx_array_group_alignment.py","scripts/verify_mlx_event_schedule.py"):
        sources[name]=digest(ROOT/name)
    out.mkdir(parents=True);native=out/"native-group";event=out/"event-schedule"
    shutil.copy2(args.native_binary,native);shutil.copy2(args.event_binary,event);binaries={str(p):digest(p) for p in (native,event)};cases=[]
    for i,job in enumerate(sorted(jobs)):
        directory=out/f"case-{i:02d}";directory.mkdir();spec=json.loads(job.read_text());program=array_group_events(spec);bound=native_group_job(spec)
        require(bound==json.loads((job.parent/"native-job.json").read_text()),"native group input changed")
        require(program==json.loads((job.parent/"event/program.json").read_text()),"array group lowering changed")
        record(directory/"native-job.json",bound);record(directory/"events.json",program)
        old=dict(program,source_tick_order=False);record(directory/"legacy-events.json",old)
        for binary,command,log in ((native,[str(directory/"native-job.json"),str(directory/"native")],"native.log"),
                                   (event,[str(directory/"events.json"),str(directory/"event.json")],"event.log"),
                                   (event,[str(directory/"legacy-events.json"),str(directory/"legacy.json")],"legacy.log")):
            with (directory/log).open("w") as stream:subprocess.run([str(binary),*command],stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=120)
        n=json.loads((directory/"native/result.json").read_text());e=json.loads((directory/"event.json").read_text());legacy=json.loads((directory/"legacy.json").read_text())
        require(n==json.loads((job.parent/"native/result.json").read_text()),"numerical shared-array report changed")
        require(e==json.loads((job.parent/"event/result.json").read_text()) and legacy==json.loads((job.parent/"legacy/result.json").read_text()),"event reports changed")
        require(n["cycles"]==e["cycles"],"source tick cycles differ from native")
        for a,b in (("contexts","peak_contexts"),("spm_vectors","peak_spm_vectors"),("rf_vectors_per_pe","peak_rf_vectors_per_pe"),("rom_words_per_pe","peak_rom_words_per_pe")):
            require(e["peak"][a]==n["array"][b],"group resource peak differs")
        output_hashes=[]
        for index,w in enumerate(n["windows"]):
            digest_output=digest(directory/f"native/output-{index}.bin")
            require(digest_output==digest(job.parent/f"native/output-{index}.bin")==digest(job.parent/f"solo{index}/out/output.bin"),"shared-array output differs from complete standalone numeric execution")
            output_hashes.append(digest_output)
            require(max(b["retire_cycle"] for b in e["block_intervals"] if b["source_operator_id"]==index)==w["cycles"],"source completion differs")
        for direction in ("read","write"):
            total=sum(w["numeric_instructions"]["global_"+direction+"_bytes"] if "global_read_bytes" in w["numeric_instructions"] else w["dma_"+direction+"_bytes"] for w in n["windows"])
            require(e["declared_work"]["dma_"+direction+"_bytes"]==total,"group traffic differs")
        audit_trace(program,e);audit_trace(old,legacy)
        cases.append(dict(job=str(job),job_sha256=digest(job),families=["matrix" if w["schema"]=="mlx_matrix_window_job_v1" else "vector" for w in spec["windows"]],
            native_cycles=n["cycles"],event_cycles=e["cycles"],legacy_cycles=legacy["cycles"],signed_cycle_error=0.0,legacy_signed_cycle_error=(legacy["cycles"]-n["cycles"])/n["cycles"],
            numerical_output_sha256=output_hashes,native_result_sha256=digest(directory/"native/result.json"),event_result_sha256=digest(directory/"event.json")))
    require(all(digest(ROOT/p)==h for p,h in sources.items()) and all(digest(Path(p))==h for p,h in binaries.items()),"group audit sources/binaries changed")
    record(out/"report.json",dict(classification="concurrent_numerical_array_groups_not_full_model_error",sources=sources,binaries=binaries,cases=cases,
        all_source_tick_cycles_equal=True,legacy_different_cases=sum(c["legacy_cycles"]!=c["native_cycles"] for c in cases),
        legacy_max_absolute_relative_error=max(abs(c["legacy_signed_cycle_error"]) for c in cases),
        full_model_execution_verified=False,full_model_event_lowering_complete=False,model_performance_error_available=False))
    print(f"ARRAY_GROUP_ALIGNMENT_PASS cases={len(cases)}")


if __name__=="__main__":main()
