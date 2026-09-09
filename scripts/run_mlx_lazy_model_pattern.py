"""Execute a full-shape compiled model window before/after lazy block loading."""
import argparse
import copy
import json
from pathlib import Path
import shutil

from mlxsim.model_lazy_event_patterns import externalize,materialize
from mlxsim.model_compact_event_streams import compact_streams
from scripts.mlx_system_attempt import digest,record,run_process,linked_libraries
from scripts.verify_mlx_event_schedule import require,audit_trace


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled",type=Path,required=True);parser.add_argument("--binary",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--compact",action="store_true")
    args=parser.parse_args();base=args.compiled.resolve();out=args.output.resolve();require(not out.exists(),"choose fresh full-shape pattern replay")
    m=json.loads((base/"manifest.json").read_text());candidates=[]
    for key,row in m["patterns"].items():
        job=json.loads((base/"patterns"/(key+"-job.json")).read_text())
        if job["schema"]=="mlx_vector_window_job_v1":candidates.append((row["stored_leaves"],key,job))
    require(candidates,"compiled model has no vector patterns");_,key,job=max(candidates,key=lambda x:(x[0],x[1]));path=base/"patterns"/(key+".json")
    require(digest(path)==m["patterns"][key]["event_sha256"],"compiled model pattern changed")
    original=json.loads(path.read_text());require(original["schema"]=="mlx_event_schedule_v4" and not original["hardware"]["template_load_timing"],"unexpected vector pattern contract")
    p=copy.deepcopy(original);p.update(schema="mlx_event_schedule_v6",source_tick_order=True,controllers=[])
    h=p["hardware"];h["latencies"].update(memory_literal=0,memory_convert=2,memory_complete=0,memory_index_read=h["latencies"]["dma_read"],memory_predicate_read=h["latencies"]["dma_read"],
        control_alu=1,control_multiply=3,control_float=3,control_branch=1,control_literal=0,control_complete=0)
    h["memory_controller"]=dict(data_register_bytes=32,staging_bytes=128,conversion_result_latch_bytes=8,request_data_latch_bytes=8,max_active=1,request_period=1,response_period=1)
    h["control_controller"]=dict(register_bytes=512,rom_words=32,max_active=1,request_period=1,response_period=1)
    if args.compact:p["source_graph"]=[dict(source_operator_id=0,family="vector",parents=[],windows=[[b["id"] for b in p["blocks"]]])]
    out.mkdir(parents=True);lazy=externalize(p,out/"patterns");eager=materialize(lazy)
    record(out/"original.json",original);record(out/"eager.json",eager);record(out/"lazy.json",lazy);record(out/"window-job.json",job)
    if args.compact:record(out/"compact.json",compact_streams(lazy,0))
    binary=out/"event-schedule";shutil.copy2(args.binary,binary);sha=digest(binary);libraries=linked_libraries(binary)
    root=Path(__file__).resolve().parents[1];sources={str(f.relative_to(root)):digest(f) for f in (root/"simulator_ext/event_schedule").iterdir() if f.is_file()}
    for name in ("src/mlxsim/model_lazy_event_patterns.py","scripts/run_mlx_lazy_model_pattern.py"):sources[name]=digest(root/name)
    if args.compact:sources["src/mlxsim/model_compact_event_streams.py"]=digest(root/"src/mlxsim/model_compact_event_streams.py")
    for mode in (("original","eager","lazy","compact") if args.compact else ("original","eager","lazy")):
        run_process([str(binary),str(out/(mode+".json")),str(out/(mode+"-result.json"))],out/(mode+".log"),out/(mode+"-execution.json"),timeout=600,
                    metadata=dict(mode="full_shape_pattern_"+mode,binary_sha256=sha,runtime_libraries=libraries,sources=sources,inputs={str(out/(mode+".json")):digest(out/(mode+".json")),str(path):digest(path)}))
    baseline=json.loads((out/"original-result.json").read_text());expected=json.loads((out/"eager-result.json").read_text());actual=json.loads((out/"lazy-result.json").read_text());extra=actual.pop("lazy_patterns")
    require(actual==expected,"lazy full-shape event result differs from eager")
    for field in ("cycles","events","blocks","counts","integrated_usage","declared_work","peak","capacity"):
        require(expected[field]==baseline[field],"schema normalization changed original pattern execution")
    audit_trace(lazy,json.loads((out/"lazy-result.json").read_text()))
    compact_result=None
    if args.compact:
        compact_result=json.loads((out/"compact-result.json").read_text())
        for field in ("cycles","events","blocks","counts","integrated_usage","declared_work","peak","capacity","source_intervals"):
            require(compact_result[field]==expected[field],"compact block generation changed execution")
        audit_trace(json.loads((out/"compact.json").read_text()),compact_result)
    require(digest(binary)==sha and all(digest(root/name)==value for name,value in sources.items()) and all(digest(Path(name))==value for name,value in libraries.items()),"full-shape replay provenance changed")
    record(out/"report.json",dict(classification="full_shape_model_window_timing_replay_not_full_model_or_numerical_execution",pattern_key=key,compiled_manifest_sha256=digest(base/"manifest.json"),
        original_pattern_sha256=digest(path),source_model_program_sha256=m["program_file_sha256"],kind=job["node"]["kind"],output=job["node"]["output"],cycles=actual["cycles"],events=actual["events"],
        blocks=actual["blocks"],lazy_patterns=extra,compact_blocks=compact_result["compact_blocks"] if compact_result else None,compact_result_equal=True if compact_result else None,
        full_result_equal=True,sources=sources,model_performance_error_available=False,full_model_execution_verified=False))
    print(f"FULL_SHAPE_LAZY_PATTERN_PASS cycles={actual['cycles']} events={actual['events']}",flush=True)


if __name__=="__main__":main()
