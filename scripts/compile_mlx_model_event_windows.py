"""Compile every source/batch pattern while preserving its real layout binding."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from mlxsim.model_event_windows import WindowCatalog,MissingWitness
from scripts.mlx_system_attempt import digest,record
from scripts.run_mlx_event_witness import decode_boolean_witness
from scripts.verify_mlx_event_schedule import require


def encoded(value):return json.dumps(value,sort_keys=True,separators=(",",":"))


def work(program):
    counts=Counter();leaves=nodes=0
    def walk(items,multiplier):
        nonlocal leaves,nodes
        for item in items:
            nodes+=1
            if "repeat" in item:walk(item["body"],multiplier*item["repeat"])
            else:
                leaves+=1;counts[item["op"]+"_events"]+=multiplier;counts[item["op"]+"_bytes"]+=multiplier*item.get("bytes",0);counts[item["op"]+"_lanes"]+=multiplier*item.get("active_lanes",0)
    for block in program["blocks"]+program.get("controllers",[]):walk(block["events"],1)
    return dict(blocks=len(program["blocks"]),controllers=len(program.get("controllers",[])),stored_leaves=leaves,stored_nodes=nodes,declared_work=dict(counts))


def load_witnesses(path,target_hash):
    bundle=json.loads(path.read_text());require(bundle["target_program_sha256"]==target_hash and bundle["full_numerical_execution_equal"],"witness target/numerical binding differs")
    native=path.parent/"native/result.json";state=json.loads((path.parent/"execution.json").read_text())
    require(state["status"]=="exited" and state["exit_code"]==0 and digest(native)==bundle["native_report_sha256"],"witness execution is not complete")
    files={str(path.resolve()):digest(path),str(native):digest(native),str(path.parent/"execution.json"):digest(path.parent/"execution.json")};values={}
    observed={r["source_operator_id"]:r for r in json.loads(native.read_text())["observations"]}
    for row in bundle["witnesses"]:
        p=Path(row["observed_file"]);require(digest(p)==row["observed_sha256"] and observed[row["producer"]]["file"]==str(p),"witness raw observation changed")
        require(row["source_operator_id"] not in values and decode_boolean_witness(row,p.read_bytes())==row["value_data"],"witness Boolean data differs")
        values[row["source_operator_id"]]=row["value_data"];files[str(p)]=digest(p)
    return values,files


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);parser.add_argument("--witnesses",type=Path)
    args=parser.parse_args();out=args.output.resolve();require(not out.exists(),"choose a fresh complete window compilation")
    program_hash=digest(args.program);catalog=WindowCatalog(json.loads(args.program.read_text()));witnesses={};witness_files={}
    if args.witnesses:witnesses,witness_files=load_witnesses(args.witnesses.resolve(),program_hash)
    root=Path(__file__).resolve().parents[1]
    names=["src/mlxsim/model_event_windows.py","src/mlxsim/model_event_graph_plan.py","src/mlxsim/model_memory_program.py","src/mlxsim/model_matrix_events.py","src/mlxsim/model_vector_events.py","src/mlxsim/model_memory_events.py","src/mlxsim/model_control_events.py","scripts/compile_mlx_model_event_windows.py","scripts/run_mlx_event_witness.py"]
    sources={name:digest(root/name) for name in names};out.mkdir(parents=True);(out/"patterns").mkdir();patterns={};rows=[];blocked=[];counts=Counter()
    for row in catalog.plan["sources"]:
        source=row["source_operator_id"]
        for batch in range(row["window_count"]):
            try:job,binding=catalog.window(source,batch,witnesses.get(source))
            except MissingWitness as error:blocked.append(dict(source_operator_id=source,batch_index=batch,reason=str(error)));continue
            key=hashlib.sha256(encoded(job).encode()).hexdigest()
            if key not in patterns:
                events=catalog.compile(job);summary=work(events);path=out/"patterns"/(key+".json");record(path,events)
                record(out/"patterns"/(key+"-job.json"),job)
                patterns[key]=dict(**summary,event_file=str(path),event_sha256=digest(path),job_sha256=key)
            counts.update(patterns[key]["declared_work"]);rows.append(dict(binding=binding,pattern=key))
        if len(rows)%100==0:print(f"WINDOW_COMPILE source={source} compiled={len(rows)} patterns={len(patterns)}",flush=True)
    require(digest(args.program)==program_hash and all(digest(root/name)==sha for name,sha in sources.items()) and all(digest(Path(name))==sha for name,sha in witness_files.items()),"window compilation inputs changed")
    require(len(rows)+len(blocked)==catalog.plan["total_windows"],"complete model window coverage changed")
    record(out/"manifest.json",dict(classification="complete_model_window_compilation_not_graph_execution",sources=sources,program_path=str(args.program.resolve()),program_file_sha256=program_hash,
        original_source_calls=catalog.plan["source_calls"],lowered_sources=catalog.plan["lowered_calls"],expected_windows=catalog.plan["total_windows"],compiled_windows=len(rows),blocked_windows=blocked,
        patterns=patterns,windows=rows,declared_work=dict(counts),witness_files=witness_files,all_window_patterns_compiled=not blocked,
        tensor_values_executed=False,full_event_lowering_complete=False,event_simulated_cycles=None,model_performance_error_available=False,
        remaining=["lazy_CPP_graph_loading","instance_namespace_and_resource_binding","actual_address_stream_execution","streaming_pair_events","physical_memory_and_template_timing"]))
    print(f"MODEL_WINDOWS_COMPILED windows={len(rows)} blocked={len(blocked)} patterns={len(patterns)}",flush=True)


if __name__=="__main__":main()
