"""Rebuild full-model event patterns, layout bindings and declared work counts."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from mlxsim.model_event_windows import WindowCatalog,MissingWitness
from scripts.compile_mlx_model_event_windows import encoded,work,load_witnesses
from scripts.mlx_system_attempt import digest,record
from scripts.verify_mlx_event_schedule import require


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);parser.add_argument("--witnesses",type=Path)
    args=parser.parse_args();base=args.compiled.resolve();path=base/"manifest.json";require(not args.output.exists(),"choose a fresh complete window audit")
    before=digest(path);m=json.loads(path.read_text());program=Path(m["program_path"]);require(digest(program)==m["program_file_sha256"],"compiled target program changed")
    root=Path(__file__).resolve().parents[1];require(all(digest(root/name)==sha for name,sha in m["sources"].items()),"compiled window sources changed")
    c=WindowCatalog(json.loads(program.read_text()));values={};files={}
    if args.witnesses:values,files=load_witnesses(args.witnesses.resolve(),m["program_file_sha256"])
    require(files==m["witness_files"],"window witness provenance differs")
    rows={(r["binding"]["source_operator_id"],r["binding"]["batch_index"]):r for r in m["windows"]}
    require(len(rows)==len(m["windows"]),"duplicated source/batch binding")
    checked={};blocked=[];counts=Counter();matrix_macs=0;verified=0
    for source in c.plan["sources"]:
        for batch in range(source["window_count"]):
            try:job,binding=c.window(source["source_operator_id"],batch,values.get(source["source_operator_id"]))
            except MissingWitness as error:blocked.append(dict(source_operator_id=source["source_operator_id"],batch_index=batch,reason=str(error)));continue
            pair=source["source_operator_id"],batch;require(pair in rows,"compiled window omitted")
            row=rows[pair];require(row["binding"]==binding,"real tensor layout/source/batch binding changed")
            key=hashlib.sha256(encoded(job).encode()).hexdigest();require(row["pattern"]==key and key in m["patterns"],"timing pattern belongs to a different job")
            if key not in checked:
                p=base/"patterns"/(key+".json");j=base/"patterns"/(key+"-job.json");meta=m["patterns"][key]
                require(Path(meta["event_file"]).resolve()==p and digest(p)==meta["event_sha256"] and json.loads(j.read_text())==job,"pattern file/job changed")
                fresh=c.compile(job);require(json.loads(p.read_text())==fresh,"full event pattern does not reproduce")
                summary=work(fresh);require(all(meta[k]==v for k,v in summary.items()),"event work/IR counts differ")
                checked[key]=dict(**summary,event_file_sha256=digest(p),job_file_sha256=digest(j))
            summary=checked[key];counts.update(summary["declared_work"]);verified+=1
            if source["family"]=="matrix":
                required=binding["matrix"]["m"]*binding["matrix"]["n"]*binding["matrix"]["k"]
                require(summary["declared_work"].get("mul_lanes",0)==required,"matrix pattern omitted MAC work");matrix_macs+=required
    require(blocked==m["blocked_windows"] and verified==len(rows)==m["compiled_windows"] and verified+len(blocked)==c.plan["total_windows"]==m["expected_windows"],"full model window coverage differs")
    require(dict(counts)==m["declared_work"] and set(checked)==set(m["patterns"]) and m["all_window_patterns_compiled"]==(not blocked),"compiled work or eligibility changed")
    native_macs=None
    if args.witnesses:
        native=json.loads((args.witnesses.resolve().parent/"native/result.json").read_text());native_macs=native["matrix_macs"]
        require(matrix_macs==native_macs and native["executed_source_calls"]==c.plan["source_calls"] and native.get("executed_lowered_calls",native["executed_source_calls"])==c.plan["lowered_calls"],"complete numerical witness run and event compilation work differ")
    require(digest(path)==before and digest(program)==m["program_file_sha256"] and all(digest(Path(name))==sha for name,sha in files.items()),"window audit inputs changed")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    record(args.output,dict(classification="full_model_pattern_rebuild_audit_not_event_execution",manifest=str(path),manifest_sha256=before,program_sha256=m["program_file_sha256"],
        compiled_windows=verified,expected_windows=c.plan["total_windows"],blocked_windows=blocked,unique_patterns=len(checked),patterns=checked,matrix_macs=matrix_macs,native_witness_run_matrix_macs=native_macs,
        all_patterns_rebuilt=True,all_layout_bindings_rebuilt=True,all_window_patterns_compiled=not blocked,full_event_lowering_complete=False,event_simulated_cycles=None,
        model_performance_error_available=False,auditor_sha256=digest(Path(__file__))))
    print(f"MODEL_WINDOW_AUDIT_PASS windows={verified} blocked={len(blocked)} matrix_macs={matrix_macs}",flush=True)


if __name__=="__main__":main()
