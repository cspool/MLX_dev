"""Rebuild completed small Rocket graph images and audit actual paired work.

This is a read-only execution audit: it rebuilds ELF/checker artifacts in a new
directory, but does not run any simulated model or certify full-model timing.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from mlxsim.model_system_evidence import ADAPTER_CLASS,require,count,terminal_execution,task_coverage,check_kernel,check_pair_kernel
from scripts.run_mlx_clocked_chipyard import ROOT,CONFIG,prepare,source_identity,identity
from scripts.verify_mlx_system_model import Evidence
from scripts.mlx_system_attempt import launch_metadata


def feedback_paths(program):
    outputs=program["outputs"]
    if len(outputs)==1:return None
    require([o["forward_id"] for o in outputs]==list(range(len(outputs))),"generation forwards are missing or reordered")
    nodes={n["id"]:n for n in program["nodes"]};embeddings={};caches={}
    for node in program["nodes"]:
        if node["kind"] not in ("embedding","cat"):continue
        table=embeddings if node["kind"]=="embedding" else caches
        require(node["forward_id"] not in table,"duplicate registered generation embedding/cache")
        table[node["forward_id"]]=node
    require(set(embeddings)==set(caches)==set(range(len(outputs))),"registered single-cache generation path is incomplete")
    links=[];previous_cache=None;previous_length=0
    for i,output in enumerate(outputs):
        embedding,cache=embeddings[i],caches[i];index=embedding["args"][1]["value"];chain=[]
        selection=nodes[output["token"]]
        require(selection["kind"]=="argmax" and selection["args"][0]["value"]==output["logits"],"token does not select the actual forward logits")
        if i:
            target=outputs[i-1]["token"]
            while index!=target:
                require(index in nodes and index not in chain,"decode indices come from a literal or cycle")
                node=nodes[index];chain.append(index)
                require(node["kind"] in ("unsqueeze","reshape","alias","contiguous","cast","cast_device")
                        and node["output"]["dtype"]=="i64" and math.prod(node["output"]["shape"])==1,"decode index path changes the selected token")
                index=node["args"][0]["value"]
        else:
            require(index in program["assets"],"initial index is not an input asset")
            asset=program["assets"][index]
            require(asset["kind"]=="literal" and asset["origin"]=="input" and asset["dtype"]=="i64","initial indices are not explicitly bound inputs")
        parts=cache["args"][0]
        require(cache["args"][1]==1 and len(parts)==2 and parts[1]["value"]==embedding["id"],"cache does not append the actual embedding")
        prefix=parts[0]["value"]
        if i:require(prefix==previous_cache,"decode cache does not consume the previous cache output")
        else:
            require(prefix in program["assets"] and program["assets"][prefix]["origin"]=="input" and math.prod(program["assets"][prefix]["shape"])==0,"initial cache is not the bound empty input")
        length=cache["output"]["shape"][1]
        require(length==previous_length+embedding["output"]["shape"][1],"cache length does not grow from actual appended data")
        links.append({"forward_id":i,"index_source":index,"token_view_chain":chain,"cache_prefix":prefix,"cache_output":cache["id"],"cache_length":length})
        previous_cache,previous_length=cache["id"],length
    return {"classification":"registered_single_cache_graph_feedback_not_general_model_cache_validation","forwards":len(outputs),"links":links}


def device_work(program,plan,device,profile):
    tasks,values,counts=task_coverage(program,plan)
    require(device["classification"]==ADAPTER_CLASS and device["effective_profile"]==profile
            and device["source_id_basis"]=="transport_launch_ordinal_not_model_operator_identity","actual adapter/profile differs")
    require(not device["frontend_error"] and not device["cache_request_owned"] and not device["cpu_response_pending"],"CPU/cache lifecycle did not drain")
    require(device["launches"]==len(tasks)==len(device["windows"]) and device["requests"]==device["responses"],"device launch/response coverage differs")
    observer=device.get("progress_observer")
    require(observer is None or not observer["failed"],"observer failed")
    totals=Counter();routes=[];previous_requests=previous_end=0
    for ordinal,(task,window) in enumerate(zip(tasks,device["windows"],strict=True)):
        require(window["source_id"]==ordinal and window["launches"]==ordinal+1 and window["backend"]==task["family"],"actual task/launch/source routing differs")
        require(window["done"] and window["complete"] and not window["busy"] and not window["error"],"actual window did not finish successfully")
        require(window["descriptor_bytes_fetched"]==task["bytes"],"descriptor fetch is incomplete")
        start,end=count(window["launch_start_cycle"],"start"),count(window["cycle"],"end")
        fetch,run,drain=[count(window[k+"_cycles"],k) for k in ("fetch","run","drain")]
        require(start>=previous_end and end==start+fetch+run+drain and not drain and run==max(1,window["kernel"]["cycles"]),"external/backend edges disagree")
        transport=window["transport"];submitted=count(transport["submitted"],"submitted")
        require(transport["idle"] and not any(transport[k] for k in ("inflight","queued","response_pending","nacks"))
                and all(transport[k]==submitted for k in ("accepted","responses","consumed")),"transport conservation differs")
        work=(check_pair_kernel(program,task,window["kernel"],profile,values,plan["pair_event_slots"],ordinal+1) if task["kind"]==3
              else check_kernel(program["nodes"][task["source_ordinal"]],task,window["kernel"],profile,values))
        if task["kind"]==3:
            array=window["kernel"]["array"];m=profile["matrix_options"];v=profile["vector_options"]
            require(all(m[k]==v[k] for k in ("rows","columns","contexts")) and array["context_slots_per_pe"]==m["contexts"]
                    and 0<=array["peak_contexts"]<=m["rows"]*m["columns"]*m["contexts"],"pair physical context capacity differs")
        require(submitted-previous_requests==task["bytes"]//8+work["requests"],"descriptor/data traffic coverage differs")
        totals.update(work);totals.update(fetch_cycles=fetch,run_cycles=run)
        routes.append({"launch_ordinal":ordinal,"source_ordinal":task["source_ordinal"],"source_id":task["source_id"],"family":task["family"],"work":work,
                       **({"consumer_ordinal":task["consumer_ordinal"],"consumer_source_id":task["consumer_source_id"]} if task["kind"]==3 else {})})
        previous_requests,previous_end=submitted,end
    require(tasks and previous_requests==device["requests"],"unexpected final request count")
    final=device["device"]
    require(final["cycle"]>=previous_end and {k:v for k,v in final.items() if k!="cycle"}=={k:v for k,v in device["windows"][-1].items() if k!="cycle"},"final controller differs from last completed window")
    return {"source_calls":plan["source_calls"],"family_source_calls":counts,"device_windows":len(tasks),"pair_launches":sum(t["kind"]==3 for t in tasks),"work":dict(totals),"routes":routes}


def audit_case(run,case,out,evidence):
    state=evidence.read(run/"execution.json");evidence.add(run/"chipyard.log")
    terminal_execution(state,(run/"chipyard.log").read_text())
    require(state["sources"]==source_identity(),"current compiler/backend sources differ from executed sources")
    for name,expected in state["sources"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts,"unsafe source snapshot path")
        evidence.add(ROOT/name,expected);evidence.add(run.parent/"sources"/name,expected)
    for values in (state["inputs"],state["runtime_libraries"]):
        for name,expected in values.items():evidence.add(name,expected)
    for key in ("program","lifetimes","reference"):
        require(str(Path(case[key]).resolve()) in state["inputs"],"requested input not bound before execution")
    program=evidence.read(Path(case["program"]));plan=evidence.read(run/"plan.json");image=evidence.read(run/"image.json")
    require(plan.get("host_abi_version")==2 and plan["asset_initialization"]=="cpu_copy_from_elf_payload","this audit registers v2 CPU-copy small graphs only")
    require(state["inputs"]==image["inputs"] and state["source_calls"]==image["source_calls"]==plan["source_calls"] and state["task_count"]==image["task_count"]==plan["task_count"],"image/process source/input scope differs")
    for key,name in (("elf_sha256","test.elf"),("commands_sha256","command_blob.bin"),("plan_sha256","plan.json"),("host_source_sha256","test.c")):
        evidence.add(run/name,image[key])
    build=evidence.read(run.parent/"build.json");binary=Path(state["command"][0]).resolve()
    require(not build["inputs"]["large_memory"] and binary==run.parent/f"simulator-chipyard-{CONFIG}","audit is not the attempt-owned default Rocket configuration")
    require(build["inputs"]["sources"]==state["sources"] and build["build_identity"]==identity(build["inputs"])==state["build_identity"],"build source/identity differs")
    evidence.add(binary,build["simulator_sha256"])
    require(state["simulator_sha256"]==build["simulator_sha256"] and state["elf_sha256"]==image["elf_sha256"] and state["command"][-1]==str(run/"test.elf"),"executed ELF/binary differs")
    profile=evidence.read(run.parent/"profile.json",state["profile_identity"]["effective_sha256"])
    device=evidence.read(run/"device.json");require(device["build_identity"]==state["build_identity"],"device identity differs")
    work=device_work(program,plan,device,profile)
    require(evidence.read(run/"launch-map.json")==launch_metadata(program,plan),"observer source map differs")
    rebuilt,_=prepare(case,out,graph_base=plan["device_base"],graph_bytes=plan["device_bytes"],block_pairs=True,event_slots=plan["pair_event_slots"])
    require(rebuilt==plan,"full task/lifetime replay differs")
    for name in ("test.c","test.elf","command_blob.bin","payload_blob.bin","assets.json","launch-map.json"):
        require(evidence.add(out/name)==evidence.add(run/name),f"executed image differs from complete rebuild: {name}")
    return {"case":case["name"],"classification":"actual_rocket_graph_with_rebuilt_cpu_checker_not_full_model_validation","complete_elf_and_payload_rebuild_equal":True,
            "generation_feedback":feedback_paths(program),
            "actual_cpu_source_completion_and_output_checks":True,"output_dump_emitted":False,"device_work":work,
            "full_model_execution_verified":False,"inference_performance_eligible":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run",type=Path,required=True);parser.add_argument("--cases",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);parser.add_argument("--case",action="append")
    args=parser.parse_args();out=args.output.resolve();run=args.run.resolve()
    if out.exists():raise RuntimeError("choose a fresh graph image audit directory")
    evidence=Evidence();audit_sources={str(p.relative_to(ROOT)):evidence.add(p) for p in (Path(__file__).resolve(),ROOT/"scripts/verify_mlx_system_model.py",ROOT/"src/mlxsim/model_system_evidence.py")}
    cases=evidence.read(args.cases);names=[c["name"] for c in cases];chosen=args.case or names
    require(len(set(names))==len(names) and chosen and len(set(chosen))==len(chosen) and set(chosen)<=set(names),"unknown/duplicate requested cases")
    require(all(name and all(c in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in name) for name in names),"unsafe case path")
    selected=[c for c in cases if c["name"] in chosen]
    # Refuse still-running or failed cases before creating any replay artifacts.
    for case in selected:
        directory=run/case["name"];state=evidence.read(directory/"execution.json");evidence.add(directory/"chipyard.log")
        terminal_execution(state,(directory/"chipyard.log").read_text())
    out.mkdir(parents=True);results=[audit_case(run/c["name"],c,out/c["name"],evidence) for c in selected];evidence.finish()
    report={"classification":"selected_registered_rocket_graph_image_audit_not_full_model_or_performance_acceptance","audit_sources":audit_sources,"cases":results,
            "all_cases_in_input_audited":set(chosen)==set(names),"input_case_count":len(names),"audited_case_count":len(results),"files":evidence.files,
            "actual_memory_timing_calibrated":False,"full_model_execution_verified":False,"mlx_system_verified":False,"inference_performance_eligible":False,"rtl_verified":False}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(f"PAIR_GRAPH_IMAGE_AUDIT_PASS {out/'report.json'}")


if __name__=="__main__":main()
