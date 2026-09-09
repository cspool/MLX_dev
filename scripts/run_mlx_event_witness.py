"""Capture Boolean timing witnesses from a complete, owned C++ numerical rerun.

Uses an already accepted microcode executable and its unchanged program. This
is numerical witness provenance, never paired/Chipyard timing certification.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import shutil

import numpy as np

from mlxsim.model_event_graph_plan import graph_plan
from scripts.mlx_system_attempt import digest,record,linked_libraries,run_process
from scripts.verify_mlx_event_schedule import require


def witness_plan(target,reference):
    omitted={"block_pipeline_plan",*(family+suffix for family in ("matrix","vector","memory","control") for suffix in ("_backend","_schedule_options"))}
    require({k:v for k,v in target.items() if k not in omitted}=={k:v for k,v in reference.items() if k not in omitted},"target and numerical reference differ beyond scheduling")
    require([reference.get(f+"_backend") for f in ("matrix","vector","memory","control")]==["microcode","microcode","planned","rv64_leaf"],"reference must execute complete registered microcode without cycle timing")
    plan=graph_plan(target);nodes={n["source_operator_id"]:n for n in target["nodes"]};producer={n["id"]:n for n in target["nodes"]};rows=[]
    for request in plan["timing_witness_requirements"]:
        require(request["kind"] in {"actual_guard_boolean","actual_where_predicate"},"argmax witness requires a separate actual RV64 branch replay")
        node=nodes[request["source_operator_id"]];arg=node["args"][0]
        require(isinstance(arg,dict) and arg.get("value") in producer,"Boolean witness must bind a computed source value")
        parent=producer[arg["value"]];require(parent["kind"]!="split" and parent["output"]["dtype"]=="bool","Boolean observation producer is not directly observable")
        rows.append(dict(**request,producer=parent["source_operator_id"],value=parent["id"],input_shape=parent["output"]["shape"],output_shape=node["output"]["shape"],
                         expected_guard=node["args"][1] if node["kind"]=="guard" else None))
    return rows


def decode_boolean_witness(row,raw):
    values=np.frombuffer(raw,dtype=np.uint8)
    require(values.size==math.prod(row["input_shape"]) and np.isin(values,[0,1]).all(),"actual Boolean observation shape/encoding differs")
    if row["kind"]=="actual_guard_boolean":
        require(values.size==1 and bool(values[0])==row["expected_guard"],"observed guard mismatch")
        return bool(values[0])
    result=np.broadcast_to(values.reshape(row["input_shape"]),row["output_shape"]).astype(bool).reshape(-1).tolist()
    require(len(result)==row["count"],"where witness count differs from actual output")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference",type=Path,required=True);parser.add_argument("--target-program",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--timeout",type=int,default=3600)
    args=parser.parse_args();reference=args.reference.resolve();out=args.output.resolve();require(not out.exists(),"choose a fresh complete witness attempt")
    state=json.loads((reference/"execution.json").read_text());require(state["status"]=="exited" and state["exit_code"]==0 and state["mode"]=="microcode","numerical reference is not a completed microcode attempt")
    acceptance=json.loads((reference/"comparison.json").read_text());require(acceptance["major_correctness_passed"],"numerical reference major correctness not accepted")
    program=json.loads((reference/"program.json").read_text());target=json.loads(args.target_program.read_text());plan=witness_plan(target,program)
    require(plan,"target has no Boolean timing witnesses")
    files={**state["inputs"],str(reference/"execution.json"):digest(reference/"execution.json"),str(reference/"comparison.json"):digest(reference/"comparison.json"),
           str(reference/"native/result.json"):digest(reference/"native/result.json"),str(args.target_program.resolve()):digest(args.target_program)}
    require(all(digest(Path(p))==h for p,h in files.items()),"reference inputs changed")
    original=Path(state["command"][0]);require(digest(original)==state["binary_sha256"],"accepted executable changed")
    baseline=json.loads((reference/"native/result.json").read_text())
    require(program.get("output_contract")=="mlx-qa-result-v1","this runner currently audits complete QA outputs")
    for row in baseline["outputs"]:
        for output in row["outputs"].values():files[output["file"]]=digest(Path(output["file"]))
    out.mkdir(parents=True);shutil.copy2(original,out/"mlx-tensor-semantics");shutil.copy2(reference/"program.json",out/"program.json")
    record(out/"witness-plan.json",plan);record(out/"observation-ids.json",sorted({r["producer"] for r in plan}));record(out/"reference-result.json",baseline)
    owned={str(out/"program.json"):digest(out/"program.json"),str(out/"witness-plan.json"):digest(out/"witness-plan.json"),str(out/"observation-ids.json"):digest(out/"observation-ids.json")}
    libraries=linked_libraries(out/"mlx-tensor-semantics");require(libraries==state["runtime_libraries"],"accepted numerical runtime libraries differ")
    snapshot=out/"sources";snapshot.mkdir()
    for name,sha in state["sources"].items():
        src=reference/"sources"/name;require(digest(src)==sha,"accepted numerical source snapshot changed");dst=snapshot/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    runner_hash=digest(Path(__file__));command=[str(out/"mlx-tensor-semantics"),str(out/"program.json"),str(out/"native"),"none","1",str(out/"observation-ids.json")]
    run_process(command,out/"native.log",out/"execution.json",timeout=args.timeout,metadata=dict(mode="full_cpp_boolean_witness_capture_not_paired_timing",sources=state["sources"],
        inputs={**files,**owned},binary_sha256=state["binary_sha256"],runtime_libraries=libraries,runner_sha256=runner_hash,input_contract=state["input_contract"]))
    require(all(digest(Path(p))==h for p,h in {**files,**owned,**libraries}.items()) and digest(out/"mlx-tensor-semantics")==state["binary_sha256"] and digest(Path(__file__))==runner_hash,"witness execution provenance changed")
    actual=json.loads((out/"native/result.json").read_text());normalized=copy.deepcopy(actual);prior=copy.deepcopy(baseline)
    for report in (normalized,prior):report.pop("observations");report.pop("observation_bytes")
    for new,old in zip(normalized["outputs"],prior["outputs"],strict=True):
        for role,value in new["outputs"].items():
            require(Path(value["file"]).read_bytes()==Path(old["outputs"][role]["file"]).read_bytes(),"full-model observed run changed numerical output")
            value["file"]=old["outputs"][role]["file"]
    require(normalized==prior,"observations changed full numerical execution report")
    observations={r["source_operator_id"]:r for r in actual["observations"]};require(set(observations)=={r["producer"] for r in plan},"witness observations incomplete or duplicated")
    witnesses=[]
    for row in plan:
        observation=observations[row["producer"]];path=Path(observation["file"])
        require(observation["dtype"]=="bool" and observation["shape"]==row["input_shape"] and observation["value_id"]==row["value"],"witness producer binding differs")
        value=decode_boolean_witness(row,path.read_bytes());witnesses.append(dict(**row,observed_file=str(path),observed_sha256=digest(path),value_data=value))
    record(out/"witnesses.json",dict(classification="full_cpp_numerical_boolean_witnesses_not_paired_or_chipyard_timing",target_program_sha256=files[str(args.target_program.resolve())],
        reference_program_sha256=owned[str(out/"program.json")],native_report_sha256=digest(out/"native/result.json"),full_numerical_execution_equal=True,
        witnesses=witnesses,paired_execution_verified=False,mlx_system_verified=False,model_performance_error_available=False))
    print(f"FULL_CPP_BOOLEAN_WITNESSES_PASS count={len(witnesses)}",flush=True)


if __name__=="__main__":main()
