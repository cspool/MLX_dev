"""Bind complete BERT compilation and numeric reference to a system image.

Preparation executes no inference and is not a Chipyard acceptance result.
"""
import argparse
import json
from pathlib import Path
import subprocess

from mlxsim.model_block_pipeline import compile_block_pipelines
from mlxsim.model_tensor_compiler import compile_inventory
from mlxsim.model_system_evidence import task_coverage
from scripts.mlx_system_attempt import digest,record,snapshot_sources
from scripts.run_mlx_clocked_chipyard import ROOT,prepare,prepare_profile,source_identity
from scripts.run_mlx_qa_model import full_bert_contract,audit_outputs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("inventory","scheduled-program","numeric-reference","device-profile","output"):
        parser.add_argument("--"+key,type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if out.exists():raise ValueError("choose a fresh BERT system image directory")
    inputs={str(path.resolve()):digest(path) for path in (args.inventory,args.scheduled_program,args.device_profile)}
    inventory=json.loads(args.inventory.read_text());options={"max_cycles":10**12}
    program,coverage=compile_inventory(inventory,matrix_backend="scheduled",vector_backend="scheduled",memory_backend="scheduled",control_backend="scheduled",
        schedule_options=options,vector_schedule_options=options,memory_schedule_options=options,control_schedule_options=options)
    program=compile_block_pipelines(program,event_slots=32)
    if program!=json.loads(args.scheduled_program.read_text()):raise ValueError("scheduled source inventory does not reproduce the full bound program")
    contract=full_bert_contract(inventory,program)
    reference=args.numeric_reference.resolve();state=json.loads((reference/"execution.json").read_text())
    if state["status"]!="exited" or state["exit_code"]!=0 or state["mode"]!="microcode":raise ValueError("numeric reference is not a completed full native run")
    if state["inputs"].get(str(args.inventory.resolve()))!=inputs[str(args.inventory.resolve())]:raise ValueError("numeric and system inputs use different inventories")
    if digest(reference/"mlx-tensor-semantics")!=state["binary_sha256"]:raise ValueError("numeric execution binary differs")
    if any(digest(reference/"sources"/name)!=value for name,value in state["sources"].items()):raise ValueError("numeric source snapshot differs")
    for name,value in {**state["inputs"],**state["runtime_libraries"]}.items():
        if digest(Path(name))!=value:raise ValueError("numeric reference inputs or libraries changed")
        inputs[name]=value
    numeric_program=json.loads((reference/"program.json").read_text())
    def numeric(value):return {k:v for k,v in value.items() if not k.endswith("_backend") and not k.endswith("_schedule_options") and k!="block_pipeline_plan"}
    if numeric(program)!=numeric(numeric_program):raise ValueError("system preparation changes the numerical graph")
    native=json.loads((reference/"native/result.json").read_text())
    comparisons=audit_outputs(numeric_program,inventory,native)
    if not all(row["major_correctness_passed"] for row in comparisons):raise ValueError("numeric reference major correctness not established")
    for name in ("execution.json","program.json","native/result.json"):inputs[str(reference/name)]=digest(reference/name)
    before=source_identity();before[str(Path(__file__).resolve().relative_to(ROOT))]=digest(Path(__file__).resolve())
    before["tests/model_storage_contract.cc"]=digest(ROOT/"tests/model_storage_contract.cc")
    out.mkdir(parents=True);snapshot_sources(ROOT,out/"sources",before)
    record(out/"program.json",program);record(out/"coverage.json",coverage)
    with (out/"lifetime-build.log").open("w") as log:
        for command in (["cmake","-S",str(ROOT/"simulator_ext/model_storage"),"-B",str(ROOT/"build/mlx-model-storage"),"-DCMAKE_BUILD_TYPE=Release"],
                        ["cmake","--build",str(ROOT/"build/mlx-model-storage"),"--target","model-storage-contract","-j4"],
                        [str(ROOT/"build/mlx-model-storage/model-storage-contract"),str(out/"program.json"),str(out/"lifetimes.json")]):
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
    profile,profile_identity=prepare_profile(args.device_profile.resolve(),out)
    if any(profile[k][field]!=value for k in ("matrix_options","vector_options") for field,value in (("rows",4),("columns",4),("contexts",2))):raise ValueError("full model system profile changes array geometry")
    case={"name":"bert-paired-resident","program":str(out/"program.json"),"lifetimes":str(out/"lifetimes.json"),"reference":str(reference/"native/result.json")}
    record(out/"cases.json",[case])
    plan,files=prepare(case,out/case["name"],graph_base=0x88000000,graph_bytes=16*2**30-128*2**20,
                       memory_bytes=16*2**30,preload_assets=True,block_pairs=True,event_slots=32)
    tasks,_,_=task_coverage(program,plan)
    if plan["original_source_calls"]!=1141 or plan["lowered_calls"]!=3046 or plan["host_abi_version"]!=3:
        raise ValueError("complete BERT source/stage scope changed")
    if any(digest(ROOT/name)!=value for name,value in before.items()) or any(digest(Path(name))!=value for name,value in {**inputs,**files}.items()):raise ValueError("system preparation inputs changed")
    report={"classification":"complete_bert_system_image_and_compiler_binding_not_execution","sources":before,"inputs":{**inputs,**files},
            "full_model_contract":contract,"numeric_reference_major_correctness":comparisons,"original_source_calls":1141,"lowered_calls":3046,
            "tasks":plan["task_count"],"device_windows":len(tasks),"block_pairs":plan["pair_count"],"command_bytes":plan["command_bytes"],
            "required_mapped_bytes":plan["required_mapped_bytes"],"initial_asset_bytes":plan["initial_asset_bytes"],"profile":profile,"profile_identity":profile_identity,
            "image_sha256":digest(out/case["name"]/"image.json"),"lifetime_sha256":digest(out/"lifetimes.json"),
            "scheduled_program_recompiled_equal":True,"numerical_graph_equal_to_completed_reference":True,
            "full_model_execution_verified":False,"actual_rocket_execution":False,"mlx_system_verified":False,"inference_performance_eligible":False}
    record(out/"report.json",report);print(f"BERT_SYSTEM_IMAGE_PREPARED_NOT_EXECUTED {out/'report.json'}")


if __name__=="__main__":main()
