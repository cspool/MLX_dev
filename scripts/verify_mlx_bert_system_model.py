"""Accept only a terminal complete BERT Rocket run with actual CPU outputs.

Native baseline, prepared images, and registered graphs cannot satisfy this
audit. No simulation or inference is performed by the Python verifier.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from mlxsim.model_block_pipeline import compile_block_pipelines
from mlxsim.model_physical_evidence import scheduled_compile_options
from mlxsim.model_tensor_compiler import compile_inventory
from scripts.mlx_system_attempt import digest,record
from scripts.run_mlx_qa_model import ATOL,RTOL,full_bert_contract,audit_outputs,require
from scripts.verify_mlx_system_model import Evidence,audit_case,ROOT


def compare_framework(inventory,actual,evidence):
    checks=inventory["reference_checks"]
    require([row["forward_id"] for row in actual["outputs"]]==[row["forward_id"] for row in checks],"CPU/framework QA forward scope differs")
    result=[]
    for check,row in zip(checks,actual["outputs"],strict=True):
        n=len(check["context_mask"])
        answer_equal=all(row["span"][key]==check["span"][key] for key in ("start","end","text"))
        item={"forward_id":row["forward_id"],"answer_equal":answer_equal,"actual_span":row["span"],"outputs":{}}
        for role in ("start_logits","end_logits"):
            own=row["outputs"][role];reference=check["outputs"][role]
            evidence.add(own["file"],own["sha256"]);evidence.add(reference["file"],reference["sha256"])
            got=np.fromfile(own["file"],dtype='<f4').astype(np.float64);ref=np.fromfile(reference["file"],dtype='<f4').astype(np.float64)
            require(got.shape==ref.shape==(n,) and np.isfinite(got).all() and np.isfinite(ref).all(),"CPU/framework QA shape or finite contract differs")
            error=np.abs(got-ref);bad=error>ATOL+RTOL*np.abs(ref)
            item["outputs"][role]={"atol":ATOL,"rtol":RTOL,"max_abs_error":float(error.max()),"failed_elements":int(bad.sum()),"within_tolerance":not bool(bad.any())}
        item["major_correctness_passed"]=answer_equal and all(v["within_tolerance"] for v in item["outputs"].values())
        result.append(item)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("run","program","lifetimes","inventory","numeric-reference","output"):
        parser.add_argument("--"+key,type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();run=args.run.resolve();reference=args.numeric_reference.resolve()
    require(not out.exists() and out!=run.parent and run.parent not in out.parents,"choose a fresh audit directory outside the system attempt")
    out.mkdir(parents=True);evidence=Evidence()
    for file in (Path(__file__).resolve(),ROOT/"scripts/verify_mlx_system_model.py",ROOT/"scripts/run_mlx_qa_model.py",ROOT/"system_sim/physical_host/qa_host.py"):
        evidence.add(file)
    try:
        native_state=evidence.read(reference/"execution.json")
        require(native_state["status"]=="exited" and native_state["exit_code"]==0 and native_state["mode"]=="microcode","numeric reference is not completed native BERT")
        inventory=evidence.read(args.inventory)
        require(native_state["inputs"].get(str(args.inventory.resolve()))==digest(args.inventory),"system source inventory differs from the prebound native reference")
        for name,value in native_state["sources"].items():evidence.add(reference/"sources"/name,value)
        for name,value in {**native_state["inputs"],**native_state["runtime_libraries"]}.items():evidence.add(name,value)
        evidence.add(reference/"mlx-tensor-semantics",native_state["binary_sha256"])
        native_program=evidence.read(reference/"program.json");native=evidence.read(reference/"native/result.json")
        require(all(row["major_correctness_passed"] for row in audit_outputs(native_program,inventory,native)),"full numeric reference major correctness failed")
        result,program,state,profile=audit_case(run,args.program.resolve(),args.lifetimes.resolve(),reference/"native/result.json",out,evidence)
        contract=full_bert_contract(inventory,program)
        options=scheduled_compile_options(program)
        require(all(options[family+"_backend"]=="scheduled" for family in ("matrix","vector","memory","control")),"system program lacks a scheduled primitive route")
        compiled,_=compile_inventory(inventory,**options)
        plan=evidence.read(run/"plan.json")
        require(plan["host_abi_version"]==3 and plan["original_source_calls"]==1141 and plan["lowered_calls"]==3046
                and plan["pair_count"]>0 and plan["pair_event_slots"]==32,"full concurrent BERT task/group scope differs")
        compiled=compile_block_pipelines(compiled,event_slots=plan["pair_event_slots"])
        require(compiled==program,"full BERT inventory does not recompile to the executed system program")
        numerical=lambda p:{k:v for k,v in p.items() if not k.endswith("_backend") and not k.endswith("_schedule_options") and k!="block_pipeline_plan"}
        require(numerical(program)==numerical(native_program),"system graph differs from full native numerical contract")
        for name in ("matrix_options","vector_options"):
            require(profile[name]["rows"]==4 and profile[name]["columns"]==4 and profile[name]["contexts"]==2 and profile[name]["overlap"],"BERT system concurrency/resource profile differs")
        require(result["actual_output_dump_emitted"] and result["output_checks"]["actual_cpu_readback_and_span_execution"],"BERT system result is not an actual CPU output")
        comparisons=compare_framework(inventory,result["output_checks"],evidence)
        require(all(row["major_correctness_passed"] for row in comparisons),"actual system BERT major correctness failed")
        evidence.finish()
        result.update(classification="full_dense_bert_system_numeric_contract_not_all_MLX_models_or_performance_acceptance",
                      full_model_execution_verified=True,full_model_system_numeric_contract_verified=True,
                      model_contract=contract,framework_comparison=comparisons,evidence=evidence.files,
                      original_source_calls=1141,executed_lowered_calls=3046,
                      all_required_models_and_inputs_verified=False,inference_performance_eligible=False)
        record(out/"report.json",result);print(f"FULL_BERT_SYSTEM_NUMERIC_CONTRACT_VERIFIED {out/'report.json'}")
    except Exception as error:
        record(out/"failure.json",{"classification":"failed_BERT_system_audit_not_acceptance","error":str(error),"evidence":evidence.files})
        raise


if __name__=="__main__":main()
