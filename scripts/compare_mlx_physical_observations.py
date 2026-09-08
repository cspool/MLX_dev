"""Compare selected completed physical outputs; never certify a partial model."""
import argparse
import json
import math
from pathlib import Path

from mlxsim.model_tensor_compiler import compile_inventory
from mlxsim.model_physical_evidence import BYTES,scheduled_compile_options
from scripts.run_mlx_tensor_semantics import sha
from scripts.verify_mlx_model_numeric import require,normalize_reference_device,verify_layers


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True);parser.add_argument("--source-inventory",type=Path,required=True)
    parser.add_argument("--numeric-reference",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();run=args.run.resolve();require(not args.output.exists(),"choose a fresh physical observation comparison path")
    source=json.loads(args.source_inventory.read_text());reference=json.loads(args.numeric_reference.read_text())
    execution=json.loads((run/"execution.json").read_text());program=json.loads((run/"program.json").read_text());options=json.loads((run/"system-options.json").read_text())
    require(execution["status"] in {"completed","failed","host_watchdog_expired"} and execution["source_and_program_identity_unchanged"] is True,"observation run has not ended with stable identities")
    require(sha(args.source_inventory)==execution["inventory_sha256"] and sha(run/"program.json")==execution["program_sha256"] and sha(run/"system-options.json")==execution["system_options_sha256"],"observation input/program identity mismatch")
    require(sha(run/"mlx-physical-model")==execution["binary_sha256"],"observation producer binary changed")
    for name,digest in execution["sources"].items():require(sha(run/"sources"/name)==digest,"observation producer source snapshot changed")
    for name,digest in execution["runtime_libraries"].items():require(sha(Path(name))==digest,"observation producer runtime library changed")
    require(source["model_identity"]["files"]==reference["model_identity"]["files"] and source["input"]==reference["input"],"observation reference changed model/input")
    for name,info in source["model_identity"]["files"].items():require(sha(Path(name))==info["sha256"],"observation model asset changed")
    verify_layers(source);verify_layers(reference)
    require(reference["runtime"]["matrix_numeric_mode"]=="mlx-matrix-f32-kasc-v1" and reference["runtime"]["float_numeric_mode"]=="mlx-vector-fp32-v1","observation reference numeric profile is not registered")
    current,_=compile_inventory(source,**scheduled_compile_options(program));ref_program,_=compile_inventory(reference,**scheduled_compile_options(program))
    require(current==program,"current compiler differs from physical observation program")
    require(normalize_reference_device(program,source["runtime"]["device"])[0]==normalize_reference_device(ref_program,reference["runtime"]["device"])[0],"observation reference differs beyond device placement")
    observed_file=run/"native/observations.json";observed=json.loads(observed_file.read_text());reference_file=args.numeric_reference.parent/"diagnostics/observations.json";ref_rows=json.loads(reference_file.read_text())["observations"]
    require(observed["classification"]=="physical_output_digests_not_model_completion" and observed["certifies_full_model"] is False,"observation file improperly claims full completion")
    by_ref={r["source_operator_id"]:r for r in ref_rows};nodes={n["source_operator_id"]:n for n in program["nodes"]};seen=set();results=[]
    torch_dtype={"f16":"torch.float16","f32":"torch.float32","i64":"torch.int64","bool":"torch.bool"}
    for row in observed["observations"]:
        identifier=row["source_operator_id"];require(identifier not in seen and identifier in options["observe_operators"] and identifier in nodes and identifier in by_ref,"unknown/duplicate physical observation")
        seen.add(identifier);node=nodes[identifier];expected=by_ref[identifier]
        require(row["producer_completed"] is True and row["forward_id"]==node["forward_id"]==expected["forward_id"] and row["layer_idx"]==node["layer_idx"]==expected["layer_idx"],"physical observation join keys differ")
        require(row["shape"]==node["output"]["shape"]==expected["shape"] and torch_dtype[row["dtype"]]==expected["dtype"],"physical observation shape/dtype differs")
        bytes=math.prod(row["shape"])*BYTES[row["dtype"]];file=Path(expected["file"])
        require(row["bytes"]==bytes==file.stat().st_size and row["sha256"]==expected["sha256"]==sha(file),"physical output digest differs from numeric reference")
        results.append({"source_operator_id":identifier,"module_path":node["module_path"],"shape":row["shape"],"bytes":bytes,"sha256":row["sha256"],"digest_equal":True})
    require(results,"no completed physical observations")
    report={"classification":"completed_physical_output_observations_not_full_model_validation","producer_status":execution["status"],"observations":results,
        "execution_sha256":sha(run/"execution.json"),"program_sha256":sha(run/"program.json"),"producer_binary_sha256":execution["binary_sha256"],"observations_sha256":sha(observed_file),
        "source_inventory_sha256":sha(args.source_inventory),"numeric_reference_inventory_sha256":sha(args.numeric_reference),"reference_observations_sha256":sha(reference_file),
        "full_model_execution_verified":False,"mlx_system_verified":False,"inference_performance_eligible":False}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(f"PHYSICAL_SELECTED_OUTPUT_DIGESTS_MATCH (not full model) {args.output}")


if __name__=="__main__":main()
