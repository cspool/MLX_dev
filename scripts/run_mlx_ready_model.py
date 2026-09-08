"""Run an unchanged complete model through native shared-array pair scheduling.

Python compiles/binds inputs and audits outputs. It never advances simulated
cycles or supplies model intermediates. This is not actual Rocket execution.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess

from mlxsim.model_block_pipeline import compile_block_pipelines,references
from mlxsim.model_physical_evidence import scheduled_compile_options
from mlxsim.model_ready_evidence import verify_ready_execution
from mlxsim.model_system_evidence import require,WIDTH
from mlxsim.model_tensor_compiler import compile_inventory
from scripts.mlx_system_attempt import record,run_process,snapshot_sources,linked_libraries
from scripts.run_mlx_clocked_chipyard import ROOT,source_identity as backend_sources
from scripts.run_mlx_tensor_semantics import sha,compare_logits
from scripts.verify_mlx_model_numeric import verify_layers,verify_generation_links,normalize_reference_device
from system_sim.model_image.image import result_reference


def sources():
    result=backend_sources()
    for file in (Path(__file__).resolve(),ROOT/"src/mlxsim/model_ready_evidence.py",ROOT/"scripts/verify_mlx_model_numeric.py"):
        result[str(file.relative_to(ROOT))]=sha(file)
    return result


def bind(file,files,expected=None):
    file=Path(file).resolve()
    # Check each shared shard once while collecting bindings; every collected
    # file is rehashed immediately before launch and after actual execution.
    actual=files[str(file)] if str(file) in files else sha(file)
    require(expected is None or actual==expected,f"input identity differs: {file}")
    files[str(file)]=actual;return actual


def parameter_extents(program,index,model_path,required):
    mapped=[a for a in program["assets"].values() if a["kind"]=="mapped_file"]
    require(len(mapped)==len(required) and {a["parameter_name"] for a in mapped}==required,"parameter assets are missing or duplicated")
    headers={};elements=0
    for asset in mapped:
        name=asset["parameter_name"];file=Path(asset["path"]).resolve()
        require(file==(Path(model_path)/index["weight_map"][name]).resolve(),"parameter was routed to a different checkpoint shard")
        if file not in headers:
            with file.open('rb') as stream:
                prefix=stream.read(8);require(len(prefix)==8,"truncated safetensors header")
                length=struct.unpack('<Q',prefix)[0];require(0<length<=16*2**20 and length+8<=file.stat().st_size,"invalid safetensors header extent")
                headers[file]=(length,json.loads(stream.read(length)))
        length,header=headers[file];entry=header[name];begin,end=entry["data_offsets"]
        require(entry["dtype"]=="F16" and asset["dtype"]=="f16" and entry["shape"]==asset["shape"]
                and 0<=begin<=end and length+8+end<=file.stat().st_size
                and asset["byte_offset"]==length+8+begin and asset["bytes"]==end-begin==math.prod(entry["shape"])*2,
                "parameter shape/precision/byte extent differs from the actual checkpoint header")
        elements+=math.prod(entry["shape"])
    return elements


def full_input_contract(program,source,reference,files):
    for inventory in (source,reference):
        m=inventory["model_identity"]
        require(m["family"]=="Llama2-7B" and m["variant"]=="public_dense_not_paper_hybrid" and m["parameters"]==6738415616
                and m["parameter_tensors"]==291 and m["all_parameters_loaded"],"full native scope needs the complete public dense checkpoint")
        for key,value in {"num_hidden_layers":32,"hidden_size":4096,"intermediate_size":11008,"vocab_size":32000,"num_attention_heads":32,"num_key_value_heads":32}.items():
            require(m["config"][key]==value,"model topology changed or shrank")
        require(inventory["instrumentation_equivalence_passed"],"reference instrumentation changed results");verify_layers(inventory)
        bind(m["model_source"],files,m["model_source_sha256"])
    model=source["model_identity"]
    require(model["files"]==reference["model_identity"]["files"] and source["input"]==reference["input"] and source["input"]["batch"]==1,"source/reference checkpoint or input differs")
    for name,info in model["files"].items():
        bind(name,files,info["sha256"]);require(Path(name).stat().st_size==info["bytes"],"checkpoint extent differs")
    ignored={f"model.layers.{i}.self_attn.rotary_emb.inv_freq" for i in range(32)}
    require(set(model["ignored_nonpersistent_checkpoint_buffers"])==ignored,"unregistered checkpoint omissions")
    index=Path(model["path"])/"model.safetensors.index.json";bind(index,files)
    weight_index=json.loads(index.read_text());required=set(weight_index["weight_map"])-ignored;used=set()
    for n in program["nodes"]:
        for name in set(references(n["args"]))|set(references(n.get("kwargs",{}))):
            asset=program["assets"].get(name)
            if asset and asset["kind"]=="mapped_file":used.add(asset["parameter_name"])
    require(len(required)==291 and used==required,"not all complete-model parameters are consumed")
    require(parameter_extents(program,weight_index,model["path"],required)==6738415616,"actual mapped parameter extents do not cover the complete model")
    options=scheduled_compile_options(program);compiled,_=compile_inventory(source,**options);expected,_=compile_inventory(reference,**options)
    require(compiled==program,"complete source inventory does not compile to the executed program")
    actual_norm,changes=normalize_reference_device(program,source["runtime"]["device"]);expected_norm,ref_changes=normalize_reference_device(expected,reference["runtime"]["device"])
    require(actual_norm==expected_norm and changes==ref_changes,"numeric reference changes more than device placement")
    links=verify_generation_links(program,source);runtime=reference["runtime"]
    require(runtime["matrix_numeric_mode"]=="mlx-matrix-f32-kasc-v1" and runtime["float_numeric_mode"]=="mlx-vector-fp32-v1","numeric contract is not explicit")
    for family,field in (("matrix","matrix_reference_calls"),("vector","float_reference_calls")):
        require(runtime[field]==dict(Counter(n["kind"] for n in program["nodes"] if family+"_program" in n)),"numeric reference operator coverage differs")
    primitive=runtime["numeric_reference_provenance"]
    require(primitive["atomic_primitives_shared_with_cpp_fu"] and primitive["independent_matrix_and_reduction_control"],"numeric reference independence/sharing is ambiguous")
    bind(primitive["libm_path"],files,primitive["libm_sha256"])
    return {"classification":"complete_public_dense_model_input_contract_not_execution","parameters":6738415616,"parameter_tensors":291,
            "source_calls":len(program["nodes"]),"generation_links":links,"device_placement_normalization":changes,"model_variant":"public_dense_not_paper_hybrid","checkpoint_header_extents_checked":True}


def compare_outputs(program,result,reference,output_directory):
    outputs=result_reference(reference)["outputs"]
    require([r["forward_id"] for r in outputs]==[o["forward_id"] for o in program["outputs"]],"reference forward scope differs")
    rows=[]
    for actual,expected in zip(result["outputs"],outputs,strict=True):
        path=Path(actual["logits_file"]).resolve()
        require(path.parent==output_directory.resolve() and actual["shape"]==expected["shape"] and actual["dtype"]==expected["dtype"],"target output path/shape/type differs")
        raw=path.read_bytes();ref=Path(expected["logits_file"]).read_bytes();size=math.prod(actual["shape"])*WIDTH[actual["dtype"]]
        require(len(raw)==len(ref)==size,"output byte count differs")
        rows.append({"forward_id":actual["forward_id"],"bytes":size,"actual_logits_sha256":sha(path),"reference_logits_sha256":sha(Path(expected["logits_file"])),
                     "bitwise_equal":raw==ref,"tokens_equal":actual["tokens"]==expected["tokens"],"tokens":actual["tokens"]})
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("program","reference","options","output"):parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--source-inventory",type=Path);parser.add_argument("--scope",choices=("registered-graph","public-dense-llama2"),default="registered-graph")
    parser.add_argument("--event-slots",type=int,default=32);parser.add_argument("--timeout",type=int,default=172800)
    args=parser.parse_args();out=args.output.resolve()
    require(not out.exists() and args.timeout>0,"choose a fresh native attempt with a positive watchdog")
    require(args.scope!="public-dense-llama2" or args.source_inventory is not None,"full native scope requires source inventory")
    out.mkdir(parents=True);before=sources();files={}
    for file in (args.program,args.reference,args.options):bind(file,files)
    base=json.loads(args.program.read_text());reference=json.loads(args.reference.read_text());options=json.loads(args.options.read_text())
    require("block_pipeline_plan" not in base,"supply the original unmodified numerical graph")
    require(options.get("tile_pipeline") is True and options.get("template_load_timing") is True
            and not options.get("pipeline_whole_source_barrier",False),"native model runner requires real block events and timed template programming")
    source=None;contract=None
    if args.source_inventory:
        bind(args.source_inventory,files);source=json.loads(args.source_inventory.read_text())
    if args.scope=="public-dense-llama2":contract=full_input_contract(base,source,reference,files)
    for asset in base["assets"].values():
        if asset["kind"]=="mapped_file":bind(asset["path"],files,asset.get("file_sha256"))
    for row in result_reference(reference)["outputs"]:bind(row["logits_file"],files)
    if source:
        for row in source["reference_checks"]:bind(row["logits_file"],files,row["logits_sha256"])
    program=compile_block_pipelines(base,event_slots=args.event_slots)
    require({k:v for k,v in program.items() if k!="block_pipeline_plan"}==base,"pipeline compiler changed numerical model")
    record(out/"program.json",program);record(out/"options.json",options)
    bind(out/"program.json",files);bind(out/"options.json",files);snapshot_sources(ROOT,out/"sources",before)
    build=ROOT/"build/mlx-ready-model"
    with (out/"build.log").open("w") as log:
        for command in (["cmake","-S",ROOT/"simulator_ext/model_system","-B",build,"-DCMAKE_BUILD_TYPE=Release"],["cmake","--build",build,"--target","mlx-ready-graph","-j4"]):
            subprocess.run(list(map(str,command)),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    binary=out/"mlx-ready-graph";shutil.copy2(build/"mlx-ready-graph",binary);binary_hash=sha(binary);libraries=linked_libraries(binary)
    if contract:
        primitive=reference["runtime"]["numeric_reference_provenance"]
        require(libraries.get(str(Path(primitive["libm_path"]).resolve()))==primitive["libm_sha256"],"native backend/reference atomic libm differs")
    require(sources()==before and all(sha(Path(p))==h for p,h in files.items()),"native sources/inputs changed before launch")
    state=run_process([binary,out/"program.json",out/"options.json",out/"native"],out/"native.log",out/"execution.json",timeout=args.timeout,cwd=ROOT,
                      metadata={"classification":"owned_native_paired_model_attempt_not_system_certificate","sources":before,"inputs":files,"binary_sha256":binary_hash,"runtime_libraries":libraries,"scope":args.scope,"input_contract":contract,
                                "source_calls":len(base["nodes"]),"pipeline_pairs":len(program["block_pipeline_plan"]["pairs"]),"actual_cpu_execution":False})
    require(sources()==before and sha(binary)==binary_hash and all(sha(Path(p))==h for p,h in {**files,**libraries}.items()),"native attempt sources/inputs/binary changed")
    result=json.loads((out/"native/result.json").read_text());coverage=verify_ready_execution(program,result,options);comparison=compare_outputs(program,result,reference,out/"native")
    passed=all(r["bitwise_equal"] and r["tokens_equal"] for r in comparison)
    gpu=[compare_logits(actual,expected) for actual,expected in zip(result["outputs"],source["reference_checks"],strict=True)] if source else None
    report={"classification":"native_paired_model_numeric_evidence_not_rocket_or_all_mlx_model_acceptance","sources":before,"inputs":files,"binary_sha256":binary_hash,"runtime_libraries":libraries,
            "scope":args.scope,"input_contract":contract,"result_sha256":sha(out/"native/result.json"),"coverage":coverage,"comparison":comparison,"numeric_reference_bitwise_passed":passed,
            "framework_gpu_comparison":gpu,"framework_gpu_comparison_passed":None if gpu is None else all(r["within_tolerance"] and r["tokens_equal"] for r in gpu),
            "full_native_model_numeric_contract_verified":args.scope=="public-dense-llama2" and passed,"full_model_execution_verified":False,"mlx_system_verified":False,"actual_cpu_execution":False,"inference_performance_eligible":False}
    record(out/"comparison.json",report);state["validation"]="native_source_work_checked_numeric_pass" if passed else "native_source_work_checked_numeric_failure";record(out/"execution.json",state)
    require(passed,"native outputs differ from the bound numeric reference; failure evidence preserved")
    print(f"NATIVE_PAIRED_MODEL_NUMERIC_PASS (not system/performance acceptance) {out/'comparison.json'}")


if __name__=="__main__":main()
