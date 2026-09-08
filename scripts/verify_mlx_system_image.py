"""Audit complete model image/reference binding and initialize its actual RAM layout."""
import argparse
import json
import subprocess
from pathlib import Path

from mlxsim.model_tensor_compiler import compile_inventory
from mlxsim.model_physical_evidence import scheduled_compile_options
from scripts.verify_mlx_model_numeric import normalize_reference_device,verify_generation_links
from scripts.run_mlx_clocked_chipyard import ROOT,source_identity,prepare
from scripts.verify_mlx_system_model import initialized_assets
from mlxsim.model_system_evidence import task_coverage
from scripts.run_mlx_spike_graph import sha
from system_sim.physical_host.graph_lowering import compile_graph


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("image","program","lifetimes","source","reference","output"):parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();image=args.image.resolve()
    if out.exists():raise RuntimeError("choose a fresh full system-image audit directory")
    out.mkdir(parents=True)
    def identity():
        result=source_identity()
        for path in (Path(__file__).resolve(),ROOT/"scripts/verify_mlx_model_numeric.py",ROOT/"scripts/verify_mlx_system_model.py",ROOT/"tests/wide_memory_contract.cc"):result[str(path.relative_to(ROOT))]=sha(path)
        return result
    sources=identity();paths=[args.program,args.lifetimes,args.source,args.reference,image/"image.json",image/"plan.json",image/"segments.json",image/"test.elf",image/"command_blob.bin",image/"test.c",image.parent/"profile.json"]
    fingerprints={str(path.resolve()):sha(path) for path in paths}
    program=json.loads(args.program.read_text());source=json.loads(args.source.read_text());reference=json.loads(args.reference.read_text());record=json.loads((image/"image.json").read_text());plan=json.loads((image/"plan.json").read_text())
    for path,digest in record["inputs"].items():
        if sha(Path(path))!=digest:raise RuntimeError("prepared image input identity changed")
        fingerprints[path]=digest
    for key,file in (("elf_sha256","test.elf"),("commands_sha256","command_blob.bin"),("plan_sha256","plan.json"),("host_source_sha256","test.c")):
        if sha(image/file)!=record[key]:raise RuntimeError("prepared image artifact digest differs")
    options=scheduled_compile_options(program);compiled,_=compile_inventory(source,**options);expected,_=compile_inventory(reference,**options)
    if compiled!=program:raise RuntimeError("source inventory no longer compiles to the image program")
    actual_normalized,changes=normalize_reference_device(program,source["runtime"]["device"]);expected_normalized,reference_changes=normalize_reference_device(expected,reference["runtime"]["device"])
    if actual_normalized!=expected_normalized or changes!=reference_changes:raise RuntimeError("reference differs beyond declared device-placement normalization")
    links=verify_generation_links(program,source)
    if record["source_calls"]!=len(program["nodes"]) or record["task_count"]!=len(plan["tasks"]):raise RuntimeError("image source/task coverage incomplete")
    paired=plan.get("host_abi_version",1)==2;event_slots=plan.get("pair_event_slots",32)
    rebuilt,layout=compile_graph(program,json.loads(args.lifetimes.read_text()),device_base=plan["device_base"],device_bytes=plan["device_bytes"],data_offset=plan["data_offset"],scratch_offset=plan["scratch_offset"],scratch_bytes=plan["scratch_bytes"],block_pairs=paired,event_slots=event_slots)
    if rebuilt!=(image/"command_blob.bin").read_bytes() or any(plan[key]!=value for key,value in layout.items()):raise RuntimeError("image commands/bindings differ from complete compiler replay")
    task_coverage(program,plan)
    replay=out/"elf-replay"
    replay_plan,_=prepare({"program":str(args.program.resolve()),"lifetimes":str(args.lifetimes.resolve()),"reference":str(args.reference.resolve())},replay,
                          graph_base=plan["device_base"],graph_bytes=plan["device_bytes"],memory_bytes=16*2**30,preload_assets=True,block_pairs=paired,event_slots=event_slots)
    if replay_plan!=plan:raise RuntimeError("complete image replay changed the task/lifetime plan")
    rebuilt_images={}
    for name in ("test.c","test.elf","command_blob.bin","launch-map.json"):
        if sha(replay/name)!=sha(image/name):raise RuntimeError(f"complete image checker/ELF replay differs: {name}")
        rebuilt_images[name]=sha(replay/name)
    profile=json.loads((image.parent/"profile.json").read_text());native_profile={"version":1,"name":profile["name"],"max_busy_cycles":profile["max_busy_cycles"]}
    for kind in ("matrix","vector","memory"):native_profile[kind+"_options"]=program[kind+"_schedule_options"]
    (out/"native-profile.json").write_text(json.dumps(native_profile)+"\n")
    parsed=subprocess.run([str(ROOT/"build/mlx-system-profile/system-profile-dump"),str(out/"native-profile.json")],capture_output=True,text=True,check=True,timeout=30)
    if json.loads(parsed.stdout)!=profile:raise RuntimeError("system device profile differs from native model resource/timing options")
    if record["cpu_asset_copy_bytes"] or record["asset_initialization"]!="preloaded_resident_model_input_not_cpu_dma":raise RuntimeError("this audit requires explicitly resident input assets")
    segments=json.loads((image/"segments.json").read_text())
    if {row["name"] for row in segments}!=set(program["assets"]):raise RuntimeError("image asset IDs differ from the full program")
    for row in segments:
        binding=plan["assets"][row["name"]]
        if row["address"]!=binding["base"] or row["file_bytes"]!=binding["bytes"] or row["memory_bytes"]!=binding["bytes"]:raise RuntimeError("resident asset placement differs from compiled binding")
    build=ROOT/"build/mlx-wide-memory";binary=build/"wide-memory-contract"
    with (out/"build.log").open("w") as log:subprocess.run(["cmake","--build",str(build),"--target","wide-memory-contract","-j4"],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    binary_hash=sha(binary);job={"base":0x80000000,"bytes":16*2**30,"elf":str(image/"test.elf"),"segments":segments}
    (out/"job.json").write_text(json.dumps(job)+"\n")
    with (out/"memory.json").open("w") as output, (out/"memory.stderr").open("w") as error:subprocess.run([str(binary),str(out/"job.json")],stdout=output,stderr=error,check=True,timeout=600)
    memory=json.loads((out/"memory.json").read_text());loaded=[row for row in memory["initialized_segments"] if not row["name"].startswith("elf:")]
    if loaded!=segments or not any(row["name"].startswith("elf:") for row in memory["initialized_segments"]):raise RuntimeError("ELF and all resident assets did not initialize together")
    if memory["cycle"] or memory["ar_requests"] or memory["aw_requests"]:raise RuntimeError("initialization audit unexpectedly executed the system")
    initialization=initialized_assets(program,plan,segments,memory,image/"test.elf",json.loads((replay/"segments.json").read_text()))
    for name,digest in rebuilt_images.items():
        if sha(replay/name)!=digest:raise RuntimeError("replayed image changed during initialization audit")
    if sources!=identity() or sha(binary)!=binary_hash or any(sha(Path(path))!=digest for path,digest in fingerprints.items()):raise RuntimeError("image audit sources/inputs changed")
    report={"classification":"full_model_system_image_reference_and_ram_initialization_audit_not_execution","sources":sources,"inputs":fingerprints,"source_calls":len(program["nodes"]),"task_count":len(plan["tasks"]),"asset_count":len(segments),"asset_bytes":sum(row["file_bytes"] for row in segments),
        "reference_device_normalization":changes,"generation_links":links,"complete_compiler_replay_equal":True,"complete_cpu_elf_rebuild_equal":True,"rebuilt_images":rebuilt_images,"initialization":initialization,
        "host_abi_version":plan.get("host_abi_version",1),"pair_count":plan.get("pair_count",0),"device_profile_matches_native":True,"memory_sha256":sha(out/"memory.json"),"binary_sha256":binary_hash,"full_model_execution_verified":False,"actual_cpu_execution":False,"mlx_system_verified":False,"inference_performance_eligible":False}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(f"COMPLETE_SYSTEM_IMAGE_AUDIT_PASS {out/'report.json'}")


if __name__=="__main__":main()
