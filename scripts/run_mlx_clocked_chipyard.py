"""Build real Rocket + externally clocked C++ tensor backends and run graph ELFs."""
import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from scripts.run_mlx_spike_graph import ROOT,HOST,sha,host_source,check_embedded_payload_limit
from scripts.verify_mlx_spike_matrix_chain import source_identity as bridge_sources
from scripts.run_mlx_native_chipyard import build_inputs as chipyard_inputs,identity,CHIPYARD_COMMIT
from system_sim.physical_host.graph_lowering import compile_graph,write_payload
from system_sim.model_image.image import initial_segments,result_reference
from scripts.mlx_system_attempt import run_process,snapshot_sources,launch_metadata,linked_libraries,record as record_attempt

BRIDGE=ROOT/"system_sim/clocked_rocc"
BUILD=ROOT/"build/mlx-clocked-rocc"
CONFIG="MLXClockedRocketConfig"
WIDE=ROOT/"system_sim/wide_memory"
WIDE_BUILD=ROOT/"build/mlx-wide-memory"
PROFILE=ROOT/"system_sim/model_image"


def source_identity():
    result=bridge_sources();files=[Path(__file__).resolve(),ROOT/"scripts/mlx_system_attempt.py",ROOT/"scripts/run_mlx_spike_graph.py",ROOT/"scripts/run_mlx_native_chipyard.py",ROOT/"src/mlxsim/model_system_evidence.py",ROOT/"system_sim/chipyard/MLXClockedRoCC.scala",ROOT/"system_sim/chipyard/MLXWideMemory.scala"]
    for directory in (BRIDGE,WIDE,PROFILE,ROOT/"system_sim/clocked_device"):files.extend(p for p in directory.iterdir() if p.is_file())
    for path in files:result[str(path.relative_to(ROOT))]=sha(path)
    return result


def execute(command,log,timeout=3600,env=None):
    print(f"CLOCKED_CHIPYARD {log}",flush=True)
    with log.open("w") as output:subprocess.run(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,check=True,timeout=timeout,env=env)


def install(chipyard,large=False):
    revision=subprocess.run(["git","-c",f"safe.directory={chipyard}","-C",str(chipyard),"rev-parse","HEAD"],capture_output=True,text=True,check=True).stdout.strip()
    if revision!=CHIPYARD_COMMIT:raise RuntimeError("clocked bridge requires the pinned Chipyard revision")
    pairs={ROOT/"system_sim/chipyard/MLXClockedRoCC.scala":chipyard/"generators/chipyard/src/main/scala/MLXClockedRoCC.scala",
        BRIDGE/"MLXClockedRoCC.sv":chipyard/"generators/chipyard/src/main/resources/vsrc/MLXClockedRoCC.sv"}
    if large:pairs.update({ROOT/"system_sim/chipyard/MLXWideMemory.scala":chipyard/"generators/chipyard/src/main/scala/MLXWideMemory.scala",WIDE/"MLXWideMemory.sv":chipyard/"generators/chipyard/src/main/resources/vsrc/MLXWideMemory.sv"})
    manifest=chipyard/"clocked-rocc-install.json";previous=json.loads(manifest.read_text()) if manifest.exists() else {}
    for source,destination in pairs.items():
        if destination.exists() and sha(destination)!=sha(source) and previous.get(str(destination))!=sha(destination):raise RuntimeError(f"unrecognized installed bridge edits: {destination}")
        destination.parent.mkdir(parents=True,exist_ok=True)
        if not destination.exists() or sha(destination)!=sha(source):shutil.copy2(source,destination)
    previous.update({str(p):sha(p) for p in pairs.values()});manifest.write_text(json.dumps(previous,indent=2)+"\n")
    return list(pairs.values())


def libraries(large=False):
    return [BUILD/"libmlx_clocked_rocc.a",BUILD/"model-image/libmlx_system_profile.a",BUILD/"clocked-device/libmlx_clocked_device.a",BUILD/"clocked-device/physical-wire/libmlx_matrix_wire.a",
        BUILD/"clocked-device/physical-wire/tensor-model/libmlx_tensor_values.a",BUILD/"clocked-device/physical-wire/tensor-model/shared-array/libmlx_shared_array.a",BUILD/"clocked-device/physical-wire/tensor-model/tagged-core/libmlx_tagged.a"]+([WIDE_BUILD/"libmlx_wide_memory.a"] if large else [])


def inputs(chipyard,large=False):
    installed=[chipyard/"generators/chipyard/src/main/scala/MLXClockedRoCC.scala",chipyard/"generators/chipyard/src/main/resources/vsrc/MLXClockedRoCC.sv"]
    if large:installed.extend([chipyard/"generators/chipyard/src/main/scala/MLXWideMemory.scala",chipyard/"generators/chipyard/src/main/resources/vsrc/MLXWideMemory.sv"])
    return {"sources":source_identity(),"dependencies":chipyard_inputs(chipyard),"large_memory":large,"libraries":{str(p):sha(p) for p in libraries(large)},"installed":{str(p):sha(p) for p in installed}}


def prepare(case,out,*,graph_base=0x81000000,graph_bytes=1048576,memory_bytes=256*2**20,preload_assets=False,block_pairs=False,event_slots=32):
    out.mkdir();paths={key:Path(case[key]).resolve() for key in ("program","lifetimes","reference")};fingerprints={str(p):sha(p) for p in paths.values()}
    program=json.loads(paths["program"].read_text());life=json.loads(paths["lifetimes"].read_text());reference=result_reference(json.loads(paths["reference"].read_text()))
    for row in reference["outputs"]:
        path=Path(row["logits_file"]).resolve();fingerprints[str(path)]=sha(path)
    for asset in program["assets"].values():
        if asset["kind"]=="mapped_file":
            path=Path(asset["path"]).resolve()
            if str(path) not in fingerprints:fingerprints[str(path)]=sha(path)
            if asset.get("file_sha256",fingerprints[str(path)])!=fingerprints[str(path)]:raise RuntimeError("compiled asset identity mismatch")
    if not 0x80000000<=graph_base or graph_bytes<=0 or graph_base+graph_bytes>0x80000000+memory_bytes:raise RuntimeError("graph window outside actual system RAM")
    blob,plan=compile_graph(program,life,device_base=graph_base,device_bytes=graph_bytes,block_pairs=block_pairs,event_slots=event_slots)
    check_embedded_payload_limit({"assets":{}} if preload_assets else program,len(blob))
    if preload_assets:
        segments=initial_segments(program,plan,out);assets=[]
        (out/"segments.json").write_text(json.dumps(segments,indent=2)+"\n")
        for path in (out/"segments.json",out/"literal_assets.bin"):fingerprints[str(path)]=sha(path)
    else:
        assets=write_payload(program,out/"payload_blob.bin");segments=[]
    plan["asset_initialization"]="preloaded_resident_model_input_not_cpu_dma" if preload_assets else "cpu_copy_from_elf_payload"
    plan["initial_asset_count"]=len(segments) if preload_assets else len(assets)
    plan["initial_asset_bytes"]=sum(row["file_bytes"] for row in segments) if preload_assets else sum(row["bytes"] for row in assets)
    plan["cpu_asset_copy_bytes"]=0 if preload_assets else plan["initial_asset_bytes"]
    (out/"launch-map.json").write_text(json.dumps(launch_metadata(program,plan),indent=2)+"\n")
    fingerprints[str(out/"launch-map.json")]=sha(out/"launch-map.json")
    (out/"command_blob.bin").write_bytes(blob);(out/"plan.json").write_text(json.dumps(plan,indent=2)+"\n");(out/"assets.json").write_text(json.dumps(assets,indent=2)+"\n")
    source=host_source(plan,assets,reference)
    if source.count('return 0;\n}')!=1:raise RuntimeError("host checker completion marker changed")
    source='#include "host_runtime.h"\n'+source.replace('return 0;\n}','mlx_clocked_pass();return 0;\n}')
    (out/"test.c").write_text(source);objects=[]
    for stem in (("command_blob",) if preload_assets else ("command_blob","payload_blob")):
        subprocess.run(["riscv64-unknown-elf-objcopy","-I","binary","-O","elf64-littleriscv","-B","riscv","--set-section-alignment",".data=8","--redefine-sym",f"_binary_{stem}_bin_start={stem}",f"{stem}.bin",f"{stem}.o"],cwd=out,capture_output=True,check=True,timeout=30);objects.append(str(out/f"{stem}.o"))
    elf=out/"test.elf"
    command=["riscv64-unknown-elf-gcc","-std=c11","-O2","-march=rv64imafd","-mabi=lp64d","-mcmodel=medany","-ffreestanding","-fno-builtin","-fno-tree-loop-distribute-patterns","-ffp-contract=off","-Wall","-Wextra","-Werror","-nostdlib","-static","-Wl,--no-relax","-DMLX_GRAPH_CLOCKED_ROCC=1","-I",str(HOST),"-I",str(BRIDGE),"-T",str(ROOT/"system_sim/native/link.ld"),str(HOST/"start.S"),str(HOST/"control_runtime.c"),str(HOST/"graph_runtime.c"),str(out/"test.c"),*objects,"-o",str(elf)]
    execute(command,out/"elf-build.log",120)
    symbols=subprocess.run(["riscv64-unknown-elf-nm",str(elf)],capture_output=True,text=True,check=True).stdout.splitlines()
    stack=[int(line.split()[0],16) for line in symbols if line.split()[-1]=="__stack_top"]
    if len(stack)!=1 or not 0x80000000<stack[0]<graph_base:raise RuntimeError("ELF/stack overlaps target graph buffers")
    image={"classification":"compiled_system_image_not_execution","elf_sha256":sha(elf),"commands_sha256":sha(out/"command_blob.bin"),"plan_sha256":sha(out/"plan.json"),"host_source_sha256":sha(out/"test.c"),"inputs":fingerprints,
        "source_calls":plan["source_calls"],"task_count":plan["task_count"],"asset_initialization":plan["asset_initialization"],"initial_asset_count":plan["initial_asset_count"],"initial_asset_bytes":plan["initial_asset_bytes"],"cpu_asset_copy_bytes":plan["cpu_asset_copy_bytes"],"full_model_execution_verified":False}
    (out/"image.json").write_text(json.dumps(image,indent=2)+"\n")
    return plan,fingerprints


def memory_map(chipyard,config):
    path=chipyard/f"sims/verilator/generated-src/chipyard.TestHarness.{config}/chipyard.TestHarness.{config}.dts";text=path.read_text()
    root=text[text.index("/ {")+3:];prefix=root[:root.index("{")]
    def cells(name):return int(re.search(r"#"+name+r"-cells\s*=\s*<([^>]+)>",prefix).group(1),0)
    node=re.search(r"memory@80000000\s*\{([^}]+)\}",text).group(1);values=[int(v,0) for v in re.search(r"reg\s*=\s*<([^>]+)>",node).group(1).split()]
    address_cells,size_cells=cells("address"),cells("size")
    if len(values)!=address_cells+size_cells:raise RuntimeError("unexpected actual memory device-tree encoding")
    def integer(words):
        result=0
        for word in words:result=result*2**32+word
        return result
    return {"base":integer(values[:address_cells]),"bytes":integer(values[address_cells:]),"device_tree_sha256":sha(path)}


def simulation_command(simulator,elf,preload=False,segments=None,max_cycles=20000000,seed=None):
    command=[str(simulator),f"+max-cycles={max_cycles}"]
    if seed is not None:command.extend(["-s",str(seed)])
    # This option is consumed in the memory/TSI C++ models, not registered as
    # an emulator option. Forward it through the existing permissive bracket.
    extra=[]
    if preload:extra.append(f"+loadmem={elf}")
    if segments is not None:extra.append(f"+mlx_memory_segments={segments}")
    if extra:command.extend(["+permissive",*extra,"+permissive-off"])
    return [*command,str(elf)]


def prepare_profile(requested,out):
    build=ROOT/"build/mlx-system-profile";input_hash=sha(requested) if requested is not None else None
    execute(["cmake","-S",str(PROFILE),"-B",str(build),"-DCMAKE_BUILD_TYPE=Release"],out/"profile-cmake.log",180)
    execute(["cmake","--build",str(build),"--target","system-profile-dump","-j4"],out/"profile-build.log",300)
    command=[str(build/"system-profile-dump")]+([str(requested)] if requested is not None else [])
    parsed=subprocess.run(command,capture_output=True,text=True,check=True,timeout=30);effective=json.loads(parsed.stdout)
    if requested is not None and sha(requested)!=input_hash:raise RuntimeError("requested profile changed while parsing")
    (out/"profile.json").write_text(json.dumps(effective,indent=2)+"\n")
    return effective,{"requested_file":str(requested) if requested else None,"requested_sha256":input_hash,"effective_sha256":sha(out/"profile.json"),"parser_sha256":sha(build/"system-profile-dump")}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--cases",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--chipyard",type=Path,default=ROOT/"build/chipyard-native");parser.add_argument("--phase",choices=("all","run","prepare"),default="all")
    parser.add_argument("--large-memory",action="store_true");parser.add_argument("--preload-elf",action="store_true");parser.add_argument("--graph-base",type=lambda x:int(x,0));parser.add_argument("--graph-bytes",type=int,default=1048576)
    parser.add_argument("--preload-assets",action="store_true");parser.add_argument("--device-profile",type=Path);parser.add_argument("--max-system-cycles",type=int,default=20000000);parser.add_argument("--watchdog-seconds",type=int,default=600)
    parser.add_argument("--progress",action="store_true");parser.add_argument("--progress-period",type=int,default=1000000)
    parser.add_argument("--seed",type=int)
    parser.add_argument("--block-pairs",action="store_true");parser.add_argument("--event-slots",type=int,default=32)
    args=parser.parse_args();out=args.output.resolve();chipyard=args.chipyard.resolve();config="MLXClockedLargeRocketConfig" if args.large_memory else CONFIG
    capacity=16*2**30 if args.large_memory else 256*2**20;graph_base=args.graph_base if args.graph_base is not None else 0x181000000 if args.large_memory else 0x81000000
    if args.preload_elf and not args.large_memory:raise RuntimeError("ELF binary preload requires the checked wide-memory profile")
    if args.preload_assets and not (args.large_memory and args.preload_elf):raise RuntimeError("resident assets require wide memory and explicit ELF initialization")
    if not 0<args.max_system_cycles<2**63 or args.watchdog_seconds<=0:raise RuntimeError("invalid system/watchdog limits")
    if not 0<args.progress_period<=10**12:raise RuntimeError("invalid system progress interval")
    if args.seed is not None and not 0<=args.seed<2**31:raise RuntimeError("invalid deterministic system seed")
    if out.exists():raise RuntimeError("choose a fresh clocked Chipyard attempt")
    out.mkdir(parents=True);sources=source_identity();cases=json.loads(args.cases.read_text());case_hash=sha(args.cases)
    record_attempt(out/"attempt.json",{"classification":"system_attempt_not_success_certificate","status":"preparing","runner_pid":os.getpid(),"sources":sources,"cases_sha256":case_hash,"phase":args.phase})
    snapshot_sources(ROOT,out/"sources",sources)
    if not isinstance(cases,list) or not cases:raise RuntimeError("actual system validation requires at least one case")
    profile,profile_identity=prepare_profile(args.device_profile.resolve() if args.device_profile else None,out)
    prepared={};case_options={}
    for case in cases:
        name=case["name"]
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in name) or name in prepared:raise RuntimeError("invalid/duplicate Chipyard case name")
        observe=case.get("progress",args.progress)
        if type(observe) is not bool:raise RuntimeError("case progress override must be boolean")
        equivalent=case.get("equivalent_to")
        if equivalent is not None and (equivalent not in prepared or args.seed is None):raise RuntimeError("equivalence requires an earlier case and a fixed seed")
        case_options[name]={"progress":observe,"equivalent_to":equivalent}
        prepared[name]=prepare(case,out/name,graph_base=graph_base,graph_bytes=args.graph_bytes,memory_bytes=capacity,preload_assets=args.preload_assets,block_pairs=case.get("block_pairs",args.block_pairs),event_slots=args.event_slots)
    if args.phase=="prepare":
        if sources!=source_identity() or case_hash!=sha(args.cases):raise RuntimeError("image preparation sources changed")
        for _,files in prepared.values():
            if any(sha(Path(path))!=digest for path,digest in files.items()):raise RuntimeError("image inputs changed during preparation")
        report={"classification":"system_images_prepared_not_executed","sources":sources,"profile":profile,"profile_identity":profile_identity,"cases":[{"case":name,"image_sha256":sha(out/name/"image.json"),"source_calls":plan["source_calls"],"task_count":plan["task_count"],"initial_asset_bytes":plan["initial_asset_bytes"]} for name,(plan,_) in prepared.items()],
            "max_system_cycles":args.max_system_cycles,"watchdog_seconds":args.watchdog_seconds,"full_model_execution_verified":False,"actual_rocket_execution":False,"mlx_system_verified":False,"inference_performance_eligible":False}
        (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
        record_attempt(out/"attempt.json",{"classification":"prepared_image_not_execution","status":"prepared","sources":sources,"report_sha256":sha(out/"report.json")})
        print(f"SYSTEM_MODEL_IMAGES_PREPARED {out/'report.json'}");return
    simulator=chipyard/f"sims/verilator/simulator-chipyard-{config}";build_manifest=chipyard/("clocked-wide-rocc-build.json" if args.large_memory else "clocked-rocc-build.json")
    if args.phase=="all":
        install(chipyard,args.large_memory)
        execute(["cmake","-S",str(BRIDGE),"-B",str(BUILD),"-DCMAKE_BUILD_TYPE=Release"],out/"cmake.log",180)
        execute(["cmake","--build",str(BUILD),"--target","mlx_clocked_rocc","-j4"],out/"native-build.log",300)
        if args.large_memory:
            execute(["cmake","-S",str(WIDE),"-B",str(WIDE_BUILD),"-DCMAKE_BUILD_TYPE=Release"],out/"wide-cmake.log",180)
            execute(["cmake","--build",str(WIDE_BUILD),"--target","mlx_wide_memory","-j4"],out/"wide-build.log",180)
        before=inputs(chipyard,args.large_memory);build_id=identity(before);header=chipyard/("clocked-wide-build-id.h" if args.large_memory else "clocked-rocc-build-id.h");contents=f'#define MLX_CLOCKED_BUILD_ID "{build_id}"\n'
        if not header.exists() or header.read_text()!=contents:header.write_text(contents)
        sbt=f"java -XX:ActiveProcessorCount=4 -Xmx6G -Dsbt.override.build.repos=true -Dsbt.repository.config={ROOT/'system_sim/native/sbt-repositories'} -Dsbt.sourcemode=true -Dsbt.workspace={chipyard/'tools'} -jar {ROOT/'build/toolchains/sbt/sbt-launch-1.4.9.jar'}"
        include=[BRIDGE,PROFILE,ROOT/"system_sim/clocked_device",ROOT/"system_sim/physical_device"]+[ROOT/"simulator_ext"/d for d in ("tensor_model","matrix_schedule","vector_model","vector_schedule","memory_model","model_io","control_model","control_schedule","tagged")]
        extra_sources=[BRIDGE/"dpi.cc"]
        if args.large_memory:
            include.extend([WIDE,chipyard/"generators/testchipip/src/main/resources/testchipip/csrc"]);extra_sources.append(WIDE/"memory_dpi.cc")
        json_cflags=subprocess.run(["pkg-config","--cflags","jsoncpp"],capture_output=True,text=True,check=True,timeout=15).stdout.strip()
        # This Verilator release appends its configured GNU++14 flag after
        # VM_USER_CFLAGS. Override that make variable too, without editing the
        # shared Verilator installation or downgrading the C++17 backend.
        command=["make","-C",str(chipyard/"sims/verilator"),f"CONFIG={config}","CFG_CXXFLAGS_STD_NEWEST=-std=gnu++17",f"RISCV={ROOT/'build/riscv-native'}",f"SBT={sbt}",
            "EXTRA_SIM_SOURCES="+" ".join(map(str,extra_sources)),"EXTRA_SIM_CXXFLAGS=-std=c++17 "+json_cflags+" "+" ".join(f"-I{p}" for p in include)+f" -include {header}",
            "EXTRA_SIM_REQS="+" ".join(map(str,[header,*extra_sources,*libraries(args.large_memory)])),"EXTRA_SIM_LDFLAGS="+" ".join(map(str,libraries(args.large_memory)))+" -ljsoncpp -ldl"+(" -lcrypto" if args.large_memory else ""),"-j4"]
        execute(command,out/"chipyard-build.log",3600)
        if before!=inputs(chipyard,args.large_memory):raise RuntimeError("clocked Chipyard build inputs changed")
        build_manifest.write_text(json.dumps({"inputs":before,"build_identity":build_id,"simulator_sha256":sha(simulator),"command":command},indent=2)+"\n")
    manifest=json.loads(build_manifest.read_text())
    if manifest["inputs"]!=inputs(chipyard,args.large_memory) or manifest["build_identity"]!=identity(manifest["inputs"]) or manifest["simulator_sha256"]!=sha(simulator):raise RuntimeError("clocked Chipyard build is stale")
    ram=memory_map(chipyard,config)
    if ram["base"]!=0x80000000 or ram["bytes"]!=capacity:raise RuntimeError("actual device tree does not match requested RAM capacity")
    owned=out/simulator.name;shutil.copy2(simulator,owned);results=[];device_results={};runtime_libraries=linked_libraries(owned)
    (out/"build.json").write_text(json.dumps(manifest,indent=2)+"\n")
    record_attempt(out/"attempt.json",{"classification":"system_attempt_not_success_certificate","status":"executing","runner_pid":os.getpid(),"sources":sources,"build_identity":manifest["build_identity"],"simulator_sha256":sha(owned),"cases_sha256":case_hash})
    for name,(plan,files) in prepared.items():
        observe=case_options[name]["progress"]
        directory=out/name;environment=os.environ.copy();environment["MLX_CLOCKED_REPORT"]=str(directory/"device.json")
        environment["MLX_CLOCKED_PROFILE"]=str(out/"profile.json")
        for key in ("MLX_CLOCKED_PROGRESS","MLX_CLOCKED_LAUNCH_MAP","MLX_CLOCKED_PROGRESS_PERIOD","MLX_WIDE_MEMORY_INIT_REPORT"):environment.pop(key,None)
        if observe:
            environment.update(MLX_CLOCKED_PROGRESS=str(directory/"progress.json"),MLX_CLOCKED_LAUNCH_MAP=str(directory/"launch-map.json"),MLX_CLOCKED_PROGRESS_PERIOD=str(args.progress_period))
        if args.large_memory:
            environment["MLX_WIDE_MEMORY_REPORT"]=str(directory/"memory.json")
            if observe:environment["MLX_WIDE_MEMORY_INIT_REPORT"]=str(directory/"memory-init.json")
        command=simulation_command(owned,directory/"test.elf",args.preload_elf,directory/"segments.json" if args.preload_assets else None,args.max_system_cycles,args.seed)
        print(f"CLOCKED_CHIPYARD executing {directory}",flush=True)
        try:
            execution=run_process(command,directory/"chipyard.log",directory/"execution.json",timeout=args.watchdog_seconds,env=environment,cwd=ROOT,
                metadata={"sources":sources,"inputs":files,"runtime_libraries":runtime_libraries,"build_identity":manifest["build_identity"],"simulator_sha256":sha(owned),"elf_sha256":sha(directory/"test.elf"),"profile_identity":profile_identity,"source_calls":plan["source_calls"],"task_count":plan["task_count"],"initial_asset_bytes":plan["initial_asset_bytes"],"progress":observe,"seed":args.seed})
        except Exception as error:
            record_attempt(out/"attempt.json",{"classification":"failed_attempt_not_success","status":"execution_failed","case":name,"execution_record":str(directory/"execution.json"),"error":str(error)});raise
        if "MLX_CLOCKED_CHAIN_PASS" not in (directory/"chipyard.log").read_text():raise RuntimeError("real Rocket ELF did not finish all result checks")
        device=json.loads((directory/"device.json").read_text());tasks=[t for t in plan["tasks"] if t["kind"] in (2,3)]
        if device["build_identity"]!=manifest["build_identity"] or device["frontend_error"] or device["cache_request_owned"] or device["cpu_response_pending"]:raise RuntimeError("clocked RoCC lifecycle/build check failed")
        if device["effective_profile"]!=profile:raise RuntimeError("actual system backend profile differs from the prepared profile")
        if observe and device["progress_observer"]["failed"]:raise RuntimeError("model completed but progress observer failed")
        if device["launches"]!=len(tasks) or len(device["windows"])!=len(tasks) or device["requests"]!=device["responses"]:raise RuntimeError("clocked graph did not execute/drain all device tasks")
        if plan.get("host_abi_version",1)==2:
            from mlxsim.model_system_evidence import check_pair_kernel,task_coverage
            model=json.loads(Path(next(c for c in cases if c["name"]==name)["program"]).read_text())
            _,values,_=task_coverage(model,plan)
        for ordinal,(task,window) in enumerate(zip(tasks,device["windows"])):
            if not window["done"] or window["error"] or not window["transport"]["idle"] or window["source_id"]!=ordinal or window["backend"]!=task["family"] or window["descriptor_bytes_fetched"]!=task["bytes"]:raise RuntimeError("clocked task/source routing failed")
            if task["kind"]==3:
                check_pair_kernel(model,task,window["kernel"],profile,values,plan["pair_event_slots"],ordinal+1)
        comparable={key:value for key,value in device.items() if key!="progress_observer"}
        equivalent=case_options[name]["equivalent_to"]
        if equivalent is not None and comparable!=device_results[equivalent]:raise RuntimeError("observation A/B changed target events or cycles")
        device_results[name]=comparable
        if any(sha(Path(path))!=digest for path,digest in files.items()):raise RuntimeError("graph inputs changed")
        memory=None
        if args.large_memory:
            memory=json.loads((directory/"memory.json").read_text())
            if memory["base"]!=ram["base"] or memory["bytes"]!=capacity or memory["max_read_address"]<graph_base:raise RuntimeError("high system addresses did not reach memory intact")
            if args.preload_elf and not memory["initialized_segments"]:raise RuntimeError("requested ELF preload did not initialize program segments")
            if args.preload_assets:
                segments=json.loads((directory/"segments.json").read_text());actual=[row for row in memory["initialized_segments"] if not row["name"].startswith("elf:")]
                if actual!=segments:raise RuntimeError("actual system resident input initialization differs from image")
        results.append({"case":name,"inputs":files,"source_calls":plan["source_calls"],"family_source_calls":plan["family_source_calls"],"device_windows":len(tasks),"elf_sha256":sha(directory/"test.elf"),"device_sha256":sha(directory/"device.json"),"log_sha256":sha(directory/"chipyard.log"),"plan_sha256":sha(directory/"plan.json"),"memory_sha256":sha(directory/"memory.json") if memory is not None else None,"asset_initialization":plan["asset_initialization"],"initial_asset_bytes":plan["initial_asset_bytes"],"cpu_asset_copy_bytes":plan["cpu_asset_copy_bytes"]})
        execution.update(validation="registered_graph_checks_passed");record_attempt(directory/"execution.json",execution)
    if sources!=source_identity() or case_hash!=sha(args.cases) or sha(owned)!=manifest["simulator_sha256"] or any(sha(Path(p))!=digest for p,digest in runtime_libraries.items()):raise RuntimeError("clocked run sources/binary/libraries changed")
    if sha(out/"profile.json")!=profile_identity["effective_sha256"]:raise RuntimeError("runtime profile file changed")
    report={"classification":"actual_rocket_cpp_tensor_graph_integration_not_full_model_or_performance_validation","sources":sources,"cases":results,"build":manifest,"simulator_sha256":sha(owned),
        "system_memory":ram,"graph_base":graph_base,"graph_bytes":args.graph_bytes,"program_initialization":"host_elf_preload_not_cpu_dma_time" if args.preload_elf else "normal_tsi_program_loading",
        "profile":profile,"profile_identity":profile_identity,"max_system_cycles":args.max_system_cycles,"watchdog_seconds":args.watchdog_seconds,
        "runtime_libraries":runtime_libraries,"case_observation":case_options,"seed":args.seed,
        "actual_rocket_execution":True,"host_completion_and_output_checks_in_elf":True,"full_model_execution_verified":False,"mlx_system_verified":False,"inference_performance_eligible":False}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    record_attempt(out/"attempt.json",{"classification":"system_attempt_not_full_model_certificate","status":"validated","report_sha256":sha(out/"report.json"),"sources":sources,"simulator_sha256":sha(owned)})
    print(f"CLOCKED_CHIPYARD_GRAPH_PASS {out/'report.json'}")


if __name__=="__main__":main()
