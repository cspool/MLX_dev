"""Run a compiler-generated graph with actual RV64 host and device wire tasks."""
import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import time
from pathlib import Path

from scripts.verify_mlx_spike_matrix_chain import ROOT,HOST,DEVICE,SPIKE,source_identity as bridge_sources
from system_sim.physical_host.graph_lowering import compile_graph,write_payload
from system_sim.physical_host.asset_source import SOURCE_BASE,source_manifest


def sha(path):
    with path.open("rb") as input:return hashlib.file_digest(input,"sha256").hexdigest()


def check_embedded_payload_limit(program,command_bytes):
    # The device aperture is sparse, but this runner still links input assets
    # into the low 64-MiB CPU ELF. Reject large inputs before reading/copying them.
    payload_bytes=0
    for _,asset in sorted(program["assets"].items()):
        payload_bytes+=(-payload_bytes)%8
        payload_bytes+=math.prod(asset["shape"])*{"f16":2,"f32":4,"i64":8,"bool":1}[asset["dtype"]]
    if command_bytes+payload_bytes>48*2**20:
        raise RuntimeError("embedded ELF commands/assets exceed 48MiB staging limit; sparse device capacity is not a full-model asset loader")
    return payload_bytes


def host_source(plan,assets,reference):
    checks=[];lines=['#include "graph_runtime.h"','volatile uint64_t tohost __attribute__((section(".tohost"),aligned(64)))=0;','volatile uint64_t fromhost __attribute__((section(".tohost"),aligned(64)))=0;',
        'extern const unsigned char command_blob[],payload_blob[];',f'static uint64_t completion[{max(plan["source_calls"],1)}];','static volatile mlx_graph_result result;']
    fields=lambda value:'{'+','.join(f'.{name}={item}' for name,item in value.items())+'}'
    rows=[]
    for asset in assets:
        source=f"UINT64_C({asset['source_address']})" if "source_address" in asset else f"(uintptr_t)(payload_blob+{asset['offset']})"
        binding=plan["assets"][asset["value"]];rows.append(fields({"source":source,"destination":f"UINT64_C({binding['base']})","bytes":asset["bytes"]}))
    lines.append('static const volatile mlx_graph_asset assets[]={'+','.join(rows or ['{0}'])+'};')
    rows=[]
    for task in plan["tasks"]:
        row={key:task[key] for key in ("kind","source_ordinal","source_id","batch_index","batch_count","bytes")};row["command"]=f"(uintptr_t)(command_blob+{task['command_offset']})" if task["bytes"] else 0;rows.append(fields(row))
    lines.append('static const volatile mlx_graph_task tasks[]={'+','.join(rows or ['{0}'])+'};')
    p={"magic":"MLX_GRAPH_MAGIC","version":1,"source_count":plan["source_calls"],"task_count":plan["task_count"],"asset_count":len(assets),"assets":"(uintptr_t)assets","tasks":"(uintptr_t)tasks",
       "device_base":f"UINT64_C({plan['device_base']})","device_bytes":f"UINT64_C({plan['device_bytes']})","scratch_offset":plan["scratch_offset"],"scratch_bytes":plan["scratch_bytes"],"poll_limit":"UINT64_C(1000000000000)","completion":"(uintptr_t)completion"}
    lines.append('static const volatile mlx_graph_program program='+fields(p)+';')
    expected={row["forward_id"]:row for row in reference["outputs"]}
    for index,output in enumerate(plan["outputs"]):
        ref=expected[output["forward_id"]];layout=output["layout"]
        raw=Path(ref["logits_file"]).read_bytes() if output["role"]=="logits" else struct.pack('<'+'q'*len(ref["tokens"]),*ref["tokens"])
        width={"f16":2,"f32":4,"i64":8,"bool":1}[layout["dtype"]]
        count=math.prod(layout["shape"])
        if len(raw)!=count*width:raise RuntimeError("graph result reference byte count mismatch")
        if output["role"]=="logits" and (ref["shape"]!=layout["shape"] or ref["dtype"]!=layout["dtype"]):raise RuntimeError("graph logits reference type/shape differs")
        lines.append(f'static const unsigned char expected_{index}[]={{'+','.join(map(str,raw or b'\x00'))+'};')
        checks.append(fields({"base":f"UINT64_C({output['binding']['base']})","offset":layout["offset"],"rank":len(layout["shape"]),"width":width,"count":count,"shape":'{'+','.join(map(str,layout["shape"]))+'}',"stride":'{'+','.join(map(str,layout["strides"]))+'}',"expected":f'expected_{index}'}))
    lines += ['struct check {uint64_t base,offset,rank,width,count,shape[8],stride[8];const unsigned char *expected;};',
        'static const volatile struct check checks[]={'+','.join(checks or ['{0}'])+'};',
        *(['static const uint64_t source_ids[]={'+','.join(str(row["source_operator_id"]+1) for row in plan["sources"])+'};'] if plan["sources"] else []),
        'int main(void){',
        'if(mlx_graph_execute(&program,&result))return 10;',
        f'if(result.status||result.completed_sources!={plan["source_calls"]}||result.host_calls!={plan["family_source_calls"]["control"]}||result.view_elisions!={plan["family_source_calls"]["view"]})return 11;',
        f'if(result.device_calls!={sum(t["kind"]==2 for t in plan["tasks"])}||result.asset_bytes!={sum(a["bytes"] for a in assets)})return 14;',
        *([f'for(unsigned i=0;i<{plan["source_calls"]};++i)if(completion[i]!=source_ids[i])return 12;'] if plan["source_calls"] else []),
        f'for(unsigned row=0;row<sizeof(checks)/sizeof(checks[0]);++row){{',
        'const volatile struct check *c=&checks[row];for(uint64_t flat=0;flat<c->count;++flat){uint64_t index=flat,at=c->offset;for(unsigned d=(unsigned)c->rank;d-->0;){at+=(index%c->shape[d])*c->stride[d];index/=c->shape[d];}const volatile unsigned char *actual=(const volatile unsigned char *)(uintptr_t)(c->base+at*c->width);for(unsigned b=0;b<c->width;++b)if(actual[b]!=c->expected[flat*c->width+b])return 13;}',
        '}','return 0;','}']
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--program",type=Path,required=True);parser.add_argument("--lifetimes",type=Path,required=True);parser.add_argument("--reference",type=Path);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device-bytes",type=int,default=1048576);parser.add_argument("--data-offset",type=int,default=65536);parser.add_argument("--plan-only",action="store_true");parser.add_argument("--asan",action="store_true")
    parser.add_argument("--asset-source",choices=("embedded","files"),default="embedded");parser.add_argument("--source-offset",type=int,default=4096)
    parser.add_argument("--load-only",action="store_true");parser.add_argument("--timeout",type=int,default=300);args=parser.parse_args();out=args.output.resolve()
    if out.exists():raise RuntimeError("choose a fresh generic Spike graph directory")
    if args.timeout<=0:raise RuntimeError("positive graph timeout required")
    if args.load_only and (args.plan_only or args.asset_source!="files"):raise RuntimeError("load-only requires file assets and is not plan-only")
    if not args.plan_only and not args.load_only and args.reference is None:raise RuntimeError("actual graph execution requires an independent result reference")
    out.mkdir(parents=True);program=json.loads(args.program.read_text());life=json.loads(args.lifetimes.read_text())
    def identity():
        result=bridge_sources();result[str(Path(__file__).resolve().relative_to(ROOT))]=sha(Path(__file__).resolve());return result
    sources=identity();inputs={str(p.resolve()):sha(p) for p in (args.program,args.lifetimes)}
    blob,plan=compile_graph(program,life,device_bytes=args.device_bytes,data_offset=args.data_offset);(out/"plan.json").write_text(json.dumps(plan,indent=2)+"\n")
    (out/"command_blob.bin").write_bytes(blob if not args.load_only else bytes(8))
    if args.plan_only:
        report={"classification":"generic_host_graph_plan_not_execution","sources":sources,"inputs":inputs,"source_calls":plan["source_calls"],"task_count":plan["task_count"],"command_bytes":len(blob),"plan_sha256":sha(out/"plan.json"),"command_sha256":sha(out/"command_blob.bin"),"full_model_execution_verified":False,"mlx_system_verified":False}
        (out/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(f"GENERIC_GRAPH_PLAN_ONLY {out/'report.json'}");return
    if args.asset_source=="files" and plan["device_base"]+plan["device_bytes"]>SOURCE_BASE:raise RuntimeError("device aperture overlaps asset source bank")
    check_embedded_payload_limit(program if args.asset_source=="embedded" else {"assets":{}},0 if args.load_only else len(blob))
    reference={"outputs":[]}
    if not args.load_only:
        inputs[str(args.reference.resolve())]=sha(args.reference);reference=json.loads(args.reference.read_text())
    for row in reference["outputs"]:inputs[str(Path(row["logits_file"]).resolve())]=sha(Path(row["logits_file"]))
    for asset in program["assets"].values():
        if asset["kind"]=="mapped_file":
            path=Path(asset["path"]).resolve();name=str(path)
            if name not in inputs:inputs[name]=sha(path)
            if asset.get("file_sha256",inputs[name])!=inputs[name]:raise RuntimeError("asset file differs from compiled checkpoint identity")
    source_config=None
    if args.asset_source=="files":
        assets,source_config=source_manifest(program,out,source_offset=args.source_offset);source_config["report"]=str(out/"asset-source.json")
        (out/"asset-source-config.json").write_text(json.dumps(source_config,indent=2)+"\n")
        for path in (out/"asset-source-config.json",out/"literal_assets.bin"):inputs[str(path)]=sha(path)
    else:assets=write_payload(program,out/"payload_blob.bin")
    execution_plan=copy.deepcopy(plan)
    if args.load_only:
        execution_plan.update(source_calls=0,task_count=0,tasks=[],sources=[],outputs=[],family_source_calls={key:0 for key in plan["family_source_calls"]})
    (out/"assets.json").write_text(json.dumps(assets,indent=2)+"\n");(out/"test.c").write_text(host_source(execution_plan,assets,reference))
    objects=[]
    for stem in (("command_blob",) if source_config is not None else ("command_blob","payload_blob")):
        subprocess.run(["riscv64-unknown-elf-objcopy","-I","binary","-O","elf64-littleriscv","-B","riscv","--set-section-alignment",".data=8","--redefine-sym",f"_binary_{stem}_bin_start={stem}",f"{stem}.bin",f"{stem}.o"],cwd=out,capture_output=True,check=True,timeout=30);objects.append(str(out/f"{stem}.o"))
    elf=out/"test.elf";command=["riscv64-unknown-elf-gcc","-std=c11","-O2","-march=rv64imafd","-mabi=lp64d","-mcmodel=medany","-ffreestanding","-fno-builtin","-fno-tree-loop-distribute-patterns","-ffp-contract=off","-Wall","-Wextra","-Werror","-nostdlib","-static","-Wl,--no-relax","-I",str(HOST),"-T",str(HOST/"link.ld"),str(HOST/"start.S"),str(HOST/"control_runtime.c"),str(HOST/"graph_runtime.c"),str(out/"test.c"),*objects,"-o",str(elf)]
    with (out/"elf-build.log").open("w") as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
    symbols=subprocess.run(["riscv64-unknown-elf-nm",str(elf)],capture_output=True,text=True,check=True,timeout=30).stdout.splitlines()
    stack=[int(line.split()[0],16) for line in symbols if line.split()[-1]=="__stack_top"]
    if len(stack)!=1 or not 0x80000000<stack[0]<=0x84000000:raise RuntimeError("graph ELF plus stack exceeds configured 64MiB CPU memory")
    build=ROOT/("build/mlx-spike-matrix-asan" if args.asan else "build/mlx-spike-matrix");configure=["cmake","-S",str(DEVICE),"-B",str(build),"-DCMAKE_BUILD_TYPE="+("Debug" if args.asan else "Release")]
    if args.asan:configure.append("-DCMAKE_CXX_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer")
    with (out/"plugin-build.log").open("w") as log:
        for command in (configure,["cmake","--build",str(build),"--target","mlx_spike_matrix","-j4"]):subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    plugin=out/"libmlx_spike_matrix.so";shutil.copy2(build/"libmlx_spike_matrix.so",plugin)
    config={"base":plan["device_base"],"bytes":plan["device_bytes"],"report":str(out/"device.json"),"memory_latency":2,"matrix_options":{"rows":1,"columns":1,"trace":False},"vector_options":{"rows":1,"columns":1,"trace":False},"memory_options":{"trace":False}}
    if source_config is not None:config["loaded_assets"]=[{"value":a["value"],"base":plan["assets"][a["value"]]["base"],"bytes":a["bytes"],"sha256":a["sha256"]} for a in assets]
    (out/"plugin.json").write_text(json.dumps(config,indent=2)+"\n");environment=os.environ.copy()
    if args.asan:
        environment["LD_PRELOAD"]=subprocess.run(["c++","-print-file-name=libasan.so"],capture_output=True,text=True,check=True).stdout.strip();environment["ASAN_OPTIONS"]="detect_leaks=0:halt_on_error=1";environment["UBSAN_OPTIONS"]="halt_on_error=1:print_stacktrace=1"
    command=[str(SPIKE),"--isa=RV64IMAFD","-m64",f"--extlib={plugin}",f"--device=mlx_matrix,0x100000000,{out/'plugin.json'}"]
    if source_config is not None:command.append(f"--device=mlx_asset_source,{hex(SOURCE_BASE)},{out/'asset-source-config.json'}")
    command.append(str(elf))
    execution={"classification":"spike_graph_or_loader_attempt_not_success_certificate","sources":sources,"inputs":inputs,"command":command,"runner_pid":os.getpid(),"plugin_sha256":sha(plugin),"spike_sha256":sha(SPIKE),"elf_sha256":sha(elf),"timeout_seconds":args.timeout,"load_only":args.load_only,"status":"starting"}
    def record_execution():(out/"execution.json").write_text(json.dumps(execution,indent=2)+"\n")
    record_execution();started=time.monotonic()
    with (out/"spike.log").open("w") as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=environment)
        execution.update(status="running",pid=process.pid);record_execution()
        try:process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            process.kill();process.wait();execution.update(status="watchdog",exit_code=process.returncode,host_elapsed_seconds=time.monotonic()-started);record_execution();raise
    execution.update(status="exited",exit_code=process.returncode,host_elapsed_seconds=time.monotonic()-started);record_execution()
    if process.returncode:raise RuntimeError(f"generic graph ELF failed code {process.returncode}: {out/'spike.log'}")
    device=json.loads((out/"device.json").read_text());expected=sum(t["kind"]==2 for t in execution_plan["tasks"])
    if device["busy"] or not device["memory_idle"] or device["error"] or device["launches"]!=expected or len(device["windows"])!=expected:raise RuntimeError("generic graph device windows did not complete")
    if any(not w["done"] or w["dma_requests"]!=w["dma_responses"] for w in device["windows"]):raise RuntimeError("generic graph device requests did not drain")
    if source_config is not None:
        source=json.loads((out/"asset-source.json").read_text());audit=device["asset_load_audit"]
        if source["failed_reads"] or source["rejected_writes"] or not source["files_unchanged"]:raise RuntimeError("asset source failed or changed")
        if len(source["regions"])!=len(assets) or any(not row["sequential"] or row["read_bytes"]!=row["bytes"] or row["sequential_bytes"]!=row["bytes"] for row in source["regions"]):raise RuntimeError("CPU did not read every asset byte exactly once")
        if not audit["attempted"] or audit["error"] or audit["assets"]!=config["loaded_assets"]:raise RuntimeError("CPU destination asset readback differs from input")
        if args.load_only and (device["cpu_payload_write_bytes"]!=sum(a["bytes"] for a in assets) or device["host_memory_backing"]["writer_pages"]):raise RuntimeError("load-only used non-CPU writes or extra data")
    if sources!=identity() or any(sha(Path(p))!=digest for p,digest in inputs.items()) or sha(plugin)!=execution["plugin_sha256"] or sha(SPIKE)!=execution["spike_sha256"]:raise RuntimeError("generic graph sources/inputs/binaries changed")
    report={"classification":"actual_cpu_asset_loading_only_not_inference" if args.load_only else "compiler_generated_rv64_graph_execution_not_full_model_or_chipyard_validation","sources":sources,"inputs":inputs,"source_calls":execution_plan["source_calls"],"task_count":execution_plan["task_count"],"family_source_calls":execution_plan["family_source_calls"],"device_windows":expected,"compiled_source_calls":plan["source_calls"],
        "elf_sha256":sha(elf),"plugin_sha256":sha(plugin),"plan_sha256":sha(out/"plan.json"),"commands_sha256":sha(out/"command_blob.bin"),"payload_sha256":sha(out/"payload_blob.bin") if source_config is None else None,"assets_sha256":sha(out/"assets.json"),"generated_host_source_sha256":sha(out/"test.c"),"device_sha256":sha(out/"device.json"),"actual_cpu_dispatch":True,"output_bytes_checked_in_elf":not args.load_only,"asan_ubsan":args.asan,
        "device_capacity_bytes":plan["device_bytes"],"data_offset":plan["data_offset"],"host_memory_backing":device["host_memory_backing"],"asset_loading":"actual_cpu_load_from_read_only_file_aperture_and_store_to_device" if source_config is not None else "actual_cpu_copy_from_embedded_small_elf_not_full_model_loader",
        "all_assets_loaded_and_digest_checked":source_config is not None,"loaded_asset_bytes":sum(a["bytes"] for a in assets),"asset_source_report_sha256":sha(out/"asset-source.json") if source_config is not None else None,"load_only":args.load_only,
        "full_model_execution_verified":False,"rocket_execution_verified":False,"mlx_system_verified":False,"inference_performance_eligible":False}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(f"GENERIC_RV64_GRAPH_EXECUTION_PASS {out/'report.json'}")


if __name__=="__main__":main()
