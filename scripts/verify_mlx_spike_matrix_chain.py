"""Real RV64 control plus matrix backend in mapped plugin memory; not timing."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from mlxsim.model_matrix_program import matrix_program
from mlxsim.model_vector_program import vector_program
from mlxsim.model_memory_program import Planner,make_layout
from mlxsim.model_numeric_reference import NumericReferenceMode,pairwise_last
from system_sim.physical_device.lowering import lower_matrix_window
from system_sim.physical_device.vector_lowering import lower_vector
from system_sim.physical_device.memory_lowering import lower_memory
from scripts.run_mlx_physical_model import source_identity as model_sources

ROOT=Path(__file__).resolve().parents[1]
HOST=ROOT/"system_sim/physical_host"
DEVICE=ROOT/"system_sim/physical_device"
BUILD=ROOT/"build/mlx-spike-matrix"
SPIKE=ROOT/"build/riscv-fesvr-build/spike"
BASE=2**32


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def source_identity():
    result=model_sources();files=[Path(__file__).resolve()]
    files.extend([ROOT/"src/mlxsim/model_numeric_reference.py",ROOT/"src/mlxsim/model_matrix_reference.py"])
    for directory in (HOST,DEVICE):files.extend(p for p in directory.iterdir() if p.is_file())
    for path in files:result[str(path.relative_to(ROOT))]=sha(path)
    return result


def firmware(perturb=False,reject=None,recover=False,precision="f32",vector_stage=False,vector_kind="mul",memory_stage=False,memory_kind="embedding"):
    embedding=np.eye(4,dtype=np.float32);weight=np.zeros((4,4),dtype=np.float32)
    weight[1,0]=2;weight[2,1]=3;weight[3,2]=4;weight[0,3]=5
    if perturb:weight[1,0]=0;weight[2,0]=6
    dtype=np.float16 if precision=="f16" else np.float32;rawtype=np.uint16 if precision=="f16" else np.uint32
    width=np.dtype(dtype).itemsize;ctype="uint16_t" if precision=="f16" else "uint32_t";wire_dtype="MLX_HOST_F16" if precision=="f16" else "MLX_HOST_F32"
    embedding=embedding.astype(dtype);weight=weight.astype(dtype);current=0;expected=[];tokens=[]
    reference=NumericReferenceMode() if vector_stage and vector_kind=="softmax" else None
    for _ in range(3):
        selected=embedding[current]
        if memory_stage and memory_kind=="where" and current>1:selected=np.zeros(4,dtype=dtype)
        logits=selected@weight.T
        if vector_stage:
            if vector_kind=="mul":logits=(logits*np.asarray(0.5,dtype=dtype)).astype(dtype)
            else:
                shifted=np.subtract(logits.astype(np.float32),np.max(logits.astype(np.float32)),dtype=np.float32)
                exp=reference.unary("expf",shifted);logits=np.divide(exp,pairwise_last(exp),dtype=np.float32).astype(dtype)
        current=int(logits.argmax());expected.extend(logits.view(rawtype).tolist());tokens.append(current)
    program=matrix_program(precision,precision)
    layouts={"a":{"root":"a","dtype":precision,"shape":[1,4],"strides":[4,1],"offset":0,"storage_elements":4 if memory_stage else 16},
        "b":{"root":"b","dtype":precision,"shape":[4,4],"strides":[4,1],"offset":0,"storage_elements":16},
        "out":{"root":"out","dtype":precision,"shape":[1,4],"strides":[4,1],"offset":0,"storage_elements":4}}
    bindings={"a":{"base":BASE+(0x2400 if memory_stage else 0x2000),"bytes":(4 if memory_stage else 16)*width,"writable":False},"b":{"base":BASE+0x2040,"bytes":16*width,"writable":False},"out":{"base":BASE+0x2080,"bytes":4*width,"writable":True}}
    node={"id":"out","kind":"linear","args":[{"value":"a"},{"value":"b"}],"output":{"shape":[1,4],"dtype":precision},"matrix_program":program,"source_operator_id":0}
    wire,wire_route=lower_matrix_window(node,layouts,bindings)
    images={"matrix_wire":wire}
    if memory_stage:
        planner=Planner();planner.add_asset("table",{"dtype":precision,"shape":[4,4]});planner.add_asset("indices",{"dtype":"i64","shape":[1]})
        if memory_kind=="embedding":margs=[{"value":"table"},{"value":"indices"}]
        elif memory_kind=="where":
            planner.add_asset("predicate",{"dtype":"bool","shape":[1]});planner.layouts["row"]={**make_layout(precision,[1,4],"table"),"storage_elements":16};margs=[{"value":"predicate"},{"value":"row"},0.0]
        elif memory_kind=="cat":
            planner.layouts["left"]={**make_layout(precision,[1,2],"table"),"strides":[4,1],"storage_elements":16}
            planner.layouts["right"]={**planner.layouts["left"],"offset":2};margs=[[{"value":"left"},{"value":"right"}],1]
        else:raise ValueError("unsupported memory chain kind")
        mnode={"id":"out","kind":memory_kind,"args":margs,"kwargs":{},"output":{"dtype":precision,"shape":[1,4]},"source_operator_id":2}
        planner.register(mnode,planned=True)
        mbindings={"table":{"base":BASE+0x2000,"bytes":16*width,"writable":False},"indices":{"base":BASE+0x20a0,"bytes":8,"writable":False},"predicate":{"base":BASE+0x2300,"bytes":1,"writable":False},"out":{"base":BASE+0x2400,"bytes":4*width,"writable":True}}
        images["memory_wire"],_=lower_memory(mnode,planner.layouts,mbindings)
    if vector_stage:
        vlayouts={"a":{"root":"a","dtype":precision,"shape":[1,4],"strides":[4,1],"offset":0,"storage_elements":4},"out":{"root":"out","dtype":precision,"shape":[1,4],"strides":[4,1],"offset":0,"storage_elements":4}}
        vbindings={"a":{"base":BASE+0x2080,"bytes":4*width,"writable":False},"out":{"base":BASE+0x2200,"bytes":4*width,"writable":True}}
        vargs=[{"value":"a"},0.5] if vector_kind=="mul" else [{"value":"a"},-1,"torch.float16" if precision=="f16" else "torch.float32"]
        vnode={"id":"out","kind":vector_kind,"args":vargs,"kwargs":{},"output":{"dtype":precision,"shape":[1,4]},"source_operator_id":1,"vector_program":vector_program(vector_kind,[precision,"f32"] if vector_kind=="mul" else [precision],precision,width=4 if vector_kind=="softmax" else None)}
        images["vector_wire"],_=lower_vector(vnode,vlayouts,vbindings)
    cwords=lambda values:"{"+",".join(f"UINT64_C({int(v)})" for v in values)+"}"
    u32=lambda values:"{"+",".join(f"UINT32_C(0x{int(v):08x})" for v in values)+"}"
    initial_invalid={"magic":"image.magic=0;","reserved":"image.reserved[0]=1;","word":"image.words[0]|=UINT64_C(1)<<40;","uninitialized":""}.get(reject,"")
    restore="\n image.magic=MLX_MATRIX_WIRE_MAGIC;image.reserved[0]=0;image.words[0]&=UINT32_MAX;"
    steps=3 if reject is None or recover else 0
    text=f'''#include "matrix_wire.h"
#include "vector_wire.h"
#include "memory_wire.h"
volatile uint64_t tohost __attribute__((section(".tohost"),aligned(64)))=0;
volatile uint64_t fromhost __attribute__((section(".tohost"),aligned(64)))=0;
#define BASE UINT64_C({BASE})
#define DESCRIPTOR (BASE+0x1000)
#define EMBEDDING (BASE+0x2000)
#define WEIGHT (BASE+0x2040)
#define LOGITS (BASE+0x2080)
#define TOKEN (BASE+0x20a0)
#define NORMALIZED (BASE+0x2200)
#define VECTOR_DESCRIPTOR (BASE+0x8000)
#define MEMORY_DESCRIPTOR (BASE+0xa000)
#define PREDICATE (BASE+0x2300)
static const {ctype} embedding[16]={u32(embedding.view(rawtype).flatten())};
static const {ctype} weight[16]={u32(weight.view(rawtype).flatten())};
static const {ctype} expected[12]={u32(expected)};
static const uint64_t expected_tokens[3]={cwords(tokens)};
/* This symbol is linked from the Python compiler's binary wire image. */
extern volatile mlx_matrix_wire image;
{'extern volatile mlx_vector_wire vector_image;' if vector_stage else ''}
{'extern volatile mlx_memory_wire memory_image;' if memory_stage else ''}
static volatile mlx_host_control_command initialize={{.magic=MLX_HOST_CONTROL_MAGIC,.version=1,.opcode=MLX_HOST_ARANGE,.extent=1,
 .output={{.base=TOKEN,.bytes=8,.rank=1,.dtype=MLX_HOST_I64,.access=MLX_HOST_READ|MLX_HOST_WRITE,.shape={{1}},.stride={{1}}}} }};
static volatile mlx_host_control_command select_token={{.magic=MLX_HOST_CONTROL_MAGIC,.version=1,.opcode=MLX_HOST_ARGMAX,
 .a={{.kind=MLX_HOST_TENSOR,.tensor={{.base={'NORMALIZED' if vector_stage else 'LOGITS'},.bytes={4*width},.rank=2,.dtype={wire_dtype},.access=MLX_HOST_READ,.shape={{1,4}},.stride={{4,1}}}}}},
 .output={{.base=TOKEN,.bytes=8,.rank=1,.dtype=MLX_HOST_I64,.access=MLX_HOST_READ|MLX_HOST_WRITE,.shape={{1}},.stride={{1}}}} }};
{'static volatile mlx_host_control_command compare_token={.magic=MLX_HOST_CONTROL_MAGIC,.version=1,.opcode=MLX_HOST_LE,.a={.kind=MLX_HOST_TENSOR,.tensor={.base=TOKEN,.bytes=8,.rank=1,.dtype=MLX_HOST_I64,.access=MLX_HOST_READ,.shape={1},.stride={1}}},.b={.kind=MLX_HOST_SCALAR,.scalar_dtype=MLX_HOST_I64,.scalar_bits=1},.output={.base=PREDICATE,.bytes=1,.rank=1,.dtype=MLX_HOST_BOOL,.access=MLX_HOST_READ|MLX_HOST_WRITE,.shape={1},.stride={1}}};' if memory_stage and memory_kind=='where' else ''}
static void fence(void){{__asm__ volatile("fence iorw, iorw":::"memory");}}
static uint64_t submit(const volatile void *source,unsigned bytes,uint64_t descriptor){{
 const volatile unsigned char *src=source;volatile unsigned char *dst=(volatile unsigned char *)(uintptr_t)descriptor;
 for(unsigned i=0;i<bytes;++i)dst[i]=src[i];
 *(volatile uint64_t *)(uintptr_t)(BASE+MLX_MATRIX_REG_DESCRIPTOR)=descriptor;fence();
 *(volatile uint64_t *)(uintptr_t)(BASE+MLX_MATRIX_REG_LAUNCH)=1;fence();
 for(unsigned spin=0;spin<100000;++spin){{uint64_t status=*(volatile uint64_t *)(uintptr_t)(BASE+MLX_MATRIX_REG_STATUS);if(!(status&MLX_MATRIX_STATUS_BUSY)){{fence();return status;}}}}
 return UINT64_MAX;
}}
static uint64_t launch(void){{return submit(&image,sizeof(image),DESCRIPTOR);}}
int main(void){{
 (void)weight;
 if(*(volatile uint64_t *)(uintptr_t)(BASE+MLX_MATRIX_REG_ID)!=MLX_MATRIX_WIRE_MAGIC)return 10;
 for(unsigned i=0;i<16;++i)*(volatile {ctype} *)(uintptr_t)(EMBEDDING+i*{width})=embedding[i];
 {'/* Deliberately leave weights invalid. */' if reject=='uninitialized' else f'for(unsigned i=0;i<16;++i)*(volatile {ctype} *)(uintptr_t)(WEIGHT+i*{width})=weight[i];'}
 fence();
 if(mlx_host_control_execute(&initialize)!=MLX_HOST_OK)return 11;
 fence();
 {initial_invalid}
 {('if(launch()!=MLX_MATRIX_STATUS_ERROR)return 12;'+restore) if reject else ''}
 volatile unsigned steps={steps};
 for(unsigned step=0;step<steps;++step){{
   uint64_t current=*(volatile uint64_t *)(uintptr_t)TOKEN;if(current>=4)return 13;
   {'if(mlx_host_control_execute(&compare_token)!=MLX_HOST_OK){return 20;} memory_image.operands[1].tensor.offset=current*4;' if memory_stage and memory_kind=='where' else ''}
   {'memory_image.operands[0].tensor.offset=current*4; memory_image.operands[1].tensor.offset=current*4+2;' if memory_stage and memory_kind=='cat' else ''}
   {'if(submit(&memory_image,sizeof(memory_image),MEMORY_DESCRIPTOR)!=MLX_MATRIX_STATUS_DONE)return 19;' if memory_stage else ''}
   image.a.offset={'0' if memory_stage else 'current*4'};
   if(launch()!=MLX_MATRIX_STATUS_DONE)return 14;
   {'if(submit(&vector_image,sizeof(vector_image),VECTOR_DESCRIPTOR)!=MLX_MATRIX_STATUS_DONE)return 18;' if vector_stage else ''}
   if(mlx_host_control_execute(&select_token)!=MLX_HOST_OK)return 15;
   fence();
   uint64_t actual=*(volatile uint64_t *)(uintptr_t)TOKEN;if(actual!=expected_tokens[step])return 16;
   for(unsigned i=0;i<4;++i)if(*(volatile {ctype} *)(uintptr_t)({'NORMALIZED' if vector_stage else 'LOGITS'}+i*{width})!=expected[step*4+i])return 17;
 }}
 return 0;
}}
'''
    windows_per_step=1+int(vector_stage)+int(memory_stage)
    return text,{"tokens":tokens if steps else [],"successful_windows":steps*windows_per_step,"launches":steps*windows_per_step+bool(reject),"perturbed":perturb,"reject":reject,"recover":recover,"precision":precision,"vector_stage":vector_kind if vector_stage else None,"memory_stage":memory_kind if memory_stage else None,"wire_route":wire_route,"reference_primitives":reference.provenance() if reference else None},images


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True);parser.add_argument("--asan",action="store_true");args=parser.parse_args();out=args.output.resolve()
    if out.exists():raise RuntimeError("choose a fresh CPU/matrix chain directory")
    out.mkdir(parents=True);before=source_identity()
    build=ROOT/"build/mlx-spike-matrix-asan" if args.asan else BUILD
    configure=["cmake","-S",str(DEVICE),"-B",str(build),"-DCMAKE_BUILD_TYPE="+("Debug" if args.asan else "Release")]
    if args.asan:configure.append("-DCMAKE_CXX_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer")
    with (out/"build.log").open("w") as log:
        for cmd in (configure,["cmake","--build",str(build),"--target","mlx_spike_matrix","-j4"]):subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    environment=os.environ.copy()
    if args.asan:
        environment["LD_PRELOAD"]=subprocess.run(["c++","-print-file-name=libasan.so"],capture_output=True,text=True,check=True).stdout.strip()
        environment["ASAN_OPTIONS"]="detect_leaks=0:halt_on_error=1" # uninstrumented third-party Spike owns unrelated allocations
        environment["UBSAN_OPTIONS"]="halt_on_error=1:print_stacktrace=1"
    plugin=build/"libmlx_spike_matrix.so";results=[]
    variants=[("normal",{}),("perturbed",{"perturb":True}),("normal-f16",{"precision":"f16"}),("perturbed-f16",{"precision":"f16","perturb":True}),("reject-magic",{"reject":"magic"}),("reject-reserved",{"reject":"reserved"}),("reject-word",{"reject":"word"}),("reject-uninitialized",{"reject":"uninitialized"}),("recover",{"reject":"magic","recover":True})]
    variants.extend((f"vector-{precision}"+suffix,{"precision":precision,"vector_stage":True,"perturb":bool(suffix)}) for precision in ("f16","f32") for suffix in ("","-perturbed"))
    variants.extend((f"softmax-{precision}"+suffix,{"precision":precision,"vector_stage":True,"vector_kind":"softmax","perturb":bool(suffix)}) for precision in ("f16","f32") for suffix in ("","-perturbed"))
    variants.extend((f"memory-{precision}"+suffix,{"precision":precision,"memory_stage":True,"vector_stage":True,"vector_kind":"softmax","perturb":bool(suffix)}) for precision in ("f16","f32") for suffix in ("","-perturbed"))
    variants.extend((f"{kind}-{precision}"+suffix,{"precision":precision,"memory_stage":True,"memory_kind":kind,"vector_stage":True,"vector_kind":"softmax","perturb":bool(suffix)}) for kind in ("where","cat") for precision in ("f16","f32") for suffix in ("","-perturbed"))
    for name,options in variants:
        directory=out/name;directory.mkdir();source,expected,images=firmware(**options);(directory/"test.c").write_text(source);elf=directory/"test.elf";objects=[]
        for stem,wire in images.items():
            (directory/f"{stem}.bin").write_bytes(wire);symbol={"matrix_wire":"image","vector_wire":"vector_image","memory_wire":"memory_image"}[stem]
            subprocess.run(["riscv64-unknown-elf-objcopy","-I","binary","-O","elf64-littleriscv","-B","riscv","--set-section-alignment",".data=8","--redefine-sym",f"_binary_{stem}_bin_start={symbol}",f"{stem}.bin",f"{stem}.o"],cwd=directory,capture_output=True,check=True,timeout=30);objects.append(str(directory/f"{stem}.o"))
        command=["riscv64-unknown-elf-gcc","-std=c11","-O2","-march=rv64imafd","-mabi=lp64d","-mcmodel=medany","-ffreestanding","-fno-builtin","-fno-tree-loop-distribute-patterns","-ffp-contract=off","-Wall","-Wextra","-Werror","-nostdlib","-static","-Wl,--no-relax","-I",str(DEVICE),"-I",str(HOST),"-T",str(HOST/"link.ld"),str(HOST/"start.S"),str(HOST/"control_runtime.c"),str(directory/"test.c"),*objects,"-o",str(elf)]
        with (directory/"build.log").open("w") as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
        config={"base":BASE,"bytes":65536,"report":str(directory/"device.json"),"memory_latency":2,"matrix_options":{"rows":1,"columns":1,"trace":True,"trace_limit":100000},"vector_options":{"rows":1,"columns":1,"trace":True,"trace_limit":100000}}
        (directory/"plugin.json").write_text(json.dumps(config,indent=2)+"\n")
        with (directory/"spike.log").open("w") as log:process=subprocess.run([str(SPIKE),"--isa=RV64IMAFD","-m64",f"--extlib={plugin}",f"--device=mlx_matrix,0x100000000,{directory/'plugin.json'}",str(elf)],stdout=log,stderr=subprocess.STDOUT,timeout=120,env=environment)
        if process.returncode:raise RuntimeError(f"CPU/matrix ELF failed: {name}, code {process.returncode}: {directory/'spike.log'}")
        report=json.loads((directory/"device.json").read_text())
        if report["busy"] or not report["memory_idle"] or len(report["windows"])!=expected["successful_windows"] or report["launches"]!=expected["launches"]:raise RuntimeError("CPU/matrix lifecycle mismatch")
        if any(not w["done"] or w["dma_requests"]!=w["dma_responses"] for w in report["windows"]):raise RuntimeError("matrix backend did not drain")
        if not options.get("reject") or options.get("recover"):
            if report["error"] or report["device_reads"]+report["device_writes"]!=sum(w["dma_requests"] for w in report["windows"]):raise RuntimeError("CPU/matrix data request accounting mismatch")
        elif not report["error"]:raise RuntimeError("invalid launch did not expose an error")
        results.append({"case":name,**expected,"elf_sha256":sha(elf),"wire_sha256":{stem:sha(directory/f"{stem}.bin") for stem in images},"source_sha256":sha(directory/"test.c"),"plugin_config_sha256":sha(directory/"plugin.json"),"device_report_sha256":sha(directory/"device.json"),"device_reads":report["device_reads"],"device_writes":report["device_writes"]})
    if before!=source_identity():raise RuntimeError("CPU/matrix chain sources changed")
    report={"classification":"actual_cpu_matrix_control_chain_not_full_model_or_chipyard_timing","sources":before,"cases":results,"plugin_sha256":sha(plugin),"spike_sha256":sha(SPIKE),"host_model_data_flow_executed":True,
        "poll_driven_accelerator_clock":True,"asan_ubsan":args.asan,"leak_checking":False if args.asan else None,"full_model_verified":False,"rocket_execution_verified":False,"mlx_system_verified":False,"inference_performance_eligible":False}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(f"SPIKE_HOST_MATRIX_CHAIN_PASS {out/'report.json'}")


if __name__=="__main__":main()
