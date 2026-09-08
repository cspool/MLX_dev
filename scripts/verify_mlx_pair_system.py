"""Run compiler-produced pair descriptors and actual RV64 control on Rocket."""
import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

import numpy as np

from mlxsim.model_control_program import control_program
from mlxsim.model_matrix_program import matrix_program
from mlxsim.model_vector_program import vector_program
from mlxsim.model_memory_program import make_layout
from system_sim.physical_device.pair_lowering import lower_pair
from system_sim.physical_host.lowering import lower_control
from scripts.run_mlx_clocked_chipyard import ROOT,BRIDGE,CONFIG,inputs,identity,source_identity,execute
from scripts.run_mlx_spike_graph import HOST,sha
from scripts.mlx_system_attempt import run_process,snapshot_sources,linked_libraries

BASE=0x81000000


def control_image(wire,token_address):
    descriptor=list(struct.unpack_from("<22Q",wire,256+4288+544))
    base,capacity,offset,rank,kind,_=descriptor[:6];shape=descriptor[6:6+rank];strides=descriptor[14:14+rank]
    dtype="f16" if kind==0 else "f32"
    layouts={"logits":{"root":"logits","dtype":dtype,"shape":shape,"strides":strides,"offset":offset,"storage_elements":capacity//(2 if kind==0 else 4)},
             "tokens":make_layout("i64",shape[:-1],"tokens")}
    bindings={"logits":{"base":base,"bytes":capacity,"writable":False},"tokens":{"base":token_address,"bytes":int(np.prod(shape[:-1]))*8,"writable":True}}
    node={"id":"tokens","kind":"argmax","args":[{"value":"logits"},-1],"kwargs":{},"output":{"dtype":"i64","shape":shape[:-1]},"control_program":control_program("argmax",dtype),"source_operator_id":2}
    image,_=lower_control(node,layouts,bindings)
    return image,shape,dtype


def generation(precision,perturbed):
    dtype=np.float16 if precision=="f16" else np.float32;width=np.dtype(dtype).itemsize
    weight=np.zeros((4,4),dtype=dtype);weight[1,0]=2;weight[2,1]=3;weight[3,2]=4;weight[0,3]=5
    if perturbed:weight[1,0]=0;weight[2,0]=6
    layouts={name:make_layout(precision,shape,name) for name,shape in (("a",[1,4]),("w",[4,4]),("p",[1,4]),("c",[1,4]))}
    bindings={name:{"base":BASE+65536+i*256,"bytes":int(np.prod(layout["shape"]))*width,"writable":name in {"p","c"}} for i,(name,layout) in enumerate(layouts.items())}
    p={"id":"p","kind":"linear","args":[{"value":"a"},{"value":"w"}],"output":{"dtype":precision,"shape":[1,4]},"matrix_program":matrix_program(precision,precision),"source_operator_id":0}
    c={"id":"c","kind":"mul","args":[{"value":"p"},2.0],"kwargs":{},"output":{"dtype":precision,"shape":[1,4]},"vector_program":vector_program("mul",[precision,"f32"],precision),"source_operator_id":1}
    wire,_=lower_pair(p,c,layouts,bindings,event_slots=2);table=np.eye(4,dtype=dtype);current=0;steps=[]
    for _ in range(3):
        a=table[current];first=(a.astype(np.float32)@weight.astype(np.float32).T).astype(dtype);second=(first.astype(np.float32)*2).astype(dtype);current=int(second.argmax())
        steps.append([first.tobytes(),second.tobytes()])
    job={"commands":[{"address":BASE+4096,"data":list(wire)}],"initial":[{"address":bindings["w"]["base"],"data":list(weight.tobytes())}],
         "outputs":[{"address":bindings[name]["base"],"bytes":4*width} for name in ("p","c")]}
    return job,steps,{"table":table.tobytes(),"rows":4,"row_bytes":4*width,"input_address":bindings["a"]["base"]}


def firmware(job,steps,feedback=None):
    wire=bytes(job["commands"][0]["data"]);command,shape,dtype=control_image(wire,BASE+0xF0000)
    token_rows=int(np.prod(shape[:-1]));expected_tokens=[]
    for _,raw in steps:expected_tokens.append(np.frombuffer(raw,dtype=np.float16 if dtype=="f16" else np.float32).reshape(token_rows,shape[-1]).argmax(-1).tolist())
    lines=['#include "host_runtime.h"','#include "control_runtime.h"',
           'volatile uint64_t tohost __attribute__((section(".tohost"),aligned(64)))=0;',
           'volatile uint64_t fromhost __attribute__((section(".tohost"),aligned(64)))=0;',
           'extern const unsigned char pair_image[],control_image[];',
           'static void copy(uint64_t address,const unsigned char *source,uint64_t bytes){volatile unsigned char *out=(volatile unsigned char *)(uintptr_t)address;for(uint64_t i=0;i<bytes;++i)out[i]=source[i];}',
           'static int equal(uint64_t address,const unsigned char *expected,uint64_t bytes){const volatile unsigned char *in=(const volatile unsigned char *)(uintptr_t)address;for(uint64_t i=0;i<bytes;++i)if(in[i]!=expected[i])return 0;return 1;}']
    for i,segment in enumerate(job["initial"]):lines.append(f'static const unsigned char initial_{i}[]={{'+','.join(map(str,segment["data"]))+'};')
    for i,step in enumerate(steps):
        for j,raw in enumerate(step):lines.append(f'static const unsigned char expected_{i}_{j}[]={{'+','.join(map(str,raw))+'};')
    lines.append('static const unsigned char *const expected[][2]={'+','.join('{expected_'+str(i)+'_0,expected_'+str(i)+'_1}' for i in range(len(steps)))+'};')
    lines.append('static const uint64_t expected_tokens[][ '+str(token_rows)+' ]={'+','.join('{'+','.join(map(str,row))+'}' for row in expected_tokens)+'};')
    if feedback:lines.append('static const unsigned char embeddings[]={'+','.join(map(str,feedback["table"]))+'};')
    lines.extend(['int main(void){','if(mlx_clocked_status(14)!=MLX_CLOCKED_ROCC_MAGIC)return 10;'])
    for i,segment in enumerate(job["initial"]):lines.append(f'copy(UINT64_C({segment["address"]}),initial_{i},sizeof(initial_{i}));')
    address=job["commands"][0]["address"];lines.append(f'copy(UINT64_C({address}),pair_image,{len(wire)});')
    if feedback:lines.append('uint64_t current=0;')
    lines.append(f'for(unsigned step=0;step<{len(steps)};++step){{')
    if feedback:
        lines.append(f'if(current>={feedback["rows"]})return 11;')
        lines.append(f'copy(UINT64_C({feedback["input_address"]}),embeddings+current*{feedback["row_bytes"]},{feedback["row_bytes"]});')
    lines.append(f'if(mlx_clocked_submit(UINT64_C({address}),{len(wire)})!=2)return 12;')
    for j,output in enumerate(job["outputs"]):lines.append(f'if(!equal(UINT64_C({output["address"]}),expected[step][{j}],{output["bytes"]}))return 13;')
    lines.append('if(mlx_host_control_execute((const volatile mlx_host_control_command *)(const void *)control_image))return 14;')
    lines.append(f'const volatile uint64_t *tokens=(const volatile uint64_t *)(uintptr_t)UINT64_C({BASE+0xF0000});')
    if feedback:lines.append('current=tokens[0];')
    lines.append(f'for(unsigned i=0;i<{token_rows};++i)if(tokens[i]!=expected_tokens[step][i])return 15;')
    lines.extend(['}','mlx_clocked_pass();return 0;','}'])
    return '\n'.join(lines)+'\n',wire,command,expected_tokens


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True);args=parser.parse_args();out=args.output.resolve();chipyard=ROOT/"build/chipyard-native"
    if out.exists():raise RuntimeError("choose a fresh pair-system attempt")
    manifest=json.loads((chipyard/"clocked-rocc-build.json").read_text());binary=chipyard/f"sims/verilator/simulator-chipyard-{CONFIG}"
    if manifest["inputs"]!=inputs(chipyard) or manifest["build_identity"]!=identity(manifest["inputs"]) or sha(binary)!=manifest["simulator_sha256"]:raise RuntimeError("pair system binary is stale")
    out.mkdir(parents=True);before=source_identity()
    for file in (Path(__file__).resolve(),ROOT/"tests/test_pair_wire.py",ROOT/"scripts/verify_mlx_pair_system.py"):before[str(file.relative_to(ROOT))]=sha(file)
    snapshot_sources(ROOT,out/"sources",before);owned=out/binary.name;shutil.copy2(binary,owned);runtime=linked_libraries(owned)
    sys.path.insert(0,str(ROOT/"tests"));import test_pair_wire as fixture
    original=fixture.BASE;fixture.BASE=BASE
    try:
        cases=[]
        for name,precision,batch,vector in (("matrix-f16","f16",False,False),("matrix-f32","f32",False,False),("batch-f32","f32",True,False),("mean-f16","f16",False,True)):
            job,expected,_=fixture.pair_job(precision,batches=batch,vector_producer=vector);cases.append((name,job,[[bytes(v) for v in expected]],None))
    finally:fixture.BASE=original
    for name,precision,perturb in (("generation-f16","f16",False),("generation-perturbed","f16",True),("generation-f32","f32",False)):
        job,steps,feedback=generation(precision,perturb);cases.append((name,job,steps,feedback))
    results=[]
    for name,job,steps,feedback in cases:
        directory=out/name;directory.mkdir();source,wire,control,tokens=firmware(job,steps,feedback)
        (directory/"test.c").write_text(source);(directory/"expected-tokens.json").write_text(json.dumps(tokens)+"\n");objects=[]
        for stem,data,symbol in (("pair",wire,"pair_image"),("control",control,"control_image")):
            (directory/f"{stem}.bin").write_bytes(data)
            subprocess.run(["riscv64-unknown-elf-objcopy","-I","binary","-O","elf64-littleriscv","-B","riscv","--set-section-alignment",".data=8","--redefine-sym",f"_binary_{stem}_bin_start={symbol}",f"{stem}.bin",f"{stem}.o"],cwd=directory,capture_output=True,check=True,timeout=30);objects.append(str(directory/f"{stem}.o"))
        command=["riscv64-unknown-elf-gcc","-std=c11","-O2","-march=rv64imafd","-mabi=lp64d","-mcmodel=medany","-ffreestanding","-fno-builtin","-fno-tree-loop-distribute-patterns","-ffp-contract=off","-Wall","-Wextra","-Werror","-nostdlib","-static","-Wl,--no-relax","-I",str(HOST),"-I",str(BRIDGE),"-T",str(ROOT/"system_sim/native/link.ld"),str(HOST/"start.S"),str(HOST/"control_runtime.c"),str(directory/"test.c"),*objects,"-o",str(directory/"test.elf")]
        execute(command,directory/"build.log",120)
        env=os.environ.copy()
        for key in ("MLX_CLOCKED_PROFILE","MLX_CLOCKED_PROGRESS","MLX_CLOCKED_LAUNCH_MAP","MLX_CLOCKED_PROGRESS_PERIOD"):env.pop(key,None)
        env["MLX_CLOCKED_REPORT"]=str(directory/"device.json")
        env["MLX_CLOCKED_PROGRESS"]=str(directory/"progress.json");env["MLX_CLOCKED_PROGRESS_PERIOD"]="100000"
        state=run_process([owned,"+max-cycles=20000000","-s","73129",directory/"test.elf"],directory/"chipyard.log",directory/"execution.json",timeout=900,env=env,cwd=ROOT,
                          metadata={"sources":before,"elf_sha256":sha(directory/"test.elf"),"simulator_sha256":sha(owned),"runtime_libraries":runtime,"build_identity":manifest["build_identity"]})
        report=json.loads((directory/"device.json").read_text())
        if (directory/"chipyard.log").read_text().splitlines().count("MLX_CLOCKED_CHAIN_PASS")!=1 or report["frontend_error"] or report["build_identity"]!=manifest["build_identity"]:raise RuntimeError("actual Rocket pair checks failed")
        if report["launches"]!=len(steps) or len(report["windows"])!=len(steps) or report["requests"]!=report["responses"] or report["cache_request_owned"] or report["cpu_response_pending"]:raise RuntimeError("pair system launch/response count differs")
        for ordinal,window in enumerate(report["windows"]):
            k=window["kernel"]
            if window["backend"]!="pair" or not window["done"] or window["error"] or not window["transport"]["idle"] or window["descriptor_bytes_fetched"]!=8832 or not k["array"]["template_configuration_cycles_modeled"] or not k["block_events"]["finished"] or k["block_events"]["epoch"]!=ordinal+1:raise RuntimeError("pair command did not complete all bounded execution state")
        results.append({"case":name,"pair_launches":len(steps),"expected_tokens":tokens,"actual_token_feedback":feedback is not None,"exit_code":state["exit_code"],"elf_sha256":sha(directory/"test.elf"),"source_sha256":sha(directory/"test.c"),"device_sha256":sha(directory/"device.json"),"log_sha256":sha(directory/"chipyard.log")})
    after=source_identity()
    for file in (Path(__file__).resolve(),ROOT/"tests/test_pair_wire.py",ROOT/"scripts/verify_mlx_pair_system.py"):after[str(file.relative_to(ROOT))]=sha(file)
    if before!=after or sha(owned)!=manifest["simulator_sha256"] or any(sha(Path(p))!=h for p,h in runtime.items()):raise RuntimeError("pair system sources/binary changed")
    report={"classification":"actual_rocket_pair_descriptor_and_cpu_feedback_integration_not_full_model_validation","sources":before,"build":manifest,"runtime_libraries":runtime,"cases":results,
            "actual_rocket_execution":True,"full_model_execution_verified":False,"mlx_system_verified":False,"inference_performance_eligible":False,"rtl_verified":False}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(f"PAIR_ROCKET_COMPONENT_CHECKS_PASS {out/'report.json'}")


if __name__=="__main__":main()
