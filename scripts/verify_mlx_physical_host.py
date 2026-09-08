"""Execute actual RV64 host control C/loads/stores in Spike; not a model gate."""
import argparse
import copy
import hashlib
import json
import math
import re
import struct
import subprocess
from pathlib import Path

import numpy as np
from mlxsim.model_control_program import control_program
from system_sim.physical_host.lowering import lower_control

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"system_sim/physical_host"
DTYPE={"f16":0,"f32":1,"i64":2,"bool":3}
NUMPY={"f16":np.float16,"f32":np.float32,"i64":np.int64,"bool":np.uint8}
OP={"arange":1,"add":2,"mul":3,"le":4,"argmax":5}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor(array,dtype,shape=None,stride=None,offset=0):
    array=np.asarray(array,dtype=NUMPY[dtype]);shape=list(array.shape) if shape is None else shape
    if stride is None:
        stride=[];step=1
        for n in reversed(shape):stride.insert(0,step);step*=max(n,1)
    return {"kind":"tensor","dtype":dtype,"shape":shape,"stride":stride,"offset":offset,"raw":array.tobytes().hex()}


def scalar(value,dtype):
    raw=np.asarray(value,dtype=NUMPY[dtype]).tobytes()
    return {"kind":"scalar","dtype":dtype,"bits":int.from_bytes(raw,"little")}


def values(arg):
    if arg["kind"]=="scalar":return np.frombuffer(arg["bits"].to_bytes(np.dtype(NUMPY[arg["dtype"]]).itemsize,"little"),dtype=NUMPY[arg["dtype"]])[0]
    width=np.dtype(NUMPY[arg["dtype"]]).itemsize
    result=np.ndarray(arg["shape"],dtype=NUMPY[arg["dtype"]],buffer=bytes.fromhex(arg["raw"]),offset=arg["offset"]*width,strides=tuple(s*width for s in arg["stride"]))
    return result!=0 if arg["dtype"]=="bool" else result


def cases():
    result=[]
    def case(name,kind,a=None,b=None,extent=0,keep=False):
        integral=a is None or a["dtype"] in {"i64","bool"}
        if kind=="arange":expected=np.arange(extent,dtype=np.int64);dtype="i64"
        elif kind in {"add","mul"}:
            av=np.asarray(values(a),dtype=np.uint64);bv=np.asarray(values(b),dtype=np.uint64)
            expected=(av+bv if kind=="add" else av*bv).view(np.int64);dtype="i64"
        elif kind=="le":
            av,bv=values(a),values(b)
            if not integral:av=np.asarray(av,dtype=np.float32);bv=np.asarray(bv,dtype=np.float32)
            expected=np.asarray(av<=bv,dtype=np.uint8);dtype="bool"
        else:expected=np.argmax(values(a),axis=-1,keepdims=keep).astype(np.int64);dtype="i64"
        flags=0
        if not integral:
            if kind=="le":flags=16 if np.isnan(np.asarray(values(a),dtype=np.float32)).any() or np.isnan(np.asarray(values(b),dtype=np.float32)).any() else 0
            elif kind=="argmax" and np.asarray(values(a)).shape[-1]>1:flags=16 if np.isnan(values(a)).any() else 0
        result.append({"name":name,"kind":kind,"a":a,"b":b,"extent":extent,"flags":int(keep),"output_dtype":dtype,"output_shape":list(expected.shape),"expected":expected.tobytes().hex(),"status":0,"fflags":flags})
    for n in (0,1,65):case(f"arange-{n}","arange",extent=n)
    a=tensor([[2**63-1,-2**63,2**60+1],[0,-1,2**60+3]],"i64")
    b=tensor([[1],[-3]],"i64")
    for kind in ("add","mul","le"):case(f"integer-{kind}",kind,a,b)
    case("integer-scalar","add",a,scalar(7,"i64"))
    case("integer-argmax","argmax",a)
    case("integer-tie","argmax",tensor([[-7,-2,-2]],"i64"),keep=True)
    case("strided-int-add","add",tensor(np.arange(24).reshape(3,8),"i64",shape=[4,3],stride=[2,8],offset=1),scalar(-4,"i64"))
    case("bool-noncanonical","le",tensor([0,1,128,255],"bool"),scalar(0,"bool"))
    for dtype in ("f16","f32"):
        case(f"{dtype}-compare","le",tensor([[-1,0,1],[4,5,6]],dtype),tensor([[0],[5]],dtype))
        case(f"{dtype}-nan-compare","le",tensor([float("nan"),-float("inf"),0,float("inf")],dtype),scalar(0,"f32"))
        data=np.asarray([[1,4,4,-1],[2,float("nan"),float("nan"),8]],dtype=NUMPY[dtype])
        case(f"{dtype}-strided-argmax","argmax",tensor(data.T.copy(),dtype,shape=[2,4],stride=[1,2]),keep=True)
        case(f"{dtype}-one-column","argmax",tensor([[float("nan")],[4]],dtype))
    case("zero-rows","argmax",tensor(np.empty((0,3)),"f32"))
    case("empty-add","add",tensor(np.empty((0,3)),"i64"),scalar(1,"i64"))
    case("signed-zero","argmax",tensor([[-0.0,0.0,-0.0]],"f32"))
    base=copy.deepcopy(result[3])
    changes=[("magic",1,lambda c:c.update(command_overrides={"magic":"0"})),
        ("version",1,lambda c:c.update(command_overrides={"version":"2"})),
        ("opcode",6,lambda c:c.update(command_overrides={"opcode":"99"})),
        ("reserved",1,lambda c:c.update(command_overrides={"reserved":"1"})),
        ("unknown-flags",1,lambda c:c.update(command_overrides={"flags":"8"})),
        ("output-readonly",1,lambda c:c.update(output_overrides={"access":"1"})),
        ("misaligned-input",4,lambda c:c.update(a_overrides={"base":"BASE+1"})),
        ("input-overflow",4,lambda c:c.update(a_overrides={"base":"UINT64_MAX-7","bytes":"16"})),
        ("short-input",4,lambda c:c.update(a_overrides={"bytes":"8"})),
        ("rank",1,lambda c:c.update(a_overrides={"rank":"9"})),
        ("stride",4,lambda c:c.update(a_overrides={"stride":"{99,1}"})),
        ("overlap",5,lambda c:c.update(output_overrides={"base":"A_BASE"})),
        ("command-overlap",5,lambda c:c.update(output_overrides={"base":"COMMAND_BASE"})),
        ("output-dtype",2,lambda c:c.update(output_overrides={"dtype":"0"})),
        ("shape",3,lambda c:c.update(output_overrides={"shape":"{1,3}"})),
        ("bad-domain",2,lambda c:c.update(b=scalar(1.0,"f32")))]
    for name,status,change in changes:
        c=copy.deepcopy(base);c.update(name="reject-"+name,status=status,fflags=0);change(c);result.append(c)
    for dtype in ("f16","f32"):
        c=copy.deepcopy(next(c for c in result if c["name"]==f"{dtype}-compare"));c.update(name=f"reject-{dtype}-frm",frm=1,status=7,fflags=0);result.append(c)
    return result


def c_bytes(data):return "{"+",".join(f"0x{b:02x}" for b in data)+"}"
def fields(values):return "{"+",".join(f".{k}={v}" for k,v in values.items())+"}"


def generate(items):
    lines=['#include "control_runtime.h"','volatile uint64_t tohost __attribute__((section(".tohost"),aligned(64)))=0;','volatile uint64_t fromhost __attribute__((section(".tohost"),aligned(64)))=0;']
    commands=[];records=[]
    for index,c in enumerate(items):
        specs={};record={}
        for slot in ("a","b"):
            arg=c[slot]
            if arg is None:specs[slot]="{0}";record[slot+"_size"]="0";continue
            if arg["kind"]=="scalar":
                specs[slot]=fields({"kind":"MLX_HOST_SCALAR","scalar_dtype":str(DTYPE[arg["dtype"]]),"scalar_bits":f"UINT64_C({arg['bits']})"});record[slot+"_size"]="0";continue
            raw=bytes.fromhex(arg["raw"]);guard=b"\xa7"*8+raw+b"\xa7"*8;symbol=f"input_{slot}_{index}";expected=f"original_{slot}_{index}"
            lines += [f'static unsigned char {symbol}[] __attribute__((section(".high"),aligned(64)))={c_bytes(guard)};',f'static const unsigned char {expected}[]={c_bytes(guard)};']
            base=f"(uint64_t)(uintptr_t)({symbol}+8)"
            desc={"base":base,"bytes":str(len(raw)),"offset":str(arg["offset"]),"rank":str(len(arg["shape"])),"dtype":str(DTYPE[arg["dtype"]]),"access":"MLX_HOST_READ","shape":"{"+",".join(map(str,arg["shape"]))+"}","stride":"{"+",".join(map(str,arg["stride"]))+"}"}
            desc.update({k:v.replace("BASE",base) for k,v in c.get(slot+"_overrides",{}).items()})
            specs[slot]=fields({"kind":"MLX_HOST_TENSOR","tensor":fields(desc)})
            record.update({slot+"_data":f"(uintptr_t){symbol}",slot+"_expected":expected,slot+"_size":str(len(guard))})
        raw=bytes.fromhex(c["expected"]);symbol=f"output_{index}";expected=f"expected_{index}";guard=b"\xcd"*8+(raw if not c["status"] else b"\xcd"*len(raw))+b"\xcd"*8
        lines += [f'static unsigned char {symbol}[{len(guard)}] __attribute__((section(".high"),aligned(64)))={c_bytes(b"\xcd"*len(guard))};',f'static const unsigned char {expected}[]={c_bytes(guard)};']
        shape=c["output_shape"];stride=[];step=1
        for n in reversed(shape):stride.insert(0,step);step*=max(n,1)
        desc={"base":f"(uint64_t)(uintptr_t)({symbol}+8)","bytes":str(len(raw)),"rank":str(len(shape)),"dtype":str(DTYPE[c["output_dtype"]]),"access":"MLX_HOST_READ|MLX_HOST_WRITE","shape":"{"+",".join(map(str,shape))+"}","stride":"{"+",".join(map(str,stride))+"}"}
        desc.update({k:v.replace("A_BASE",f"(uint64_t)(uintptr_t)(input_a_{index}+8)").replace("COMMAND_BASE",f"(uint64_t)(uintptr_t)&commands[{index}]") for k,v in c.get("output_overrides",{}).items()})
        cmd={"magic":"MLX_HOST_CONTROL_MAGIC","version":"1","opcode":str(OP[c["kind"]]),"flags":str(c["flags"]),"extent":str(c["extent"]),"a":specs["a"],"b":specs["b"],"output":fields(desc)};cmd.update(c.get("command_overrides",{}));commands.append(fields(cmd))
        record.update({"output_data":f"(uintptr_t){symbol}","output_expected":expected,"output_size":str(len(guard)),"status":str(c["status"]),"fflags":str(c["fflags"]),"frm":str(c.get("frm",0))});records.append(fields(record))
    lines += ['static volatile mlx_host_control_command commands[]={'+','.join(commands)+'};',
        'struct record {uintptr_t a_data,b_data,output_data;const unsigned char *a_expected,*b_expected,*output_expected;uint64_t a_size,b_size,output_size,status,fflags,frm;};',
        'static const volatile struct record records[]={'+','.join(records)+'};']
    raw=np.arange(65536,dtype=np.uint16);floats=raw.view(np.float16).astype(np.float32).view(np.uint32);nan=((raw&0x7c00)==0x7c00)&((raw&1023)!=0)
    floats[nan]=((raw[nan].astype(np.uint32)&0x8000)<<16)|0x7fc00000
    lines += ['static const uint32_t half_expected[]={'+','.join(f'0x{int(x):08x}' for x in floats)+'};',
        'static int same(uintptr_t base,const unsigned char *expected,uint64_t n){const volatile unsigned char *p=(const volatile unsigned char *)base;for(uint64_t i=0;i<n;++i)if(p[i]!=expected[i])return 0;return 1;}',
        'int main(void){',
        f'for(unsigned i=0;i<{len(items)};++i){{',
        'const volatile struct record *r=&records[i];uint64_t fcsr=r->frm<<5;__asm__ volatile("fscsr %0"::"r"(fcsr):"memory");',
        'unsigned status=mlx_host_control_execute(&commands[i]);uint64_t flags;__asm__ volatile("frflags %0":"=r"(flags));__asm__ volatile("fscsr zero":::"memory");',
        'if(status!=r->status||flags!=r->fflags||!same(r->output_data,r->output_expected,r->output_size)||!same(r->a_data,r->a_expected,r->a_size)||!same(r->b_data,r->b_expected,r->b_size))return i+1;',
        '}',
        'for(unsigned h=0;h<65536;++h)if(mlx_host_half_to_float_bits((uint16_t)h)!=half_expected[h])return 200;',
        'return 0;','}']
    return "\n".join(lines)+"\n"


def serialized_commands_match(elf,items):
    listing=subprocess.run(["riscv64-unknown-elf-nm","-n",str(elf)],capture_output=True,text=True,check=True).stdout
    symbols={parts[2]:int(parts[0],16) for line in listing.splitlines() if len(parts:=line.split())==3 and re.fullmatch(r"[0-9a-fA-F]+",parts[0])}
    image=elf.read_bytes();header=struct.unpack_from("<16sHHIQQQIHHHHHH",image);offset,entry_size,count=header[5],header[9],header[10]
    segments=[struct.unpack_from("<IIQQQQQQ",image,offset+i*entry_size) for i in range(count)]
    def data_at(address,bytes):
        for kind,_,file_offset,virtual,_,size,_,_ in segments:
            if kind==1 and virtual<=address and address+bytes<=virtual+size:return image[file_offset+address-virtual:file_offset+address-virtual+bytes]
        raise RuntimeError("host command is not in a file-backed ELF segment")
    checked=0
    for index,case in enumerate(items):
        if case["status"]:continue
        layouts={};bindings={};args=[]
        for slot in ("a","b"):
            arg=case[slot]
            if arg is None:continue
            if arg["kind"]=="scalar":
                value=values(arg);args.append(bool(value) if arg["dtype"]=="bool" else int(value) if arg["dtype"]=="i64" else float(value));continue
            width=np.dtype(NUMPY[arg["dtype"]]).itemsize;size=len(bytes.fromhex(arg["raw"]))
            layouts[slot]={"root":slot,"dtype":arg["dtype"],"shape":arg["shape"],"strides":arg["stride"],"offset":arg["offset"],"storage_elements":size//width}
            bindings[slot]={"base":symbols[f"input_{slot}_{index}"]+8,"bytes":size,"writable":False};args.append({"value":slot})
        shape=case["output_shape"];strides=[];step=1
        for n in reversed(shape):strides.insert(0,step);step*=max(n,1)
        layouts["out"]={"root":"out","dtype":case["output_dtype"],"shape":shape,"strides":strides,"offset":0,"storage_elements":math.prod(shape)}
        bindings["out"]={"base":symbols[f"output_{index}"]+8,"bytes":len(bytes.fromhex(case["expected"])) ,"writable":True}
        if case["kind"]=="arange":args=[case["extent"]]
        elif case["kind"]=="argmax":args.extend([-1,bool(case["flags"])])
        domain="i64" if case["a"] is None or case["a"]["dtype"] in {"i64","bool"} else case["a"]["dtype"]
        node={"id":"out","kind":case["kind"],"args":args,"kwargs":{},"output":{"dtype":case["output_dtype"],"shape":shape},"source_operator_id":index,"control_program":control_program(case["kind"],domain)}
        encoded,_=lower_control(node,layouts,bindings)
        if encoded!=data_at(symbols["commands"]+index*640,640):raise RuntimeError(f"host binary lowering differs from C ABI: {case['name']}")
        checked+=1
    return checked


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True);parser.add_argument("--spike",type=Path,default=ROOT/"build/riscv-fesvr-build/spike");args=parser.parse_args();out=args.output.resolve()
    if out.exists():raise RuntimeError("choose a fresh physical host verification directory")
    out.mkdir(parents=True);source_files=[Path(__file__).resolve(),ROOT/"src/mlxsim/model_control_program.py",ROOT/"src/mlxsim/model_memory_program.py",*SOURCE.iterdir()];before={str(p.relative_to(ROOT)):sha(p) for p in source_files if p.is_file()}
    items=cases();(out/"cases.json").write_text(json.dumps(items,indent=2)+"\n");(out/"test.c").write_text(generate(items));elf=out/"test.elf"
    cmd=["riscv64-unknown-elf-gcc","-std=c11","-O2","-march=rv64imafd","-mabi=lp64d","-mcmodel=medany","-ffreestanding","-fno-builtin","-fno-tree-loop-distribute-patterns","-ffp-contract=off","-Wall","-Wextra","-Werror","-nostdlib","-static","-Wl,--no-relax","-I",str(SOURCE),"-T",str(SOURCE/"link.ld"),str(SOURCE/"start.S"),str(SOURCE/"control_runtime.c"),str(out/"test.c"),"-o",str(elf)]
    with (out/"build.log").open("w") as log:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=120)
    disassembly=subprocess.run(["riscv64-unknown-elf-objdump","-d",str(elf)],capture_output=True,text=True,check=True).stdout;(out/"disassembly.txt").write_text(disassembly)
    instructions=set(re.findall(r"\t([a-z][a-z0-9.]*)\s",disassembly))
    required={"ld","lwu","lhu","lbu","sd","sb","flt.s","fle.s","mul","fence"}
    if not required<=instructions:raise RuntimeError(f"missing real host instruction families: {required-instructions}")
    headers=subprocess.run(["riscv64-unknown-elf-readelf","-lW",str(elf)],capture_output=True,text=True,check=True).stdout;(out/"elf-segments.txt").write_text(headers)
    if "0x0000000100000000" not in headers:raise RuntimeError("host test did not bind >4GiB data segment")
    serialized=serialized_commands_match(elf,items)
    with (out/"spike.log").open("w") as log:result=subprocess.run([str(args.spike.resolve()),"--isa=RV64IMAFD","-m0x80000000:0x1000000,0x100000000:0x1000000",str(elf)],stdout=log,stderr=subprocess.STDOUT,timeout=120)
    if result.returncode:raise RuntimeError(f"actual host ELF failed (case/trap code {result.returncode}): {out/'spike.log'}")
    if before!={str(p.relative_to(ROOT)):sha(p) for p in source_files if p.is_file()}:raise RuntimeError("host runtime sources changed")
    report={"classification":"actual_rv64_host_control_elf_not_model_or_chipyard_validation","sources":before,"cases":len(items),"half_conversion_patterns":65536,
        "all_passed":True,"serialized_commands_match_c_abi":serialized,"case_manifest_sha256":sha(out/"cases.json"),"elf_sha256":sha(elf),"generated_source_sha256":sha(out/"test.c"),"spike_sha256":sha(args.spike.resolve()),
        "observed_instruction_families":sorted(required),"data_base":2**32,"host_load_store_and_loops_executed":True,
        "model_runner_integrated":False,"rocket_execution_verified":False,"mlx_system_verified":False,"inference_performance_eligible":False}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(f"PHYSICAL_HOST_RV64_ELF_PASS {out/'report.json'}")


if __name__=="__main__":main()
