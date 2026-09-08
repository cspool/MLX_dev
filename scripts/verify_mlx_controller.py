"""Differentially check RV64 controller leaves against independent Spike ISA execution.

The memory/loop binding of model control operators is still not a Rocket ELF.
"""

import argparse
import json
import random
import subprocess
from pathlib import Path

from mlxsim.model_control_program import r, i, f, beq
from scripts.run_mlx_tensor_semantics import ROOT, sha

MASK = 2**64 - 1


def int_to_f32(value):
    if not value:
        return 0, 0
    sign, magnitude = int(value < 0), abs(value)
    exponent = magnitude.bit_length() - 1
    remainder = 0
    if exponent <= 23:
        mantissa = magnitude << (23 - exponent)
    else:
        shift = exponent - 23
        mantissa, remainder = magnitude >> shift, magnitude & ((1 << shift) - 1)
        midpoint = 1 << (shift - 1)
        if remainder > midpoint or (remainder == midpoint and mantissa & 1):
            mantissa += 1
            if mantissa == 1 << 24:
                mantissa >>= 1
                exponent += 1
    return sign << 31 | (exponent + 127) << 23 | (mantissa & 0x7FFFFF), int(bool(remainder))


def cases():
    result = []
    rng = random.Random(812)
    for _ in range(40):
        a, b = rng.randrange(-2**63, 2**63), rng.randrange(-2**63, 2**63)
        for name, word, answer in [("add", r(0,0,12,10,11), (a+b)&MASK), ("mul", r(1,0,12,10,11), (a*b)&MASK), ("slt", r(0,2,12,10,11), int(a<b)), ("sltu", r(0,3,12,10,11), int((a&MASK)<(b&MASK)))]:
            result.append({"name": name, "words": [word], "x": {"10": a&MASK, "11": b&MASK}, "f": {}, "expect_x": {"12": answer}, "expect_f": {}, "fflags": 0})
    classes = [(0xFF800000,1),(0xBF800000,2),(0x80000001,4),(0x80000000,8),(0,16),(1,32),(0x3F800000,64),(0x7F800000,128),(0x7F800001,256),(0x7FC00123,512)]
    for raw, classification in classes:
        result.append({"name": "fclass", "words": [f(0x70,1,12,10)], "x": {}, "f": {"10": 0xFFFFFFFF00000000|raw}, "expect_x": {"12": classification}, "expect_f": {}, "fflags": 0})
    result.append({"name": "unboxed", "words": [f(0x70,1,12,10)], "x": {}, "f": {"10": 0x3F800000}, "expect_x": {"12": 512}, "expect_f": {}, "fflags": 0})
    for a,b,le,lt,flags in [(0x3F800000,0x40000000,1,1,0),(0x80000000,0,1,0,0),(0x7FC00000,0,0,0,16),(0x7F800001,0,0,0,16)]:
        for funct,answer in ((0,le),(1,lt)):
            result.append({"name": "fcmp", "words": [f(0x50,funct,12,10,11)], "x": {}, "f": {"10":0xFFFFFFFF00000000|a,"11":0xFFFFFFFF00000000|b}, "expect_x":{"12":answer},"expect_f":{},"fflags":flags})
    for value in [0,1,-1,2**60+1,2**60+2**36-1,2**60+2**36,2**60+2**36+1,-(2**60+2**36+1),2**63-1,-2**63]:
        raw,flags=int_to_f32(value)
        result.append({"name":"fcvt_s_l","words":[f(0x68,0,10,10,2)],"x":{"10":value&MASK},"f":{},"expect_x":{},"expect_f":{"10":0xFFFFFFFF00000000|raw},"fflags":flags})
    for take in (0,1):
        result.append({"name":"conditional_update","words":[beq(12,0,12),i(0,20,21,0),f(0x10,0,11,10,10)],"x":{"12":take,"20":3,"21":7},"f":{"10":0xFFFFFFFF40000000,"11":0xFFFFFFFF3F800000},"expect_x":{"20":7 if take else 3},"expect_f":{"11":0xFFFFFFFF40000000 if take else 0xFFFFFFFF3F800000},"fflags":0})
    return result


def assembly(items):
    lines = [".option norvc", ".option norelax", ".section .text.init", ".globl _start", "_start:", "li t0, 0x6000", "csrs mstatus, t0"]
    data = []
    for number, item in enumerate(items, 1):
        lines += [f"case_{number}:", "csrw fcsr, zero"]
        for reg in (10,11,12,20,21):
            lines.append(f"li x{reg}, 0x{item['x'].get(str(reg),0):016x}")
        for reg in (10,11):
            label=f"fp_{number}_{reg}"
            data += [f"{label}:",f".dword 0x{item['f'].get(str(reg),0xFFFFFFFF00000000):016x}"]
            lines += [f"la t0, {label}",f"fld f{reg}, 0(t0)"]
        lines += [f".word 0x{word:08x}" for word in item["words"]]
        for reg,value in item["expect_x"].items():
            lines += [f"li t0, 0x{value:016x}",f"bne x{reg}, t0, fail_{number}"]
        for reg,value in item["expect_f"].items():
            lines += [f"fmv.x.d t2, f{reg}",f"li t0, 0x{value:016x}",f"bne t2, t0, fail_{number}"]
        lines += ["csrr t2, fflags",f"li t0, {item['fflags']}",f"bne t2, t0, fail_{number}"]
    lines += ["li t1, 1", "j finish"]
    for number in range(1,len(items)+1):
        lines += [f"fail_{number}:",f"li t1, {number*2+1}","j finish"]
    lines += ["finish:","la t0, tohost","sd t1, 0(t0)","1: j 1b", ".section .rodata", ".balign 8", *data,
              ".section .tohost,\"aw\",@progbits", ".balign 64", ".globl tohost", "tohost: .dword 0", ".globl fromhost", "fromhost: .dword 0"]
    return "\n".join(lines) + "\n"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spike",type=Path,default=ROOT/"build/riscv-fesvr-build/spike")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if out.exists():raise RuntimeError("choose a fresh controller comparison directory")
    out.mkdir(parents=True);items=cases()
    source_files=[Path(__file__).resolve(),ROOT/"src/mlxsim/model_control_program.py",*(ROOT/"simulator_ext/control_model").glob("*.cc"),*(ROOT/"simulator_ext/control_model").glob("*.h"),ROOT/"simulator_ext/control_model/CMakeLists.txt"]
    sources={str(path.relative_to(ROOT)):sha(path) for path in source_files}
    binary=ROOT/"build/mlx-control-leaf/mlx-control-leaf"
    cpp=[]
    for number,item in enumerate(items):
        job=out/f"case-{number}.json";answer=out/f"case-{number}-cpp.json"
        job.write_text(json.dumps({"schema":"rv64_leaf_probe_v1","words":item["words"],"x":item["x"],"f":item["f"]})+"\n")
        result=subprocess.run([str(binary),str(job),str(answer)],capture_output=True,text=True,timeout=30)
        if result.returncode:raise RuntimeError(f"C++ controller probe failed: {number}: {result.stderr}")
        actual=json.loads(answer.read_text())
        if actual["fflags"]!=item["fflags"] or any(actual["x"][int(r)]!=v for r,v in item["expect_x"].items()) or any(actual["f"][int(r)]!=v for r,v in item["expect_f"].items()):raise RuntimeError(f"C++ register/fflags mismatch: {number}")
        cpp.append({"case":number,"name":item["name"],"passed":True,"result_sha256":sha(answer)})
    source,linker,elf=out/"test.S",out/"test.ld",out/"test.elf"
    source.write_text(assembly(items))
    linker.write_text("OUTPUT_ARCH(riscv)\nENTRY(_start)\nSECTIONS { . = 0x80000000; .text : { *(.text.init) *(.text*) } .rodata : { *(.rodata*) } . = ALIGN(64); .tohost : { *(.tohost) } }\n")
    subprocess.run(["riscv64-unknown-elf-gcc","-march=rv64imafdc","-mabi=lp64d","-nostdlib","-static","-mcmodel=medany","-Wl,--no-relax","-T",str(linker),str(source),"-o",str(elf)],check=True,capture_output=True)
    with (out/"spike.log").open("w") as stream:
        result=subprocess.run([str(args.spike.resolve()),"--isa=RV64IMAFDC","-m64",str(elf)],stdout=stream,stderr=subprocess.STDOUT,timeout=120)
    if result.returncode:raise RuntimeError(f"Spike ISA comparison failed, code {result.returncode}: {out/'spike.log'}")
    if sources!={str(path.relative_to(ROOT)):sha(path) for path in source_files}:raise RuntimeError("controller sources changed during differential validation")
    report={"classification":"controller_rv64_leaf_differential_isa_check_not_rocket_system_validation","cases":cpp,"all_passed":True,"sources":sources,"spike_sha256":sha(args.spike.resolve()),"cpp_binary_sha256":sha(binary),"elf_sha256":sha(elf),"assembly_sha256":sha(source),"rocket_execution_verified":False,"inference_performance_eligible":False}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(f"CONTROLLER_SPIKE_ISA_CONFORMANCE_PASS cases={len(items)} {out/'report.json'}")


if __name__=="__main__":main()
