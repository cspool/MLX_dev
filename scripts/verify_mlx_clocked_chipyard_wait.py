"""Run blocking RoCC WAIT chains on the built real Rocket tensor configuration."""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import scripts.verify_mlx_spike_matrix_chain as chain
from scripts.run_mlx_clocked_chipyard import ROOT,BRIDGE,CONFIG,inputs,identity,execute,source_identity
from scripts.run_mlx_spike_graph import sha,HOST


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True);args=parser.parse_args();out=args.output.resolve();chipyard=ROOT/"build/chipyard-native"
    if out.exists():raise RuntimeError("choose a fresh Rocket WAIT evidence directory")
    manifest=json.loads((chipyard/"clocked-rocc-build.json").read_text());simulator=chipyard/f"sims/verilator/simulator-chipyard-{CONFIG}"
    if manifest["inputs"]!=inputs(chipyard) or manifest["build_identity"]!=identity(manifest["inputs"]) or sha(simulator)!=manifest["simulator_sha256"]:raise RuntimeError("clocked simulator build stale")
    out.mkdir(parents=True);before=source_identity();before[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__));owned=out/simulator.name;shutil.copy2(simulator,owned);results=[]
    for name,precision,perturbed in (("f16","f16",False),("f16-perturbed","f16",True),("f32","f32",False)):
        directory=out/name;directory.mkdir();original_base=chain.BASE
        try:
            chain.BASE=0x81000000
            source,expected,images=chain.firmware(precision=precision,perturb=perturbed,memory_stage=True,vector_stage=True,vector_kind="softmax")
        finally:chain.BASE=original_base
        begin=source.index('static uint64_t submit(');end=source.index('static uint64_t launch(',begin)
        source=source[:begin]+'''static uint64_t submit(const volatile void *source,unsigned bytes,uint64_t descriptor){
 const volatile unsigned char *src=source;volatile unsigned char *dst=(volatile unsigned char *)(uintptr_t)descriptor;
 for(unsigned i=0;i<bytes;++i)dst[i]=src[i];
 return mlx_clocked_submit(descriptor,bytes);
}
'''+source[end:]
        old='*(volatile uint64_t *)(uintptr_t)(BASE+MLX_MATRIX_REG_ID)!=MLX_MATRIX_WIRE_MAGIC'
        if source.count(old)!=1 or source.count(' return 0;\n}')!=1:raise RuntimeError("chain source transport/checker changed")
        source='#include "host_runtime.h"\n'+source.replace(old,'mlx_clocked_status(14)!=MLX_CLOCKED_ROCC_MAGIC').replace(' return 0;\n}',' mlx_clocked_pass();return 0;\n}')
        (directory/"test.c").write_text(source);(directory/"expected.json").write_text(json.dumps(expected,indent=2)+"\n");objects=[]
        for stem,data in images.items():
            (directory/f"{stem}.bin").write_bytes(data);symbol={"matrix_wire":"image","vector_wire":"vector_image","memory_wire":"memory_image"}[stem]
            subprocess.run(["riscv64-unknown-elf-objcopy","-I","binary","-O","elf64-littleriscv","-B","riscv","--set-section-alignment",".data=8","--redefine-sym",f"_binary_{stem}_bin_start={symbol}",f"{stem}.bin",f"{stem}.o"],cwd=directory,capture_output=True,check=True);objects.append(str(directory/f"{stem}.o"))
        command=["riscv64-unknown-elf-gcc","-std=c11","-O2","-march=rv64imafd","-mabi=lp64d","-mcmodel=medany","-ffreestanding","-fno-builtin","-fno-tree-loop-distribute-patterns","-ffp-contract=off","-Wall","-Wextra","-Werror","-nostdlib","-static","-Wl,--no-relax","-I",str(HOST),"-I",str(BRIDGE),"-I",str(ROOT/"system_sim/physical_device"),"-T",str(ROOT/"system_sim/native/link.ld"),str(HOST/"start.S"),str(HOST/"control_runtime.c"),str(directory/"test.c"),*objects,"-o",str(directory/"test.elf")]
        execute(command,directory/"build.log",120)
        env=os.environ.copy();env["MLX_CLOCKED_REPORT"]=str(directory/"device.json")
        execute([str(owned),"+max-cycles=10000000",str(directory/"test.elf")],directory/"chipyard.log",600,env)
        report=json.loads((directory/"device.json").read_text())
        if "MLX_CLOCKED_CHAIN_PASS" not in (directory/"chipyard.log").read_text() or report["launches"]!=9 or len(report["windows"])!=9 or report["requests"]!=report["responses"] or report["frontend_error"]:raise RuntimeError("Rocket WAIT chain failed")
        if report["build_identity"]!=manifest["build_identity"] or any(not w["done"] or w["error"] or not w["transport"]["idle"] for w in report["windows"]):raise RuntimeError("Rocket WAIT windows did not drain")
        results.append({"case":name,"expected":expected,"elf_sha256":sha(directory/"test.elf"),"source_sha256":sha(directory/"test.c"),"device_sha256":sha(directory/"device.json"),"log_sha256":sha(directory/"chipyard.log")})
    after=source_identity();after[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    if before!=after or sha(owned)!=manifest["simulator_sha256"]:raise RuntimeError("Rocket WAIT sources/binary changed")
    report={"classification":"actual_rocket_blocking_wait_control_and_compute_chains_not_full_model_validation","sources":before,"build":manifest,"cases":results,"actual_rocket_execution":True,"full_model_execution_verified":False,"mlx_system_verified":False,"inference_performance_eligible":False}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(f"CLOCKED_ROCKET_WAIT_PASS {out/'report.json'}")


if __name__=="__main__":main()
