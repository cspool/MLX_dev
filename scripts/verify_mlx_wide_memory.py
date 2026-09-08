"""Validate wide physical addressing, checked preload, and the real DPI boundary."""
import argparse
import json
import os
import subprocess
from pathlib import Path

from scripts.run_mlx_spike_graph import ROOT,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True);args=parser.parse_args();out=args.output.resolve();source=ROOT/"system_sim/wide_memory";build=ROOT/"build/mlx-wide-memory"
    if out.exists():raise RuntimeError("choose a fresh wide-memory validation directory")
    out.mkdir(parents=True)
    def identity():
        paths=list(source.iterdir())+[Path(__file__).resolve(),ROOT/"tests/wide_memory_contract.cc",ROOT/"tests/wide_memory_wire.cc",ROOT/"tests/test_wide_memory.py"]
        paths.extend(ROOT/"build/chipyard-native/generators/testchipip/src/main/resources/testchipip/csrc"/name for name in ("mm.cc","mm.h"))
        return {str(p.relative_to(ROOT)):sha(p) for p in paths if p.is_file()}
    before=identity()
    def run(command,log,env=None):
        with (out/log).open("w") as stream:subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=env,check=True,timeout=300)
    run([str(ROOT/".venv/bin/python"),"-m","pytest","-q","tests/test_wide_memory.py",f"--basetemp={out/'pytest'}",f"--junitxml={out/'regression.xml'}"],"pytest.log")
    san=ROOT/"build/mlx-wide-memory-asan"
    run(["cmake","-S",str(source),"-B",str(san),"-DCMAKE_BUILD_TYPE=Debug","-DCMAKE_CXX_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer"],"asan-configure.log")
    run(["cmake","--build",str(san),"--target","wide-memory-contract","-j4"],"asan-build.log")
    env=os.environ.copy();env["ASAN_OPTIONS"]="detect_leaks=1:halt_on_error=1";env["UBSAN_OPTIONS"]="halt_on_error=1:print_stacktrace=1"
    jobs=[None]+[p for p in sorted((out/"pytest").rglob("job.json")) if not any(parent.is_symlink() for parent in p.parents)];cases=[]
    for index,job in enumerate(jobs):
        extra=[] if job is None else [str(job)]
        ordinary=subprocess.run([str(build/"wide-memory-contract"),*extra],capture_output=True,text=True,timeout=60)
        instrumented=subprocess.run([str(san/"wide-memory-contract"),*extra],capture_output=True,text=True,timeout=60,env=env)
        if ordinary.returncode!=instrumented.returncode or ordinary.stdout!=instrumented.stdout or ordinary.stderr!=instrumented.stderr:raise RuntimeError("wide memory sanitizer parity failed")
        (out/f"native-{index}.log").write_text(ordinary.stdout+ordinary.stderr);cases.append({"job":str(job) if job else "builtin","exit_code":ordinary.returncode,"log_sha256":sha(out/f"native-{index}.log")})
    wire=ROOT/"build/mlx-wide-memory-wire";upstream=ROOT/"build/chipyard-native/generators/testchipip/src/main/resources/testchipip/csrc"
    run(["verilator","--cc","--exe","--vpi","--top-module","MLXWideMemory","-Wno-fatal","-GADDR_BITS=35","-GMEM_SIZE=64'd17179869184","-Mdir",str(wire),
         "-CFLAGS",f"-std=c++17 -I{source} -I{upstream}","-LDFLAGS",f"{build/'libmlx_wide_memory.a'} -ljsoncpp -lcrypto",str(source/"MLXWideMemory.sv"),str(source/"memory_dpi.cc"),str(ROOT/"tests/wide_memory_wire.cc")],"wire-configure.log")
    run(["make","-C",str(wire),"-f","VMLXWideMemory.mk","CFG_CXXFLAGS_STD_NEWEST=-std=gnu++17","-j4"],"wire-build.log")
    environment=os.environ.copy();environment["MLX_WIDE_MEMORY_REPORT"]=str(out/"wire-memory.json")
    run([str(wire/"VMLXWideMemory")],"wire.log",environment)
    state=json.loads((out/"wire-memory.json").read_text())
    if "WIDE_MEMORY_DPI_PASS" not in (out/"wire.log").read_text() or state["bytes"]!=16*2**30 or not state["idle"] or state["max_read_address"]!=0x80000000+16*2**30-8:raise RuntimeError("actual SV/DPI address test incomplete")
    if len(cases)!=10 or before!=identity():raise RuntimeError("wide memory cases missing or sources changed")
    report={"classification":"wide_memory_cpp_and_sv_dpi_contract_not_full_model_or_dram_performance","sources":before,"cases":cases,"regression_sha256":sha(out/"regression.xml"),"wire_memory_sha256":sha(out/"wire-memory.json"),"wire_binary_sha256":sha(wire/"VMLXWideMemory"),"mlx_system_verified":False,"inference_performance_eligible":False}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(f"WIDE_MEMORY_CONTRACT_VERIFIED {out/'report.json'}")


if __name__=="__main__":main()
