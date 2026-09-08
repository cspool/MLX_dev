"""Run actual RV64 graph dispatcher contract failures and completion checks."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from scripts.verify_mlx_spike_matrix_chain import ROOT,HOST,DEVICE,SPIKE,source_identity as bridge_sources

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True);args=parser.parse_args();out=args.output.resolve()
    if out.exists():raise RuntimeError("choose a fresh graph runtime contract directory")
    out.mkdir(parents=True);source=ROOT/"tests/fixtures/graph_runtime_contract.c"
    def identity():
        result=bridge_sources()
        for path in (Path(__file__).resolve(),source):result[str(path.relative_to(ROOT))]=sha(path)
        return result
    before=identity();build=ROOT/"build/mlx-spike-matrix"
    with (out/"build.log").open("w") as log:
        for command in (["cmake","-S",str(DEVICE),"-B",str(build),"-DCMAKE_BUILD_TYPE=Release"],["cmake","--build",str(build),"--target","mlx_spike_matrix","-j4"]):subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
        subprocess.run(["riscv64-unknown-elf-gcc","-std=c11","-O2","-march=rv64imafd","-mabi=lp64d","-mcmodel=medany","-ffreestanding","-fno-builtin","-fno-tree-loop-distribute-patterns","-ffp-contract=off","-Wall","-Wextra","-Werror","-nostdlib","-static","-Wl,--no-relax","-I",str(HOST),"-T",str(HOST/"link.ld"),str(HOST/"start.S"),str(HOST/"control_runtime.c"),str(HOST/"graph_runtime.c"),str(source),"-o",str(out/"test.elf")],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
    config={"base":2**32,"bytes":65536,"report":str(out/"device.json"),"matrix_options":{"trace":False},"vector_options":{"trace":False},"memory_options":{"trace":False}}
    (out/"plugin.json").write_text(json.dumps(config)+"\n");plugin=build/"libmlx_spike_matrix.so"
    with (out/"spike.log").open("w") as log:result=subprocess.run([str(SPIKE),"--isa=RV64IMAFD","-m64",f"--extlib={plugin}",f"--device=mlx_matrix,0x100000000,{out/'plugin.json'}",str(out/"test.elf")],stdout=log,stderr=subprocess.STDOUT,timeout=120)
    if result.returncode:raise RuntimeError(f"graph runtime contract failed at case/trap {result.returncode}")
    device=json.loads((out/"device.json").read_text())
    if device["launches"] or device["device_reads"] or device["device_writes"] or not device["memory_idle"]:raise RuntimeError("invalid graph tasks reached the accelerator")
    if before!=identity():raise RuntimeError("graph runtime contract sources changed")
    report={"classification":"actual_rv64_dispatch_contract_not_model_or_system_validation","sources":before,"checks":15,"all_passed":True,"elf_sha256":sha(out/"test.elf"),"plugin_sha256":sha(plugin),"device_sha256":sha(out/"device.json"),"mlx_system_verified":False,"inference_performance_eligible":False}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(f"GRAPH_RUNTIME_CONTRACT_PASS {out/'report.json'}")


if __name__=="__main__":main()
