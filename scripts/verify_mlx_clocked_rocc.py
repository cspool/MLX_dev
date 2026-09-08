"""Check RoCC command/response flow and sanitizer parity without a CPU claim."""
import argparse
import json
import os
import subprocess
from pathlib import Path

from scripts.run_mlx_clocked_chipyard import ROOT,BRIDGE,source_identity
from scripts.run_mlx_spike_graph import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True);args=parser.parse_args();out=args.output.resolve()
    if out.exists():raise RuntimeError("choose a fresh RoCC contract directory")
    out.mkdir(parents=True)
    def sources():
        result=source_identity()
        for path in (Path(__file__).resolve(),ROOT/"tests/test_clocked_rocc.py",ROOT/"tests/clocked_rocc_driver.cc",ROOT/"tests/test_clocked_device.py"):
            result[str(path.relative_to(ROOT))]=sha(path)
        return result
    before=sources()
    with (out/"pytest.log").open("w") as log:subprocess.run([str(ROOT/".venv/bin/python"),"-m","pytest","-q","tests/test_clocked_rocc.py",f"--basetemp={out/'pytest'}",f"--junitxml={out/'regression.xml'}"],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600,cwd=ROOT)
    build=ROOT/"build/mlx-clocked-rocc-asan"
    with (out/"asan-build.log").open("w") as log:
        for command in (["cmake","-S",str(BRIDGE),"-B",str(build),"-DCMAKE_BUILD_TYPE=Debug","-DCMAKE_CXX_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer"],
                        ["cmake","--build",str(build),"--target","clocked-rocc-driver","-j4"]):subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
    env=os.environ.copy();env["ASAN_OPTIONS"]="detect_leaks=1:halt_on_error=1";env["UBSAN_OPTIONS"]="halt_on_error=1:print_stacktrace=1";cases=[]
    for file in sorted((out/"pytest").rglob("job.json")):
        if any(p.is_symlink() for p in file.parents):continue
        report=file.parent/"report.json";san=file.parent/"asan.json"
        p=subprocess.run([str(build/"clocked-rocc-driver"),str(file),str(san)],capture_output=True,text=True,timeout=120,env=env)
        (file.parent/"asan.log").write_text(p.stdout+p.stderr)
        if report.exists():
            if p.returncode or json.loads(report.read_text())!=json.loads(san.read_text()):raise RuntimeError("RoCC sanitizer result differs")
        elif p.returncode!=1 or p.stderr.strip()!="RoCC response has no matching accepted tag":raise RuntimeError("RoCC sanitizer rejection differs")
        cases.append({"case":str(file.parent.relative_to(out)),"job_sha256":sha(file),"report_sha256":sha(report) if report.exists() else None,"asan_sha256":sha(san) if san.exists() else None,"exit_code":p.returncode})
    if len(cases)!=5 or before!=sources():raise RuntimeError("RoCC cases missing or sources changed")
    report={"classification":"rocc_requestor_contract_not_cpu_or_chipyard_execution","sources":before,"cases":cases,"regression_sha256":sha(out/"regression.xml"),"mlx_system_verified":False,"inference_performance_eligible":False}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(f"CLOCKED_ROCC_CONTRACT_VERIFIED {out/'report.json'}")


if __name__=="__main__":main()
