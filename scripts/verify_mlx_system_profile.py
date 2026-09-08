"""Bind native profile validation and sanitizer parity to the current implementation."""
import argparse
import json
import os
import subprocess
from pathlib import Path

from scripts.run_mlx_clocked_chipyard import ROOT,PROFILE,source_identity
from scripts.run_mlx_spike_graph import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True);args=parser.parse_args();out=args.output.resolve()
    if out.exists():raise RuntimeError("choose a fresh profile verification directory")
    out.mkdir(parents=True)
    def identity():
        result=source_identity()
        for path in (Path(__file__).resolve(),ROOT/"tests/test_system_profile.py",ROOT/"tests/test_system_image.py"):result[str(path.relative_to(ROOT))]=sha(path)
        return result
    sources=identity()
    with (out/"pytest.log").open("w") as log:subprocess.run([str(ROOT/".venv/bin/python"),"-m","pytest","-q","tests/test_system_profile.py","tests/test_system_image.py",f"--basetemp={out/'pytest'}",f"--junitxml={out/'regression.xml'}"],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600,cwd=ROOT)
    build=ROOT/"build/mlx-system-profile-asan"
    with (out/"asan-build.log").open("w") as log:
        for command in (["cmake","-S",str(PROFILE),"-B",str(build),"-DCMAKE_BUILD_TYPE=Debug","-DCMAKE_CXX_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer"],["cmake","--build",str(build),"--target","system-profile-dump","-j4"]):subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
    environment=os.environ.copy();environment["ASAN_OPTIONS"]="detect_leaks=1:halt_on_error=1";environment["UBSAN_OPTIONS"]="halt_on_error=1:print_stacktrace=1"
    profiles=[None]+[p for p in sorted((out/"pytest").rglob("profile.json")) if not any(a.is_symlink() for a in p.parents)];cases=[]
    for index,path in enumerate(profiles):
        args=[] if path is None else [str(path)]
        regular=subprocess.run([str(ROOT/"build/mlx-system-profile/system-profile-dump"),*args],capture_output=True,text=True,timeout=30)
        checked=subprocess.run([str(build/"system-profile-dump"),*args],capture_output=True,text=True,timeout=30,env=environment)
        if (regular.returncode,regular.stdout,regular.stderr)!=(checked.returncode,checked.stdout,checked.stderr):raise RuntimeError("system profile sanitizer parity failed")
        log=out/f"profile-{index}.log";log.write_text(checked.stdout+checked.stderr);cases.append({"profile":str(path) if path else "demo_default","exit_code":checked.returncode,"log_sha256":sha(log)})
    if len(cases)!=8 or sources!=identity():raise RuntimeError("profile verification cases missing or sources changed")
    report={"classification":"system_profile_and_image_contract_not_model_execution","sources":sources,"cases":cases,"regression_sha256":sha(out/"regression.xml"),"full_model_execution_verified":False,"mlx_system_verified":False,"inference_performance_eligible":False}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(f"SYSTEM_PROFILE_CONTRACT_PASS {out/'report.json'}")


if __name__=="__main__":main()
