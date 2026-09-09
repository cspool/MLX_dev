"""Verify external-edge device control, transport failures and sanitizer parity."""
import argparse
import json
import os
import subprocess
from pathlib import Path

from scripts.run_mlx_spike_graph import ROOT,sha
from scripts.verify_mlx_spike_matrix_chain import source_identity as bridge_sources


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True);parser.add_argument("--include-memory-v2",action="store_true");args=parser.parse_args();out=args.output.resolve()
    if out.exists():raise RuntimeError("choose a fresh clocked-device evidence directory")
    out.mkdir(parents=True)
    def identity():
        result=bridge_sources();paths=list((ROOT/"system_sim/clocked_device").iterdir())
        paths.extend([Path(__file__).resolve(),ROOT/"tests/test_clocked_device.py",ROOT/"tests/clocked_device_driver.cc"])
        if args.include_memory_v2:paths.extend(ROOT/name for name in ("tests/test_system_memory_v2.py","tests/test_memory_wire_lowering.py","tests/test_model_tensor_semantics.py"))
        for path in paths:
            if path.is_file():result[str(path.relative_to(ROOT))]=sha(path)
        return result
    sources=identity()
    tests=["tests/test_clocked_device.py"]+(["tests/test_system_memory_v2.py"] if args.include_memory_v2 else [])
    with (out/"pytest.log").open("w") as log:subprocess.run([str(ROOT/".venv/bin/python"),"-m","pytest","-q",*tests,
        f"--basetemp={out/'pytest'}",f"--junitxml={out/'regression.xml'}"],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    build=ROOT/"build/mlx-clocked-device-asan"
    with (out/"asan-build.log").open("w") as log:
        for command in (["cmake","-S",str(ROOT/"system_sim/clocked_device"),"-B",str(build),"-DCMAKE_BUILD_TYPE=Debug","-DCMAKE_CXX_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer"],
                        ["cmake","--build",str(build),"--target","clocked-device-driver",*(["memory-wire-dump"] if args.include_memory_v2 else []),"-j4"]):
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
    environment=os.environ.copy();environment["ASAN_OPTIONS"]="detect_leaks=1:halt_on_error=1";environment["UBSAN_OPTIONS"]="halt_on_error=1:print_stacktrace=1"
    cases=[]
    for job in sorted((out/"pytest").rglob("job.json")):
        if any(p.is_symlink() for p in job.parents):continue
        report=job.parent/"report.json";san=job.parent/"report-asan.json";native=job.parent/"report-replay.json"
        ordinary=subprocess.run([str(ROOT/"build/mlx-clocked-device/clocked-device-driver"),str(job),str(native)],capture_output=True,text=True,timeout=120)
        instrumented=subprocess.run([str(build/"clocked-device-driver"),str(job),str(san)],capture_output=True,text=True,timeout=120,env=environment)
        (job.parent/"asan.log").write_text(instrumented.stdout+instrumented.stderr)
        if ordinary.returncode!=instrumented.returncode or ordinary.stderr!=instrumented.stderr:raise RuntimeError("sanitizer terminal result differs")
        if report.exists():
            if ordinary.returncode or json.loads(report.read_text())!=json.loads(san.read_text()) or json.loads(report.read_text())!=json.loads(native.read_text()):raise RuntimeError("clocked controller replay differs")
        elif not ordinary.returncode:raise RuntimeError("invalid job unexpectedly accepted")
        cases.append({"case":str(job.parent.relative_to(out)),"job_sha256":sha(job),"report_sha256":sha(report) if report.exists() else None,
                      "asan_sha256":sha(san) if san.exists() else None,"exit_code":ordinary.returncode})
    decoders=[]
    if args.include_memory_v2:
        for wire in sorted((out/"pytest").rglob("wire.bin")):
            if any(p.is_symlink() for p in wire.parents):continue
            expected=wire.parent/"decoded.json";san=wire.parent/"decoded-asan.json";native=wire.parent/"decoded-replay.json"
            ordinary=subprocess.run([str(ROOT/"build/mlx-spike-matrix/memory-wire-dump"),str(wire),str(native)],capture_output=True,text=True,timeout=120)
            instrumented=subprocess.run([str(build/"physical-wire/memory-wire-dump"),str(wire),str(san)],capture_output=True,text=True,timeout=120,env=environment)
            (wire.parent/"decoder-asan.log").write_text(instrumented.stdout+instrumented.stderr)
            if ordinary.returncode!=instrumented.returncode or ordinary.stderr!=instrumented.stderr:raise RuntimeError("memory v2 decoder sanitizer terminal differs")
            if expected.exists():
                if ordinary.returncode or json.loads(expected.read_text())!=json.loads(san.read_text()) or json.loads(expected.read_text())!=json.loads(native.read_text()):raise RuntimeError("memory v2 decoder replay differs")
            elif ordinary.returncode!=1:raise RuntimeError("invalid memory v2 descriptor accepted")
            decoders.append({"case":str(wire.parent.relative_to(out)),"wire_sha256":sha(wire),"exit_code":ordinary.returncode,
                             "report_sha256":sha(expected) if expected.exists() else None,"asan_sha256":sha(san) if san.exists() else None})
    if len(cases)!=(34 if args.include_memory_v2 else 16) or (args.include_memory_v2 and len(decoders)!=26) or sources!=identity():raise RuntimeError("clocked controller cases missing or sources changed")
    report={"classification":"external_system_edge_controller_and_bus_contract_not_actual_cpu_or_chipyard_validation","sources":sources,"cases":cases,
        "regression_sha256":sha(out/"regression.xml"),"release_binary_sha256":sha(ROOT/"build/mlx-clocked-device/clocked-device-driver"),"asan_binary_sha256":sha(build/"clocked-device-driver"),
        "full_model_execution_verified":False,"mlx_system_verified":False,"inference_performance_eligible":False}
    if args.include_memory_v2:report["memory_v2_decoder_replays"]=decoders
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(f"CLOCKED_DEVICE_CONTRACT_VERIFIED {out/'report.json'}")


if __name__=="__main__":main()
