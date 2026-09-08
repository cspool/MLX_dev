"""Audit Boolean/guard native replay and independent Spike ALU conformance."""
import argparse
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import xml.etree.ElementTree as ET

from mlxsim.model_control_program import control_program, i
from scripts.run_mlx_ready_model import ROOT, sources
from scripts.run_mlx_tensor_semantics import sha
from scripts.run_mlx_physical_model import runtime_libraries
from scripts.verify_mlx_controller import assembly
from scripts.verify_mlx_ready_graph import normalized


def require(value, message):
    if not value:
        raise RuntimeError(message)


def isa_cases():
    result = []
    def add(name, words, x, expect, floats=None, flags=0):
        result.append({"name": name, "words": words, "x": {str(k): v & (2**64-1) for k,v in x.items()},
                       "f": {str(k): 0xffffffff00000000 | v for k,v in (floats or {}).items()},
                       "expect_x": {str(k): v for k,v in expect.items()}, "expect_f": {}, "fflags": flags})
    for a,b in [(-2**63,2**63-1), (2**63-1,-2**63), (2**60+1,2**60), (2**60,2**60+1),
                (-1,0), (0,-1), (0,0), (-2**63,-2**63)]:
        add("ge-i64", control_program("ge")["phases"]["body"], {10:a,11:b}, {12:int(a>=b)})
    for a,b in [(0xff800000,0), (0x80000000,0), (0x7f800000,0x3f800000),
                (0x7fc00001,0x3f800000), (0x3f800000,0x7f800001), (0x40400000,0x40000000)]:
        left,right = (struct.unpack("<f", struct.pack("<I", raw))[0] for raw in (a,b))
        add("ge-f32", control_program("ge","f32")["phases"]["body"], {}, {12:int(left>=right)},
            {10:a,11:b}, 16 if math.isnan(left) or math.isnan(right) else 0)
    for a in (0,1):
        for b in (0,1):
            add("boolean-and", control_program("bitwise_and")["phases"]["body"], {10:a,11:b}, {12:a&b})
            add("guard-compare", control_program("guard")["phases"]["body"], {10:a,11:b}, {12:a,13:a^b})
    for values in ([], [1], [0], [1]*8, [1,1,0,1], [1,255]):
        p=control_program("all")["phases"]; words=list(p["init"])
        for value in values:
            words += [i(0,10,0,value), *p["body"]]
        add("all-identity-and-loop", words, {}, {12:int(all(values))})
    require(len(result)==28, "ISA case coverage changed")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("tests", "asan-port", "asan-graph", "asan-physical", "spike", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args=parser.parse_args(); test=args.tests.resolve(); out=args.output.resolve()
    require(not out.exists(), "choose a fresh mask verification directory")
    suite=ET.parse(test/"regression.xml").getroot().find("testsuite")
    require(suite is not None and all(suite.get(k)=="0" for k in ("errors","failures","skipped")), "mask regressions did not pass")
    before=sources()
    for p in (Path(__file__).resolve(), ROOT/"scripts/verify_mlx_controller.py", ROOT/"scripts/verify_mlx_ready_graph.py",
              ROOT/"tests/test_mask_control.py", ROOT/"tests/control_external_memory.cc"):
        before[str(p.relative_to(ROOT))]=sha(p)
    binaries={"port":args.asan_port.resolve(),"graph":args.asan_graph.resolve(),"physical":args.asan_physical.resolve(),
              "spike":args.spike.resolve(),"leaf":ROOT/"build/mlx-control-leaf/mlx-control-leaf"}
    files={str(p):sha(p) for p in binaries.values()}
    for p in binaries.values(): files.update(runtime_libraries(p))
    files[str(test/"regression.xml")]=sha(test/"regression.xml")
    out.mkdir(parents=True)
    env=os.environ.copy();env["ASAN_OPTIONS"]="detect_leaks=1:halt_on_error=1";env["UBSAN_OPTIONS"]="halt_on_error=1:print_stacktrace=1"
    prefixes=("test_ge_reads_actual_", "test_all_boolean_reduction_", "test_boolean_and_broadcast_", "test_guard_reads_real_", "test_guard_contract_")
    selected=[]
    for p in sorted((test/"pytest").rglob("job.json")):
        if any(parent.is_symlink() for parent in p.parents):continue
        if p.relative_to(test/"pytest").parts[0].startswith(prefixes): selected.append(("port",p))
    for p in sorted((test/"pytest").rglob("program.json")):
        if any(parent.is_symlink() for parent in p.parents):continue
        if not p.relative_to(test/"pytest").parts[0].startswith(("test_guarded_graph_compiles_","test_compiled_guard_edges_")):continue
        selected.append(("physical" if p.parent.name=="serial" else "graph", p))
    require(len(selected)==31, f"expected 31 new native safety replays, found {len(selected)}")
    def bind_assets(value):
        if isinstance(value,dict):
            if value.get("kind")=="mapped_file":
                file=Path(value["path"]).resolve();files[str(file)]=sha(file)
            for child in value.values():bind_assets(child)
        elif isinstance(value,list):
            for child in value:bind_assets(child)
    for kind,p in selected:
        bind_assets(json.loads(p.read_text()))
        for file in (p.parent/"out").glob("*"):
            if file.is_file():files[str(file)]=sha(file)
    replays=[]
    for number,(kind,p) in enumerate(selected):
        base=p.parent; dst=out/f"asan-{number}"; log=out/f"asan-{number}.log"
        files[str(p)]=sha(p); command=[binaries[kind],p]
        if kind!="port":
            options=base/("system-options.json" if kind=="physical" else "options.json")
            files[str(options)]=sha(options);command.append(options)
        command.append(dst); expected=0 if (base/"out/result.json").exists() else 1
        with log.open("w") as stream:
            run=subprocess.run(list(map(str,command)),cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=180)
        require(run.returncode==expected, f"mask sanitizer outcome differs: {log}")
        text=log.read_text()
        require(not any(marker in text for marker in ("ERROR: AddressSanitizer", "runtime error:", "LeakSanitizer", "DEADLYSIGNAL")), f"sanitizer failure: {log}")
        if not expected:
            original=json.loads((base/"out/result.json").read_text()); actual=json.loads((dst/"result.json").read_text())
            if kind=="port":
                require(original==actual and (base/"out/output.bin").read_bytes()==(dst/"output.bin").read_bytes(), "sanitizer changed controller events/values")
            else:
                require(normalized(original)==normalized(actual), "sanitizer changed complete graph report")
                for a,b in zip(original["outputs"],actual["outputs"],strict=True):
                    require(Path(a["logits_file"]).read_bytes()==Path(b["logits_file"]).read_bytes(), "sanitizer changed graph output")
        else:
            require(not (dst/"result.json").exists(), "failed guard/descriptor produced success evidence")
            if (base/"out/failure-evidence.json").exists():
                require(json.loads((base/"out/failure-evidence.json").read_text())==json.loads((dst/"failure-evidence.json").read_text()), "guard mismatch wrote a result")
        replays.append({"case":str(p.relative_to(test)),"kind":kind,"expected_exit":expected,"log_sha256":sha(log),
                        "output":str(dst),"complete_report_and_values_equal":not expected})
    items=isa_cases(); (out/"isa-cases.json").write_text(json.dumps(items,indent=2)+"\n")
    for number,item in enumerate(items):
        job=out/f"isa-{number}.json"; answer=out/f"isa-{number}-cpp.json"
        job.write_text(json.dumps({"schema":"rv64_leaf_probe_v1", "words":item["words"],"x":item["x"],"f":item["f"]})+"\n")
        run=subprocess.run(list(map(str,[binaries["leaf"],job,answer])),capture_output=True,text=True,timeout=30)
        require(run.returncode==0, f"C++ ISA probe failed: {run.stderr}")
        result=json.loads(answer.read_text())
        require(result["fflags"]==item["fflags"] and all(result["x"][int(k)]==v for k,v in item["expect_x"].items()), "C++ ISA result differs from independent expectation")
    source,linker,elf=out/"isa.S",out/"isa.ld",out/"isa.elf"
    source.write_text(assembly(items))
    linker.write_text("OUTPUT_ARCH(riscv)\nENTRY(_start)\nSECTIONS { . = 0x80000000; .text : { *(.text.init) *(.text*) } .rodata : { *(.rodata*) } . = ALIGN(64); .tohost : { *(.tohost) } }\n")
    subprocess.run(["riscv64-unknown-elf-gcc","-march=rv64imafdc","-mabi=lp64d","-nostdlib","-static","-mcmodel=medany","-Wl,--no-relax","-T",str(linker),str(source),"-o",str(elf)],check=True,capture_output=True)
    with (out/"spike.log").open("w") as log:
        result=subprocess.run([str(binaries["spike"]),"--isa=RV64IMAFDC","-m64",str(elf)],stdout=log,stderr=subprocess.STDOUT,timeout=120)
    require(result.returncode==0, "independent Spike ISA execution failed")
    require(all(sha(ROOT/p)==h for p,h in before.items()) and all(sha(Path(p))==h for p,h in files.items()), "verification sources or inputs changed")
    report={"classification":"native_mask_control_guard_safety_and_isa_not_full_model_or_system_validation",
            "sources":before,"files":files,"regression_tests":int(suite.get("tests")),"replays":replays,
            "isa_cases":len(items),"spike_exit_code":0,"elf_sha256":sha(elf),
            "full_model_execution_verified":False,"actual_rocket_execution_verified":False,"inference_performance_eligible":False}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(f"MASK_CONTROL_NATIVE_SAFETY_AND_SPIKE_PASS replays={len(replays)} isa={len(items)} {out/'report.json'}")


if __name__=="__main__":main()
