"""Compile/decode full-model memory plans without claiming data execution."""
import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path

from scripts.run_mlx_physical_model import source_identity as runtime_sources
from system_sim.physical_host.address_plan import iter_bindings
from system_sim.physical_device.memory_lowering import lower_memory

ROOT=Path(__file__).resolve().parents[1]
BUILD=ROOT/"build/mlx-spike-matrix"
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--program",type=Path,required=True);parser.add_argument("--lifetimes",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);args=parser.parse_args();out=args.output.resolve()
    if out.exists():raise RuntimeError("choose a fresh memory wire audit directory")
    out.mkdir(parents=True);inputs={str(p.resolve()):sha(p) for p in (args.program,args.lifetimes)}
    def identity():
        result=runtime_sources();files=[Path(__file__).resolve()]
        for name in ("physical_host","physical_device"):files.extend(p for p in (ROOT/"system_sim"/name).iterdir() if p.is_file())
        for file in files:result[str(file.relative_to(ROOT))]=sha(file)
        return result
    sources=identity();program=json.loads(args.program.read_text());life=json.loads(args.lifetimes.read_text());routes=[];expected=[];files=[]
    for node,layouts,bindings in iter_bindings(program,life):
        if "memory_program" not in node:continue
        blob,route=lower_memory(node,layouts,bindings);file=out/f"memory-{node['source_operator_id']}.bin";file.write_bytes(blob);files.append(str(file));route.update(command_file=str(file),command_sha256=sha(file));routes.append(route)
        p=copy.deepcopy(node["memory_program"]);p["reason"]="wire_checked_layout";p["input_layouts"]={f"a{route['input_slots'][name]:02}":{**layout,"root":"r"+str(route["storage_root_ids"][layout["root"]])} for name,layout in p["input_layouts"].items()}
        p["output_layout"]["root"]="r"+str(route["storage_root_ids"][p["output_layout"]["root"]]) if p["mode"]=="view" else "output"
        expected.append(p)
    (out/"files.json").write_text(json.dumps(files)+"\n")
    with (out/"build.log").open("w") as log:
        for command in (["cmake","-S",str(ROOT/"system_sim/physical_device"),"-B",str(BUILD),"-DCMAKE_BUILD_TYPE=Release"],["cmake","--build",str(BUILD),"--target","memory-wire-dump","-j4"]):subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    binary=BUILD/"memory-wire-dump"
    with (out/"decode.log").open("w") as log:subprocess.run([str(binary),str(out/"files.json"),str(out/"decoded.json")],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    decoded=json.loads((out/"decoded.json").read_text())["windows"]
    if len(decoded)!=len(expected):raise RuntimeError("memory wire decode omitted model nodes")
    for actual,wanted in zip(decoded,expected,strict=True):
        if not actual["backend_constructor_validated"] or actual["node"]["memory_program"]!=wanted or actual["view_elided"]!=(wanted["mode"]=="view"):raise RuntimeError("memory wire plan/layout/root roundtrip changed")
    if sources!=identity() or any(sha(Path(name))!=digest for name,digest in inputs.items()):raise RuntimeError("memory wire audit inputs changed")
    views=sum(r["mode"]=="view" for r in routes)
    report={"classification":"full_model_memory_wire_compile_and_view_validation_not_data_execution","sources":sources,"inputs":inputs,"source_nodes":len(program["nodes"]),"memory_source_nodes":len(routes),"checked_views":views,"transfer_nodes":len(routes)-views,"command_bytes_total":len(routes)*15872,
        "routes":routes,"all_cpp_decodes_equal":True,"decoded_sha256":sha(out/"decoded.json"),"decoder_binary_sha256":sha(binary),"model_data_executed":False,"mlx_system_verified":False,"inference_performance_eligible":False}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(f"MEMORY_WIRE_FULL_MODEL_LOWERING_PASS {out/'report.json'}")


if __name__=="__main__":main()
