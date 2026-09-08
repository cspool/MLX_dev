"""Encode/decode every full-model matrix window; not a model execution claim."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from system_sim.physical_device.lowering import lower_matrix_window,geometry
from system_sim.physical_host.address_plan import iter_bindings
from scripts.run_mlx_physical_model import source_identity as runtime_sources

ROOT=Path(__file__).resolve().parents[1]
BUILD=ROOT/"build/mlx-spike-matrix"
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--program",type=Path,required=True);parser.add_argument("--lifetimes",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);args=parser.parse_args();out=args.output.resolve()
    if out.exists():raise RuntimeError("choose a fresh matrix wire audit directory")
    out.mkdir(parents=True);inputs={str(p.resolve()):sha(p) for p in (args.program,args.lifetimes)}
    def identity():
        result=runtime_sources();files=[Path(__file__).resolve()]
        for name in ("physical_host","physical_device"):files.extend(p for p in (ROOT/"system_sim"/name).iterdir() if p.is_file())
        for file in files:result[str(file.relative_to(ROOT))]=sha(file)
        return result
    sources=identity();program=json.loads(args.program.read_text());life=json.loads(args.lifetimes.read_text());routes=[];expectations=[];files=[];matrix_nodes=0
    for node,layouts,bindings in iter_bindings(program,life):
        if "matrix_program" not in node:continue
        matrix_nodes+=1
        for batch in range(geometry(node,layouts)["batches"]):
            blob,route=lower_matrix_window(node,layouts,bindings,batch);file=out/f"matrix-{node['source_operator_id']}-{batch}.bin";file.write_bytes(blob);files.append(str(file))
            route["command_sha256"]=sha(file);route["command_file"]=str(file);routes.append(route)
            p={key:value for key,value in node["matrix_program"].items() if key not in {"numeric_contract","target_status"}}
            descriptors={}
            names=[node["args"][0]["value"],node["args"][1]["value"],node["args"][2]["value"] if route["has_bias"] else None,node["id"]]
            for key,name in zip(("a","b","bias","output"),names,strict=True):
                if name is None:continue
                layout=layouts[name];binding=bindings[layout["root"]]
                descriptors[key]={"dtype":layout["dtype"],"shape":layout["shape"],"strides":layout["strides"],"offset":layout["offset"],"base":binding["base"],"bytes":binding["bytes"]}
            expectations.append({"program":p,**{key:route[key] for key in ("m","n","k","a_batch","b_batch","output_batch","transpose_b","has_bias")},**descriptors})
    (out/"files.json").write_text(json.dumps(files)+"\n")
    with (out/"build.log").open("w") as log:
        for cmd in (["cmake","-S",str(ROOT/"system_sim/physical_device"),"-B",str(BUILD),"-DCMAKE_BUILD_TYPE=Release"],["cmake","--build",str(BUILD),"--target","matrix-wire-dump","-j4"]):subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    binary=BUILD/"matrix-wire-dump"
    with (out/"decode.log").open("w") as log:subprocess.run([str(binary),str(out/"files.json"),str(out/"decoded.json")],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    decoded=json.loads((out/"decoded.json").read_text())["windows"]
    if len(decoded)!=len(expectations):raise RuntimeError("C++ wire decoder omitted model windows")
    for actual,expected in zip(decoded,expectations,strict=True):
        if not actual["backend_constructor_validated"] or any(actual.get(key)!=value for key,value in expected.items()):raise RuntimeError("C++ matrix wire decode differs from compiler metadata/microprogram")
    if sources!=identity() or any(sha(Path(path))!=digest for path,digest in inputs.items()):raise RuntimeError("matrix wire audit sources changed")
    report={"classification":"full_model_matrix_wire_compilation_and_decode_not_execution","sources":sources,"inputs":inputs,"source_nodes":len(program["nodes"]),"matrix_source_nodes":matrix_nodes,"matrix_windows":len(routes),"command_bytes_total":len(routes)*1088,
        "routes":routes,"decoded_sha256":sha(out/"decoded.json"),"decoder_binary_sha256":sha(binary),"all_cpp_decodes_equal":True,"model_data_executed":False,"mlx_system_verified":False,"inference_performance_eligible":False}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(f"MATRIX_WIRE_FULL_MODEL_LOWERING_PASS {out/'report.json'}")


if __name__=="__main__":main()
