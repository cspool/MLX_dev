"""Check full-model vector wire lowering/decoding, not model execution."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from scripts.run_mlx_physical_model import source_identity as runtime_sources
from system_sim.physical_host.address_plan import iter_bindings
from system_sim.physical_device.vector_lowering import lower_vector,double_bits

ROOT=Path(__file__).resolve().parents[1]
BUILD=ROOT/"build/mlx-spike-matrix"
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--program",type=Path,required=True);parser.add_argument("--lifetimes",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);args=parser.parse_args();out=args.output.resolve()
    if out.exists():raise RuntimeError("choose a fresh vector wire audit directory")
    out.mkdir(parents=True);inputs={str(p.resolve()):sha(p) for p in (args.program,args.lifetimes)}
    def identity():
        result=runtime_sources();files=[Path(__file__).resolve()]
        for name in ("physical_host","physical_device"):files.extend(p for p in (ROOT/"system_sim"/name).iterdir() if p.is_file())
        for p in files:result[str(p.relative_to(ROOT))]=sha(p)
        return result
    sources=identity();program=json.loads(args.program.read_text());life=json.loads(args.lifetimes.read_text());routes=[];expected=[];files=[]
    for node,layouts,bindings in iter_bindings(program,life):
        if "vector_program" not in node:continue
        blob,route=lower_vector(node,layouts,bindings);file=out/f"vector-{node['source_operator_id']}.bin";file.write_bytes(blob);files.append(str(file));route.update(command_file=str(file),command_sha256=sha(file));routes.append(route)
        operands=[]
        for arg in node["args"][:len(node["vector_program"]["input_dtypes"])]:
            if isinstance(arg,dict) and "value" in arg:
                layout=layouts[arg["value"]];binding=bindings[layout["root"]]
                operands.append({"tensor":{key:layout[key] for key in ("dtype","shape","strides","offset")}|{"bytes":binding["bytes"]},"base":binding["base"]})
            else:operands.append({"literal_bits":double_bits(arg)})
        output=layouts[node["id"]];binding=bindings[output["root"]]
        expected.append({"program":node["vector_program"],"operands":operands,"output":node["output"],"output_base":binding["base"],"output_bytes":binding["bytes"]})
    (out/"files.json").write_text(json.dumps(files)+"\n")
    with (out/"build.log").open("w") as log:
        for command in (["cmake","-S",str(ROOT/"system_sim/physical_device"),"-B",str(BUILD),"-DCMAKE_BUILD_TYPE=Release"],["cmake","--build",str(BUILD),"--target","vector-wire-dump","-j4"]):subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    binary=BUILD/"vector-wire-dump"
    with (out/"decode.log").open("w") as log:subprocess.run([str(binary),str(out/"files.json"),str(out/"decoded.json")],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    decoded=json.loads((out/"decoded.json").read_text())["windows"]
    if len(decoded)!=len(expected):raise RuntimeError("vector wire windows omitted")
    for actual,wanted in zip(decoded,expected,strict=True):
        if not actual["backend_constructor_validated"] or actual["node"]["vector_program"]!=wanted["program"] or actual["node"]["output"]!=wanted["output"]:raise RuntimeError("vector program/phase/constant/output roundtrip changed")
        for i,operand in enumerate(wanted["operands"]):
            if "tensor" in operand:
                if actual["values"]["a" if i==0 else "b"]!=operand["tensor"] or actual["regions"][i]["base"]!=operand["base"]:raise RuntimeError("vector operand layout/address roundtrip changed")
            elif double_bits(actual["node"]["args"][i])!=operand["literal_bits"]:raise RuntimeError("vector scalar literal bits changed")
        if actual["regions"][2]["base"]!=wanted["output_base"] or actual["regions"][2]["bytes"]!=wanted["output_bytes"]:raise RuntimeError("vector output physical binding changed")
    if sources!=identity() or any(sha(Path(p))!=digest for p,digest in inputs.items()):raise RuntimeError("vector lowering audit inputs changed")
    report={"classification":"full_model_vector_wire_compilation_and_decode_not_execution","sources":sources,"inputs":inputs,"source_nodes":len(program["nodes"]),"vector_source_nodes":len(routes),"command_bytes_total":len(routes)*4288,
        "routes":routes,"all_cpp_decodes_equal":True,"decoded_sha256":sha(out/"decoded.json"),"decoder_binary_sha256":sha(binary),"model_data_executed":False,"mlx_system_verified":False,"inference_performance_eligible":False}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(f"VECTOR_WIRE_FULL_MODEL_LOWERING_PASS {out/'report.json'}")


if __name__=="__main__":main()
