"""Initialize all compiled model assets in the wide C++ memory, without execution."""
import argparse
import json
import subprocess
from pathlib import Path

from scripts.run_mlx_spike_graph import ROOT,sha
from scripts.run_mlx_clocked_chipyard import source_identity
from system_sim.physical_host.graph_lowering import compile_graph
from system_sim.physical_host.asset_source import source_manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--program",type=Path,required=True);parser.add_argument("--lifetimes",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);args=parser.parse_args();out=args.output.resolve()
    if out.exists():raise RuntimeError("choose a fresh wide-memory asset initialization directory")
    out.mkdir(parents=True)
    def identity():
        result=source_identity()
        for path in (Path(__file__).resolve(),ROOT/"tests/wide_memory_contract.cc"):result[str(path.relative_to(ROOT))]=sha(path)
        return result
    sources=identity();inputs={str(p.resolve()):sha(p) for p in (args.program,args.lifetimes)}
    program=json.loads(args.program.read_text());life=json.loads(args.lifetimes.read_text())
    for asset in program["assets"].values():
        if asset["kind"]=="mapped_file":
            path=Path(asset["path"]).resolve()
            if str(path) not in inputs:inputs[str(path)]=sha(path)
            if asset.get("file_sha256",inputs[str(path)])!=inputs[str(path)]:raise RuntimeError("model checkpoint fingerprint mismatch")
    blob,plan=compile_graph(program,life,device_base=0x88000000,device_bytes=16*2**30-128*2**20)
    (out/"commands.bin").write_bytes(blob);(out/"plan.json").write_text(json.dumps(plan,indent=2)+"\n")
    entries,manifest=source_manifest(program,out);expected={entry["value"]:entry for entry in entries};segments=[]
    for row in manifest["regions"]:
        segments.append({"name":row["name"],"path":row["path"],"file_offset":row["file_offset"],"file_bytes":row["bytes"],"memory_bytes":row["bytes"],"address":plan["assets"][row["name"]]["base"],"sha256":expected[row["name"]]["sha256"]})
    (out/"segments.json").write_text(json.dumps(segments,indent=2)+"\n");(out/"job.json").write_text(json.dumps({"base":0x80000000,"bytes":16*2**30,"segments":segments})+"\n")
    for path in (out/"literal_assets.bin",out/"segments.json",out/"job.json"):inputs[str(path)]=sha(path)
    build=ROOT/"build/mlx-wide-memory"
    with (out/"build.log").open("w") as log:
        for command in (["cmake","-S",str(ROOT/"system_sim/wide_memory"),"-B",str(build),"-DCMAKE_BUILD_TYPE=Release"],["cmake","--build",str(build),"--target","wide-memory-contract","-j4"]):subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    binary=build/"wide-memory-contract";binary_hash=sha(binary)
    with (out/"memory.json").open("w") as log:subprocess.run([str(binary),str(out/"job.json")],stdout=log,stderr=subprocess.PIPE,check=True,timeout=600)
    memory=json.loads((out/"memory.json").read_text());actual=memory["initialized_segments"]
    if len(actual)!=len(segments) or any(row!=expected for row,expected in zip(actual,segments)):raise RuntimeError("complete model asset preload readback differs")
    if memory["cycle"] or memory["ar_requests"] or memory["aw_requests"] or not memory["idle"]:raise RuntimeError("initialization mislabeled as a timed memory operation")
    if sources!=identity() or any(sha(Path(p))!=digest for p,digest in inputs.items()) or sha(binary)!=binary_hash:raise RuntimeError("wide asset initialization sources/inputs changed")
    report={"classification":"complete_compiled_model_asset_initialization_in_cpp_memory_not_inference_or_cpu_transfer","sources":sources,"inputs":inputs,"asset_count":len(segments),"asset_bytes":sum(row["file_bytes"] for row in segments),"compiled_source_calls":plan["source_calls"],
        "command_bytes":len(blob),"plan_sha256":sha(out/"plan.json"),"commands_sha256":sha(out/"commands.bin"),"memory_sha256":sha(out/"memory.json"),"binary_sha256":binary_hash,
        "full_model_execution_verified":False,"actual_cpu_execution":False,"mlx_system_verified":False,"inference_performance_eligible":False}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(f"WIDE_MODEL_ASSET_INITIALIZATION_PASS {out/'report.json'}")


if __name__=="__main__":main()
