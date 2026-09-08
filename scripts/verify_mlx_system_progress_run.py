"""Verify completed real-system progress A/B preflight, not the following model run."""
import argparse
import json
from pathlib import Path

from scripts.mlx_system_attempt import digest,record


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--attempt",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);args=parser.parse_args();root=args.attempt.resolve()
    if args.output.exists():raise RuntimeError("choose a fresh progress preflight report")
    plain=root/"preflight-plain";observed=root/"preflight-observed";states=[];devices=[];hashes={}
    for path in (plain,observed):
        state=json.loads((path/"execution.json").read_text());device=json.loads((path/"device.json").read_text());states.append(state);devices.append(device)
        if state["status"]!="exited" or state["exit_code"]!=0 or state.get("validation")!="registered_graph_checks_passed":raise RuntimeError("preflight has no validated real-system terminal result")
        if "MLX_CLOCKED_CHAIN_PASS" not in (path/"chipyard.log").read_text():raise RuntimeError("CPU output checks did not finish")
        for name in ("execution.json","device.json","chipyard.log","test.elf","launch-map.json"):hashes[str(path/name)]=digest(path/name)
    if states[0]["seed"] is None or states[0]["seed"]!=states[1]["seed"] or states[0]["simulator_sha256"]!=states[1]["simulator_sha256"] or states[0]["elf_sha256"]!=states[1]["elf_sha256"]:raise RuntimeError("A/B simulator, seed or ELF differ")
    observer=devices[1].pop("progress_observer")
    if devices[0]!=devices[1] or observer["failed"]:raise RuntimeError("progress changed complete target report or observer failed")
    mapping=json.loads((observed/"launch-map.json").read_text());events=[json.loads(line) for line in (observed/"progress.jsonl").read_text().splitlines()]
    if sum(row["launch_event"] for row in events)!=len(mapping) or sum(row["terminal_event"] for row in events)!=len(mapping):raise RuntimeError("progress boundary count differs from compiled task count")
    for event in events:
        if event["launches"] and event["source"]!=mapping[event["launches"]-1]:raise RuntimeError("progress source join differs from compiled map")
    for path in (observed/"progress.json",observed/"progress.jsonl",observed/"memory-init.json"):hashes[str(path)]=digest(path)
    for name,expected in states[1]["sources"].items():
        if digest(root/"sources"/name)!=expected:raise RuntimeError("attempt source snapshot changed")
    record(args.output,{"classification":"real_rocket_progress_preflight_not_full_model_completion","evidence":hashes,"source_calls":states[1]["source_calls"],"device_windows":len(mapping),"target_reports_equal":True,"seed":states[1]["seed"],
        "full_model_execution_verified":False,"mlx_system_verified":False,"inference_performance_eligible":False})
    print(f"SYSTEM_PROGRESS_PREFLIGHT_PASS {args.output}")


if __name__=="__main__":main()
