"""Sanitizer replay of event calendar components; no full-model certification."""
import argparse
import copy
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from scripts.run_mlx_tensor_semantics import sha, ROOT
from scripts.mlx_system_attempt import record


def require(value, message):
    if not value: raise RuntimeError(message)


def audit_trace(program, result):
    if result["trace_truncated"]: return
    if program["schema"]=="mlx_event_schedule_v2":
        program=copy.deepcopy(program);result=copy.deepcopy(result);occurrences=Counter()
        def expand(items):
            for item in items:
                if "repeat" in item:
                    for _ in range(item["repeat"]):yield from expand(item["body"])
                else:
                    row=copy.deepcopy(item);key=row["id"];row["id"]=json.dumps([key,occurrences[key]])
                    occurrences[key]+=1;yield row
        require(result["events"]<=500000,"untruncated event trace exceeds audit bound")
        for block in program["blocks"]:block["events"]=list(expand(block["events"]))
        for block in program["blocks"]:
            for event in block["events"]:event["dependencies"]=[json.dumps([name,occurrences[name]-1]) for name in event["dependencies"]]
        for row in result["trace"]:
            if "operation" in row:
                require(type(row.get("instance")) is int and row["instance"]>=0,"loop trace lacks an instance identity")
                row["operation"]=json.dumps([row["operation"],row["instance"]])
    specs = {e["id"]: (b, e) for b in program["blocks"] for e in b["events"]}
    issued = {e["operation"]: e for e in result["trace"] if e["event"] == "issue"}
    completed = {e["operation"]: e for e in result["trace"] if e["event"] == "complete"}
    require(set(issued) == set(completed) == set(specs), "event trace source coverage differs")
    require(sum(row["event"]=="issue" for row in result["trace"])==len(specs)
            and sum(row["event"]=="complete" for row in result["trace"])==len(specs),"event trace duplicated an instance")
    pe_issues = set(); busy = {}; compute = []; dma = []
    for name, (block, spec) in specs.items():
        start = issued[name]["cycle"]; end = completed[name]["cycle"]
        require(end-start == program["hardware"]["latencies"][spec["op"]], "event duration differs from architecture input")
        key = (block["pe"], start); require(key not in pe_issues, "PE issued twice on one edge"); pe_issues.add(key)
        previous = None
        for e in block["events"]:
            if e["id"] == name: break
            previous = e["id"]
        for parent in set(spec["dependencies"]) | ({previous} if previous else set()):
            require(start >= completed[parent]["cycle"]+1, "event bypassed dependency visibility")
        op = spec["op"]
        unit = ("dma", 0) if op.startswith("dma") else ("spm", 0) if op.startswith("spm") else ("sfu", block["pe"]) if op in {"exp", "div", "sqrt"} else ("compute", block["pe"])
        busy.setdefault(unit, []).append((start, end+1))
        if unit[0] == "compute": compute.append((start, end+1))
        if unit[0] == "dma": dma.append((start, end+1))
    for intervals in busy.values():
        ordered = sorted(intervals)
        require(all(a[1] <= b[0] for a, b in zip(ordered, ordered[1:])), "exclusive service overlapped itself")
    areas = result["integrated_usage"]
    require(areas.get("compute_busy_pe_cycles", 0) == sum(b-a for a,b in compute), "compute occupancy integration differs")
    require(areas.get("dma_busy_cycles", 0) == sum(b-a for a,b in dma), "DMA occupancy integration differs")
    overlap = sum(max(0, min(b,d)-max(a,c)) for a,b in compute for c,d in dma)
    require(areas.get("compute_dma_overlap_pe_cycles", 0) == overlap, "compute/DMA overlap double counted")
    require(areas["resident_context_cycles"] == sum(b["retire_cycle"]-b["admit_cycle"] for b in result["block_intervals"]), "residency integration differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("tests", "binary", "output"): parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--include-loops",action="store_true")
    args = parser.parse_args(); out = args.output.resolve(); tests = args.tests.resolve()
    require(not out.exists(), "choose a fresh event verification directory")
    suite = ET.parse(tests / "regression.xml").getroot().find("testsuite")
    require(suite is not None and all(suite.get(k) == "0" for k in ("failures", "errors", "skipped")), "event regression failed")
    paths = [p for p in (tests / "pytest").rglob("program.json") if not any(a.is_symlink() for a in p.parents)]
    require(len(paths) == (58 if args.include_loops else 29), "event safety replay scope differs")
    sources = {str(p.relative_to(ROOT)): sha(p) for p in (ROOT / "simulator_ext/event_schedule").iterdir() if p.is_file()}
    for p in (Path(__file__).resolve(), ROOT / "tests/test_event_schedule.py", ROOT / "src/mlxsim/model_event_resources.py"):
        sources[str(p.relative_to(ROOT))] = sha(p)
    if args.include_loops:sources["tests/test_event_loops.py"]=sha(ROOT/"tests/test_event_loops.py")
    binary_hash = sha(args.binary); files = {str(p): sha(p) for p in paths}; out.mkdir(parents=True)
    replays = []; env = dict(os.environ, ASAN_OPTIONS="detect_leaks=1:halt_on_error=1", UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1")
    for i, p in enumerate(sorted(paths)):
        actual = out / f"case-{i:02d}.json"; log = out / f"case-{i:02d}.log"; expected = 0 if (p.parent / "result.json").exists() else 1
        with log.open("w") as stream:
            process = subprocess.run([str(args.binary.resolve()), str(p), str(actual)], stdout=stream, stderr=subprocess.STDOUT, env=env, timeout=60)
        require(process.returncode == expected and not any(x in log.read_text() for x in ("ERROR: AddressSanitizer", "runtime error:", "LeakSanitizer", "DEADLYSIGNAL")), "event sanitizer replay failed")
        if expected: require(not actual.exists(), "rejected event program produced success")
        else:
            result = json.loads(actual.read_text()); baseline = json.loads((p.parent / "result.json").read_text())
            require(result == baseline, "event sanitizer changed full result")
            audit_trace(json.loads(p.read_text()), result)
        replays.append(dict(program=str(p), expected_exit=expected, result=str(actual) if not expected else None, log_sha256=sha(log)))
    require(sum(r["expected_exit"] == 0 for r in replays) == (37 if args.include_loops else 18), "event safety success coverage differs")
    require(all(sha(ROOT / p) == h for p,h in sources.items()) and all(sha(Path(p)) == h for p,h in files.items()) and sha(args.binary) == binary_hash, "event safety sources/inputs changed")
    record(out / "report.json", dict(classification="concurrent_event_core_component_validation_not_full_model", sources=sources, inputs=files,
                                     regression_tests=int(suite.get("tests")), replays=replays, asan_binary_sha256=binary_hash, full_model_verified=False,
                                     performance_error_available=False, trace_resource_accounting_checked=True))
    print(f"EVENT_CORE_SAFETY_PASS replays={len(replays)}")


if __name__ == "__main__": main()
