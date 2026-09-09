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
from scripts.mlx_system_attempt import record,linked_libraries


def require(value, message):
    if not value: raise RuntimeError(message)


def audit_trace(program, result):
    if "event_patterns" in program:
        from mlxsim.model_lazy_event_patterns import materialize
        r=result["lazy_patterns"]
        if "rom_records_drained" in r:
            require(r["rom_records_drained"] and r["allocated_rom_record_slots"]<=program["hardware"]["rows"]*program["hardware"]["columns"]*program["hardware"]["contexts"],"lazy ROM records exceed residency or failed to drain")
        require(r["event_state_drained"] and r["peak_serialized_cache_bytes"]<=program["pattern_cache_bytes"] and r["peak_live_sequence_nodes"]<=1000000,"lazy pattern storage did not drain or exceeded bounds")
        require(r["allocated_event_slots"]==r["peak_live_event_leaves"],"lazy event slots grew beyond residency peak")
        if result["trace_truncated"]:return
        original=program;program=materialize(program)
        if "rom_records_drained" in r:
            definitions={b["id"]:b for b in program["blocks"]};transitions=[];live={};peak_rom=generations=0
            for b in result["block_intervals"]:
                if b["id"] not in definitions:continue
                key=(definitions[b["id"]]["pe"],definitions[b["id"]]["template"])
                transitions.extend([(b["admit_cycle"],1,key),(b["retire_cycle"],-1,key)])
            for _,delta,key in sorted(transitions):
                if delta<0:
                    live[key]-=1
                    if not live[key]:del live[key]
                else:
                    if key not in live:generations+=1
                    live[key]=live.get(key,0)+1;peak_rom=max(peak_rom,len(live))
            require(not live and (peak_rom,generations)==(r["allocated_rom_record_slots"],r["rom_record_generations"]),"ROM host record reuse differs from actual template leases")
        def size(items):
            nodes=leaves=0
            for item in items:
                nodes+=1
                if "repeat" in item:
                    n,l=size(item["body"]);nodes+=n;leaves+=l
                else:leaves+=1
            return nodes,leaves
        dimensions={b["id"]:size(b["events"]) for b in program["blocks"]+program.get("controllers",[])}
        changes={};totals=[0,0];peaks=[0,0]
        for b in result["block_intervals"]:
            n,l=dimensions[b["id"]]
            for cycle,sign in ((b["admit_cycle"],1),(b["retire_cycle"],-1)):
                delta=changes.setdefault(cycle,[0,0]);delta[0]+=sign*n;delta[1]+=sign*l
        for _,delta in sorted(changes.items()):
            for i in range(2):totals[i]+=delta[i];peaks[i]=max(peaks[i],totals[i])
        require(totals==[0,0] and peaks==[r["peak_live_sequence_nodes"],r["peak_live_event_leaves"]],"lazy resident IR peak does not match block lifetimes")
        block_patterns={b["id"]:b["event_pattern"] for b in original["blocks"]+original.get("controllers",[])}
        sizes={key:Path(spec["path"]).stat().st_size for key,spec in original["event_patterns"].items()};cache={};used=peak=reads=hits=stamp=0
        for row in result["trace"]:
            if row["event"]!="admit":continue
            stamp+=1;key=block_patterns[row["block"]]
            if key in cache:hits+=1
            else:
                while used+sizes[key]>original["pattern_cache_bytes"]:
                    victim=min(cache,key=cache.get);used-=sizes[victim];del cache[victim]
                reads+=1;used+=sizes[key];peak=max(peak,used)
            cache[key]=stamp
        require((reads,hits,peak)==(r["file_reads"],r["cache_hits"],r["peak_serialized_cache_bytes"]),"lazy cache counters differ from actual admission order")
    if result["trace_truncated"]: return
    control=program["schema"]=="mlx_event_schedule_v6"
    memory=program["schema"]=="mlx_event_schedule_v5" or control
    ports=program["schema"] in {"mlx_event_schedule_v3","mlx_event_schedule_v4"} or memory
    vector=program["schema"]=="mlx_event_schedule_v4" or memory
    if program["schema"]=="mlx_event_schedule_v2" or ports:
        program=copy.deepcopy(program);result=copy.deepcopy(result);occurrences=Counter()
        if memory:program["blocks"].extend({**controller,"controller":True,"pe":None} for controller in program["controllers"])
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
    pe_issues = set(); busy = {}; compute = []; dma = [];spm_ports=set();writebacks=set();compute_issues={};inflight={};sfu_issues={};pe_inflight={}
    def spm_claim(cycle):
        require(cycle%program["hardware"]["spm_port_period"]==0 and cycle not in spm_ports,"SPM port overcommit or phase mismatch");spm_ports.add(cycle)
    def writeback_claim(pe,cycle):
        key=pe,cycle
        require(cycle%program["hardware"]["writeback_period"]==0 and key not in writebacks,"RF writeback overcommit or phase mismatch");writebacks.add(key)
    origins={row["id"]:row["admit_cycle"] for row in result["block_intervals"]}
    for name, (block, spec) in specs.items():
        start = issued[name]["cycle"]; end = completed[name]["cycle"]
        latency=program["hardware"]["latencies"][spec["op"]]
        require(end-start>=latency if ports else end-start==latency, "event duration differs from architecture input")
        op=spec["op"]
        controller=block.get("controller",False)
        if not controller and (not ports or (not op.startswith("dma") and op not in {"operand_prepare","spm_initialize"})):
            key = (block["pe"], start); require(key not in pe_issues, "PE issued twice on one edge"); pe_issues.add(key)
        previous = None
        for e in block["events"]:
            if e["id"] == name: break
            previous = e["id"]
        for parent in set(spec["dependencies"]) | ({previous} if previous else set()):
            require(start >= completed[parent]["cycle"]+1, "event bypassed dependency visibility")
        if op in {"predicate_skip","operand_prepare","spm_initialize","memory_literal","memory_complete","control_literal","control_complete"}:
            require(end==start,"predicated/setup event occupied a service unit")
            if op=="spm_initialize":spm_claim(start)
            continue
        unit = ("control",0) if op.startswith("control_") else ("memory_convert",0) if op=="memory_convert" else ("dma", 0) if op.startswith("dma") or op in {"memory_index_read","memory_predicate_read"} else ("spm", 0) if op.startswith("spm") else ("sfu", block["pe"]) if op in {"exp", "div", "sqrt","cos","sin"} else ("compute", block["pe"])
        busy.setdefault(unit, []).append((start, end+1))
        inflight.setdefault(unit,[]).append((start,end))
        if not controller:pe_inflight.setdefault(block["pe"],[]).append((start,end))
        if controller and unit[0]=="dma":
            h=program["hardware"]["control_controller" if block.get("domain")=="control" else "memory_controller"];origin=origins[block["id"]]
            require((start-origin)%h["request_period"]==0 and (end-origin)%h["response_period"]==0,"memory controller local port phase differs")
        if ports and not controller:
            if unit[0]=="spm" or op=="dma_write":spm_claim(start)
            if unit[0]=="dma":
                require(start%program["hardware"]["dma_request_period"]==0 and end%program["hardware"]["dma_response_period"]==0,"DMA port period mismatch")
                if op=="dma_read":spm_claim(end)
            elif op=="spm_write":spm_claim(end)
            else:writeback_claim(block["pe"],end)
            if unit[0]=="compute":compute_issues.setdefault(block["pe"],[]).append(start)
            if unit[0]=="sfu":sfu_issues.setdefault(block["pe"],[]).append(start)
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
    require(areas["resident_context_cycles"] == sum(b["retire_cycle"]-b["admit_cycle"] for b in result["block_intervals"] if b.get("domain") not in {"memory_controller","control_controller"}), "residency integration differs")
    if ports:
        require(result["counts"].get("spm_port_claims",0)==len(spm_ports) and result["counts"].get("writeback_port_claims",0)==len(writebacks),"port claim totals differ")
        for issues in compute_issues.values():
            ordered=sorted(issues);require(all(b-a>=program["hardware"]["compute_ii"] for a,b in zip(ordered,ordered[1:])),"compute issue interval violated")
        for unit,key in (("compute","compute_inflight_pe_cycles"),("sfu","sfu_inflight_pe_cycles"),("spm","spm_inflight_cycles"),("dma","dma_inflight_cycles")):
            require(areas.get(key,0)==sum(b-a for (kind,_),values in inflight.items() if kind==unit for a,b in values),"inflight service occupancy differs")
        admitted=set();last={}
        for block in result["block_intervals"]:
            source=block["source_operator_id"];cycle=block["admit_cycle"]
            require((source,cycle) not in admitted and cycle>last.get(source,-1),"source admission rate/order violated")
            admitted.add((source,cycle));last[source]=cycle
            controller=block.get("domain") in {"memory_controller","control_controller"}
            require(all(row["cycle"]>=cycle if controller else row["cycle"]>cycle for row in issued.values() if row["block"]==block["id"]),"newly admitted context issued on the admission edge")
    if memory:
        intervals=[b for b in result["block_intervals"] if b.get("domain")=="memory_controller" and b["window_cycles"]>0]
        ordered=sorted((b["admit_cycle"],b["retire_cycle"]) for b in intervals)
        require(all(a[1]<=b[0] for a,b in zip(ordered,ordered[1:])),"multiple memory controllers simultaneously own private hardware")
        require(areas.get("memory_controller_resident_cycles",0)==sum(b-a for a,b in ordered),"private memory controller residency differs")
        require(areas.get("memory_conversion_inflight_cycles",0)==sum(b-a for a,b in inflight.get(("memory_convert",0),[])),"private conversion occupancy differs")
        r=result["memory_controller_resources"]
        require(r["data_register_bytes"]==32 and r["staging_bytes"]==128 and r["conversion_result_latch_bytes"]==8 and r["request_data_latch_bytes"]==8 and not r["uses_array_rf_spm"] and r["peak_active"]<=1,"memory private resources changed")
    if control:
        ordered=sorted((b["admit_cycle"],b["retire_cycle"]) for b in result["block_intervals"] if b.get("domain")=="control_controller")
        require(all(a[1]<=b[0] for a,b in zip(ordered,ordered[1:])),"control frontend residency overlapped")
        require(areas.get("control_controller_resident_cycles",0)==sum(b-a for a,b in ordered),"control residency integration differs")
        require(areas.get("control_instruction_inflight_cycles",0)==sum(b-a for a,b in inflight.get(("control",0),[])),"control instruction occupancy differs")
        r=result["control_controller_resources"]
        require(r["register_bytes"]==512 and r["rom_words"]==32 and r["peak_active"]<=1 and not r["uses_array_rf_spm"] and not r["rocket_cpu_timing"],"control private resources changed")
    if vector:
        for issues in sfu_issues.values():
            ordered=sorted(issues);require(all(b-a>=program["hardware"]["sfu_ii"] for a,b in zip(ordered,ordered[1:])),"SFU issue interval violated")
        overlap=0
        for pe in pe_inflight:
            overlap+=sum(max(0,min(b,d)-max(a,c)) for a,b in inflight.get(("compute",pe),[]) for c,d in inflight.get(("sfu",pe),[]))
        require(areas.get("compute_sfu_inflight_overlap_pe_cycles",0)==overlap,"vector/SFU overlap differs")
        overlap=0
        for spans in pe_inflight.values():
            changes=Counter()
            for a,b in spans:changes[a]+=1;changes[b]-=1
            previous=live=0
            for time,delta in sorted(changes.items()):
                if live>1:overlap+=time-previous
                live+=delta;previous=time
        require(areas.get("same_pe_inflight_context_overlap_pe_cycles",0)==overlap,"same-PE context overlap differs")
    if "source_graph" in program:
        rows={r["source_operator_id"]:r for r in result["source_intervals"]};blocks={b["id"]:b for b in result["block_intervals"]}
        require(result["source_graph_completed"] and set(rows)=={s["source_operator_id"] for s in program["source_graph"]},"source graph completion coverage differs")
        changes=Counter()
        for source in program["source_graph"]:
            r=rows[source["source_operator_id"]];begin=r["begin_cycle"];end=r["publish_cycle"]
            require(begin<end and all(rows[parent]["publish_cycle"]<=begin for parent in source["parents"]),"source graph dependency publication bypassed")
            require(not any(a<begin<b for a,b in dma),"source launched before shared DMA quiescence")
            require(len(r["windows"])==len(source["windows"]) and r["windows"][0]["begin_cycle"]==begin and r["windows"][-1]["end_cycle"]==end,"source batch coverage differs")
            previous=begin
            for index,(window,names) in enumerate(zip(r["windows"],source["windows"])):
                require(window["index"]==index and window["begin_cycle"]==previous,"source batch gap/order differs")
                require(all(blocks[name]["admit_cycle"]>=previous for name in names) and max(blocks[name]["retire_cycle"] for name in names)==window["end_cycle"],"source window starts/publishes before its blocks")
                previous=window["end_cycle"]
            changes[begin]+=1;changes[end]-=1
        live=peak=0
        for time,delta in sorted(changes.items()):live+=delta;peak=max(peak,live)
        require(live==0 and peak==result["source_frontend_peak"]<=program["hardware"]["source_window_limit"],"source frontend slots counted by blocks instead of source lifetime")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("tests", "binary", "output"): parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--include-loops",action="store_true")
    parser.add_argument("--include-ports",action="store_true")
    parser.add_argument("--include-vectors",action="store_true")
    parser.add_argument("--include-memory",action="store_true")
    parser.add_argument("--include-control",action="store_true")
    parser.add_argument("--include-source-order",action="store_true")
    parser.add_argument("--include-source-graph",action="store_true")
    parser.add_argument("--include-lazy",action="store_true")
    parser.add_argument("--include-rom-recycling",action="store_true")
    parser.add_argument("--case-timeout",type=int,default=60)
    args = parser.parse_args(); out = args.output.resolve(); tests = args.tests.resolve()
    require(1<=args.case_timeout<=600,"invalid safety case watchdog")
    if args.include_rom_recycling:args.include_lazy=True
    if args.include_lazy:args.include_source_graph=True
    if args.include_source_graph:args.include_source_order=True
    if args.include_source_order:args.include_control=True
    require(not out.exists(), "choose a fresh event verification directory")
    suite = ET.parse(tests / "regression.xml").getroot().find("testsuite")
    require(suite is not None and all(suite.get(k) == "0" for k in ("failures", "errors", "skipped")), "event regression failed")
    paths = [p for p in (tests / "pytest").rglob("program.json") if not any(a.is_symlink() for a in p.parents)
             and json.loads(p.read_text()).get("schema","").startswith("mlx_event_schedule_")]
    require(len(paths) == (328 if args.include_rom_recycling else 324 if args.include_lazy else 291 if args.include_source_graph else 272 if args.include_source_order else 202 if args.include_control else 166 if args.include_memory else 142 if args.include_vectors else 94 if args.include_ports else 58 if args.include_loops else 29), "event safety replay scope differs")
    sources = {str(p.relative_to(ROOT)): sha(p) for p in (ROOT / "simulator_ext/event_schedule").iterdir() if p.is_file()}
    for p in (Path(__file__).resolve(), ROOT / "tests/test_event_schedule.py", ROOT / "src/mlxsim/model_event_resources.py"):
        sources[str(p.relative_to(ROOT))] = sha(p)
    if args.include_loops or args.include_ports or args.include_vectors or args.include_memory or args.include_control:sources["tests/test_event_loops.py"]=sha(ROOT/"tests/test_event_loops.py")
    if args.include_ports or args.include_vectors or args.include_memory or args.include_control:
        for name in ("tests/test_event_ports.py","src/mlxsim/model_matrix_events.py","tests/test_matrix_window_scheduler.py"):
            sources[name]=sha(ROOT/name)
    if args.include_vectors or args.include_memory or args.include_control:
        for name in ("tests/test_vector_events.py","src/mlxsim/model_vector_events.py","tests/test_vector_window_scheduler.py"):
            sources[name]=sha(ROOT/name)
    if args.include_memory or args.include_control:
        for name in ("tests/test_memory_events.py","src/mlxsim/model_memory_events.py","tests/test_memory_window_scheduler.py"):
            sources[name]=sha(ROOT/name)
    if args.include_control:
        for name in ("tests/test_control_events.py","src/mlxsim/model_control_events.py","tests/test_control_window_scheduler.py"):
            sources[name]=sha(ROOT/name)
    if args.include_source_order:
        for name in ("tests/test_event_source_order.py","src/mlxsim/model_array_group_events.py","simulator_ext/event_alignment/main.cc","simulator_ext/event_alignment/CMakeLists.txt"):
            sources[name]=sha(ROOT/name)
    if args.include_source_graph:
        for name in ("tests/test_event_source_graph.py","tests/test_event_graph_plan.py","src/mlxsim/model_array_graph_events.py","src/mlxsim/model_event_graph_plan.py"):
            sources[name]=sha(ROOT/name)
    if args.include_lazy:
        for name in ("tests/test_lazy_event_patterns.py","src/mlxsim/model_lazy_event_patterns.py"):
            sources[name]=sha(ROOT/name)
    binary_hash = sha(args.binary); files = {str(p): sha(p) for p in paths};missing=set();libraries=linked_libraries(args.binary.resolve())
    for p in paths:
        for spec in json.loads(p.read_text()).get("event_patterns",{}).values():
            path=Path(spec["path"])
            if path.is_file():files[str(path)]=sha(path)
            else:missing.add(str(path))
    out.mkdir(parents=True)
    replays = []; env = dict(os.environ, ASAN_OPTIONS="detect_leaks=1:halt_on_error=1", UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1")
    for i, p in enumerate(sorted(paths)):
        actual = out / f"case-{i:02d}.json"; log = out / f"case-{i:02d}.log"; expected = 0 if (p.parent / "result.json").exists() else 1
        with log.open("w") as stream:
            process = subprocess.run([str(args.binary.resolve()), str(p), str(actual)], stdout=stream, stderr=subprocess.STDOUT, env=env, timeout=args.case_timeout)
        require(process.returncode == expected and not any(x in log.read_text() for x in ("ERROR: AddressSanitizer", "runtime error:", "LeakSanitizer", "DEADLYSIGNAL")), "event sanitizer replay failed")
        if expected: require(not actual.exists(), "rejected event program produced success")
        else:
            result = json.loads(actual.read_text()); baseline = json.loads((p.parent / "result.json").read_text())
            require(result == baseline, "event sanitizer changed full result")
            audit_trace(json.loads(p.read_text()), result)
        replays.append(dict(program=str(p), expected_exit=expected, result=str(actual) if not expected else None, log_sha256=sha(log)))
    require(sum(r["expected_exit"] == 0 for r in replays) == (277 if args.include_rom_recycling else 273 if args.include_lazy else 248 if args.include_source_graph else 236 if args.include_source_order else 168 if args.include_control else 136 if args.include_memory else 115 if args.include_vectors else 67 if args.include_ports else 37 if args.include_loops else 18), "event safety success coverage differs")
    require(all(sha(ROOT / p) == h for p,h in sources.items()) and all(sha(Path(p)) == h for p,h in {**files,**libraries}.items()) and all(not Path(p).exists() for p in missing) and sha(args.binary) == binary_hash, "event safety sources/inputs changed")
    record(out / "report.json", dict(classification="concurrent_event_core_component_validation_not_full_model", sources=sources, inputs=files,
                                     regression_tests=int(suite.get("tests")), replays=replays, asan_binary_sha256=binary_hash, full_model_verified=False,
                                     performance_error_available=False, trace_resource_accounting_checked=True,runtime_libraries=libraries,missing_pattern_paths=sorted(missing),case_watchdog_seconds=args.case_timeout))
    print(f"EVENT_CORE_SAFETY_PASS replays={len(replays)}")


if __name__ == "__main__": main()
