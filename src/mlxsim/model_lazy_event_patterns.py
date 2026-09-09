"""Externalize local block IR without changing source/resource/timing contracts."""
import copy
import hashlib
import json
from pathlib import Path


def externalize(program,directory,cache_bytes=8*1024*1024):
    if program.get("schema")!="mlx_event_schedule_v6" or not program.get("source_tick_order") or "event_patterns" in program:raise ValueError("lazy patterns require source-ordered v6 input")
    if type(cache_bytes) is not int or not 1<=cache_bytes<=268435456:raise ValueError("invalid pattern cache capacity")
    directory=Path(directory).resolve()
    if directory.exists():raise ValueError("choose a fresh block pattern directory")
    result=copy.deepcopy(program);patterns={};payloads={}
    for block in result["blocks"]+result.get("controllers",[]):
        if block["admission_dependencies"]:raise ValueError("lazy blocks require source graph dependencies")
        serial=0
        def normalize(items):
            nonlocal serial
            total=0
            for item in items:
                if "repeat" in item:total+=item["repeat"]*normalize(item["body"])
                else:
                    if item["dependencies"]:raise ValueError("lazy patterns cannot hide cross-block event dependencies")
                    item["id"]=f"e{serial}";serial+=1;total+=1
            return total
        events=block.pop("events");total=normalize(events)
        payload=json.dumps(dict(schema="mlx_block_event_pattern_v1",events=events),sort_keys=True,separators=(",",":"))+"\n"
        raw=payload.encode();key=hashlib.sha256(raw).hexdigest()
        if len(raw)>cache_bytes:raise ValueError("block event pattern exceeds configured cache")
        zero=total==1 and len(events)==1 and events[0].get("op") in {"memory_complete","control_complete"}
        patterns[key]=dict(path=str(directory/(key+".json")),sha256=key,dynamic_events=total,zero_work=zero)
        payloads[key]=payload;block["event_pattern"]=key
    directory.mkdir(parents=True)
    for key,payload in payloads.items():(directory/(key+".json")).write_text(payload)
    result["event_patterns"]=patterns;result["pattern_cache_bytes"]=cache_bytes
    return result


def materialize(program):
    """Audit projection: restore exactly the namespaced IR seen by the C++ loader."""
    p=copy.deepcopy(program);patterns=p.pop("event_patterns");p.pop("pattern_cache_bytes")
    for block in p["blocks"]+p.get("controllers",[]):
        spec=patterns[block.pop("event_pattern")];raw=Path(spec["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=spec["sha256"]:raise ValueError("event pattern digest changed")
        value=json.loads(raw);events=value["events"]
        def rename(items):
            for item in items:
                if "repeat" in item:rename(item["body"])
                else:item["id"]=block["id"]+":"+item["id"]
        rename(events);block["events"]=events
    return p
