"""Explicit source lifecycle over complete array windows; not full model lowering."""
import copy

from .model_array_group_events import array_group_events,native_group_job


def array_graph_events(job):
    if set(job)-{"schema","windows","sources","max_active_sources"} or job["schema"]!="mlx_array_event_graph_v1":raise ValueError("expected array source graph")
    p=array_group_events(dict(schema="mlx_array_event_group_v1",windows=job["windows"]))
    limit=job.get("max_active_sources",32)
    if type(limit) is not int or not 1<=limit<=64:raise ValueError("invalid graph source capacity")
    p["hardware"]["source_window_limit"]=limit;p["source_graph"]=[];seen=set();assigned=[]
    blocks={index:[b for b in p["blocks"] if b["source_operator_id"]==index] for index in range(len(job["windows"]))}
    for source in job["sources"]:
        if set(source)!={"source_operator_id","parents","windows"}:raise ValueError("invalid graph source descriptor")
        identity=source["source_operator_id"];parents=source["parents"];indices=source["windows"]
        if type(identity) is not int or identity<0 or identity in seen or not isinstance(parents,list) or any(type(x) is not int or x not in seen for x in parents) or len(set(parents))!=len(parents):raise ValueError("graph source identities/parents invalid")
        if not isinstance(indices,list) or not indices or any(type(x) is not int or x not in blocks or x in assigned for x in indices):raise ValueError("graph source windows invalid")
        seen.add(identity);assigned.extend(indices)
        families={"matrix" if job["windows"][i]["schema"]=="mlx_matrix_window_job_v1" else "vector" for i in indices}
        if len(families)!=1:raise ValueError("one source cannot change its array family")
        windows=[]
        for index in indices:
            windows.append([b["id"] for b in blocks[index]])
            for block in blocks[index]:block["source_operator_id"]=identity
        p["source_graph"].append(dict(source_operator_id=identity,family=families.pop(),parents=parents,windows=windows))
    if assigned!=list(range(len(job["windows"]))):raise ValueError("graph must partition all windows in source/batch order")
    # Every input referring to another window must have a source dependency;
    # no uninitialized data may be smuggled in as an independent asset.
    owners={w:s["source_operator_id"] for s in job["sources"] for w in s["windows"]}
    ancestors={}
    for s in job["sources"]:
        ancestors[s["source_operator_id"]]=set(s["parents"])
        for parent in s["parents"]:ancestors[s["source_operator_id"]]|=ancestors[parent]
    for index,w in enumerate(job["windows"]):
        assets=w["assets"].values() if "assets" in w else [w[k] for k in ("a","b","bias") if k in w]
        for asset in assets:
            if asset.get("kind")!="group_output":continue
            parent=asset.get("window")
            if type(parent) is not int or not 0<=parent<index:raise ValueError("graph input has a forward/missing window")
            owner=owners[index]
            if owners[parent]!=owner and owners[parent] not in ancestors[owner]:raise ValueError("graph input lacks producer dependency")
            pw=job["windows"][parent];out=pw["node"]["output"] if "node" in pw else dict(shape=[pw["m"],pw["n"]],dtype=pw["program"]["output_dtype"])
            if any(asset[k]!=out[k] for k in ("shape","dtype")):raise ValueError("graph input differs from producer layout")
    return p


def native_array_graph_job(job):
    p=array_graph_events(job)
    native=native_group_job(dict(schema="mlx_array_event_group_v1",windows=job["windows"]))
    native["sources"]=copy.deepcopy(job["sources"]);native["max_active_sources"]=p["hardware"]["source_window_limit"]
    return native
