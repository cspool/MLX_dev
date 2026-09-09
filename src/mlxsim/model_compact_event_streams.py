"""Lossless block-run representation with stable source/window/block identities."""
import copy


def compact_streams(program,interval_limit=1000000):
    if program.get("schema")!="mlx_event_schedule_v6" or not program.get("source_tick_order") or "event_patterns" not in program or "source_graph" not in program:
        raise ValueError("compact streams require lazy source-ordered graph input")
    if type(interval_limit) is not int or not 0<=interval_limit<=1000000:raise ValueError("invalid compact interval limit")
    p=copy.deepcopy(program);blocks={b["id"]:b for b in p["blocks"]+p["controllers"]};used=set();streams=[];pes=p["hardware"]["rows"]*p["hardware"]["columns"]
    for source in p.pop("source_graph"):
        private=source["family"] in {"memory","control"};windows=[];own=[]
        for names in source["windows"]:
            runs=[]
            for name in names:
                if name not in blocks or name in used:raise ValueError("missing or repeated compact block")
                b=blocks[name];used.add(name);own.append(name)
                if b["source_operator_id"]!=source["source_operator_id"] or b["admission_dependencies"]:raise ValueError("compact block owner/dependency differs")
                row=dict(count=1,event_pattern=b["event_pattern"])
                if private:
                    if b.get("domain")!=source["family"]:raise ValueError("compact private domain differs")
                else:row.update(template=b["template"],pe_base=b["pe"],pe_stride=0)
                if runs and all(runs[-1][key]==row[key] for key in (("event_pattern",) if private else ("event_pattern","template"))):
                    last=runs[-1]
                    if private:last["count"]+=1;continue
                    if last["count"]==1:last["pe_stride"]=(b["pe"]-last["pe_base"])%pes
                    if (last["pe_base"]+last["count"]*last["pe_stride"])%pes==b["pe"]:last["count"]+=1;continue
                runs.append(row)
            windows.append(dict(runs=runs))
        expected=[b["id"] for b in program["blocks"]+program["controllers"] if b["source_operator_id"]==source["source_operator_id"]]
        if own!=expected:raise ValueError("compact windows must preserve original source block order")
        streams.append(dict(source_operator_id=source["source_operator_id"],family=source["family"],parents=source["parents"],windows=windows))
    if len(used)!=len(blocks):raise ValueError("compact graph omitted blocks")
    p.update(source_streams=streams,blocks=[],controllers=[],block_interval_limit=interval_limit)
    return p


def expand_streams(program,limit=500000):
    """Bounded audit projection only; the C++ execution path never calls this."""
    p=copy.deepcopy(program);sources=p.pop("source_streams");p.pop("block_interval_limit");p["source_graph"]=[];p["blocks"]=[];p["controllers"]=[]
    pes=p["hardware"]["rows"]*p["hardware"]["columns"];count=0
    for s in sources:
        windows=[];private=s["family"] in {"control","memory"}
        for wi,w in enumerate(s["windows"]):
            names=[];ordinal=0
            for run in w["runs"]:
                count+=run["count"]
                if count>limit:raise ValueError("compact audit expansion exceeds explicit bound")
                for i in range(run["count"]):
                    name=f"{s['source_operator_id']}:{wi}:b{ordinal}";ordinal+=1;names.append(name)
                    b=dict(id=name,source_operator_id=s["source_operator_id"],admission_dependencies=[],event_pattern=run["event_pattern"])
                    if private:b["domain"]=s["family"];p["controllers"].append(b)
                    else:b.update(pe=(run["pe_base"]+i*run["pe_stride"])%pes,template=run["template"]);p["blocks"].append(b)
            windows.append(names)
        p["source_graph"].append(dict(source_operator_id=s["source_operator_id"],family=s["family"],parents=s["parents"],windows=windows))
    return p
