"""Link audited full-model window patterns directly to compact C++ source streams.

Only one unique window pattern is expanded while linking; logical model block
instances are represented as run counts, not as a model-wide block list.
"""
import copy
import hashlib
import json
from pathlib import Path

from .model_event_graph_plan import graph_plan
from .model_block_pipeline import compile_block_pipelines


def canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":"))


def hardware_contract(program,options):
    if options.get("overlap",True) is not True or options.get("pipeline_whole_source_barrier",False) is not False:raise ValueError("linker requires concurrent non-barrier execution")
    if options.get("template_word_period",1)!=1:raise ValueError("template word periods other than one are not linked")
    mo=program.get("matrix_schedule_options",{});vo=program.get("vector_schedule_options",{});me=program.get("memory_schedule_options",{});co=program.get("control_schedule_options",{})
    common=dict(rows=4,columns=4,contexts=2,spm_period=1,writeback_period=1,multiply_latency=4,add_latency=2,convert_latency=2,spm_latency=3,dma_request_period=1,dma_response_period=1)
    for key,default in common.items():
        if mo.get(key,default)!=vo.get(key,default):raise ValueError("full model matrix/vector shared hardware differs")
    if mo.get("compute_ii",1)!=vo.get("vector_ii",1):raise ValueError("full model shared compute II differs")
    h={key:mo.get(key,default) for key,default in common.items() if key in {"rows","columns","contexts","dma_request_period","dma_response_period"}}
    h.update(rf_vectors_per_pe=16,spm_vectors_total=128,rom_words_per_pe=32,source_window_limit=options.get("max_active_nodes",32),template_load_timing=options.get("template_load_timing",False),
             spm_port_period=mo.get("spm_period",1),writeback_period=mo.get("writeback_period",1),compute_ii=mo.get("compute_ii",1),sfu_ii=vo.get("trans_ii",1))
    add=mo.get("add_latency",2);spm=mo.get("spm_latency",3);exp=vo.get("exp_latency",8);dma=mo.get("dma_latency",8)
    h["latencies"]=dict(zero=1,mul=mo.get("multiply_latency",4),add=add,convert=mo.get("convert_latency",2),exp=exp,div=vo.get("div_latency",12),sqrt=vo.get("sqrt_latency",12),
        spm_read=spm,spm_write=spm,dma_read=dma,dma_write=dma,predicate_skip=0,operand_prepare=0,spm_initialize=0,neg=1,sub=add,maximum=add,constant=1,move=1,broadcast=1,shuffle=1,cos=exp,sin=exp,
        memory_literal=0,memory_convert=me.get("convert_latency",2),memory_complete=0,memory_index_read=me.get("dma_latency",8),memory_predicate_read=me.get("dma_latency",8),
        control_alu=co.get("alu_latency",1),control_multiply=co.get("multiply_latency",3),control_float=co.get("float_latency",3),control_branch=co.get("branch_latency",1),control_literal=0,control_complete=0)
    h["memory_controller"]=dict(data_register_bytes=32,staging_bytes=128,conversion_result_latch_bytes=8,request_data_latch_bytes=8,max_active=1,request_period=me.get("request_period",1),response_period=me.get("response_period",1))
    h["control_controller"]=dict(register_bytes=512,rom_words=32,max_active=1,request_period=co.get("request_period",1),response_period=co.get("response_period",1))
    memory={key:options.get("memory",{}).get(key,default) for key,default in dict(latency=8,accept_period=1,nack_every=0).items()}
    if memory["nack_every"]!=0:raise ValueError("physical memory retry linking is not implemented")
    return h,memory


class ModelEventLinker:
    def __init__(self,program,manifest,read_pattern,directory,options):
        self.program=program;self.manifest=manifest;self.read_pattern=read_pattern;self.directory=Path(directory).resolve();self.options=options
        if not manifest["all_window_patterns_compiled"] or manifest["blocked_windows"]:raise ValueError("cannot link a model with missing actual branch witnesses")
        self.plan=graph_plan(program);self.hardware,self.memory=hardware_contract(program,options)
        self.nodes={n["source_operator_id"]:n for n in program["nodes"]};self.templates={};self.patterns={};self.payloads={};self.window_runs={};self.pattern_inputs={}
        self.logical_blocks=self.dynamic_events=self.unique_window_blocks_scanned=0

    def template(self,node,spec):
        if "matrix_program" in node:key=["matrix",[len(node["matrix_program"][part]) for part in ("prologue","body","epilogue")]]
        else:key=["vector",node["vector_program"]["phases"]]
        identity=hashlib.sha256(canonical([key,spec["words"]]).encode()).hexdigest();value={k:copy.deepcopy(spec[k]) for k in ("words","rf_vectors","spm_vectors")};value["id"]="code:"+identity
        if identity in self.templates and self.templates[identity]!=value:raise ValueError("shared ROM identity has different context demands")
        self.templates[identity]=value;return value["id"]

    def block_pattern(self,events,window):
        serial=0
        def normalize(items):
            nonlocal serial
            total=0
            for item in items:
                if "repeat" in item:total+=item["repeat"]*normalize(item["body"])
                else:
                    if item["dependencies"]:raise ValueError("window contains an unlinked cross-block event edge")
                    op=item["op"]
                    if not (op.startswith("dma_") or op in {"memory_index_read","memory_predicate_read"}) and window["hardware"]["latencies"][op]!=self.hardware["latencies"][op]:raise ValueError("window execution latency differs from full target")
                    item["id"]=f"e{serial}";serial+=1;total+=1
            return total
        total=normalize(events);raw=(canonical(dict(schema="mlx_block_event_pattern_v1",events=events))+"\n").encode();key=hashlib.sha256(raw).hexdigest()
        zero=total==1 and len(events)==1 and events[0].get("op") in {"memory_complete","control_complete"}
        if key not in self.patterns:
            self.patterns[key]=dict(path=str(self.directory/(key+".json")),sha256=key,dynamic_events=total,zero_work=zero);self.payloads[key]=raw
        return key

    def compile_runs(self,key,node,family):
        if key in self.window_runs:return self.window_runs[key]
        p=self.read_pattern(key);self.pattern_inputs[key]=hashlib.sha256(canonical(p).encode()).hexdigest();private=family in {"memory","control"}
        if not private:
            for field in ("rows","columns","contexts","rf_vectors_per_pe","spm_vectors_total","rom_words_per_pe","spm_port_period","writeback_period","dma_request_period","dma_response_period","compute_ii"):
                if p["hardware"][field]!=self.hardware[field]:raise ValueError("window array profile differs from model")
            if family=="vector" and p["hardware"]["sfu_ii"]!=self.hardware["sfu_ii"]:raise ValueError("window SFU II differs")
        else:
            field="control_controller" if family=="control" else "memory_controller"
            if p["hardware"][field]!=self.hardware[field]:raise ValueError("window private controller profile differs")
        codes={t["id"]:self.template(node,t) for t in p["templates"]};blocks=p["blocks"]+p.get("controllers",[]);runs=[];pes=self.hardware["rows"]*self.hardware["columns"]
        for b in blocks:
            if b["source_operator_id"]!=0 or b["admission_dependencies"]:raise ValueError("compiled pattern is not a single isolated window")
            event=self.block_pattern(b["events"],p);row=dict(count=1,event_pattern=event)
            if not private:row.update(template=codes[b["template"]],pe_base=b["pe"],pe_stride=0)
            if runs and all(runs[-1][k]==row[k] for k in (("event_pattern",) if private else ("event_pattern","template"))):
                last=runs[-1]
                if private:last["count"]+=1;continue
                if last["count"]==1:last["pe_stride"]=(b["pe"]-last["pe_base"])%pes
                if (last["pe_base"]+last["count"]*last["pe_stride"])%pes==b["pe"]:last["count"]+=1;continue
            runs.append(row)
        if not runs:raise ValueError("compiled window has no executable blocks")
        self.unique_window_blocks_scanned+=len(blocks);self.window_runs[key]=runs;return runs

    def link(self):
        if self.directory.exists():raise ValueError("choose a fresh linked block-pattern directory")
        rows={(r["binding"]["source_operator_id"],r["binding"]["batch_index"]):r for r in self.manifest["windows"]}
        if len(rows)!=len(self.manifest["windows"]) or len(rows)!=self.plan["total_windows"]:raise ValueError("compiled model window coverage differs")
        streams=[]
        for source in self.plan["sources"]:
            node=self.nodes[source["source_operator_id"]];windows=[]
            for batch in range(source["window_count"]):
                row=rows[(source["source_operator_id"],batch)];binding=row["binding"]
                if binding["parents"]!=source["parents"] or binding["family"]!=source["family"] or binding["batch_count"]!=source["window_count"]:raise ValueError("source/batch/dependency binding changed")
                runs=self.compile_runs(row["pattern"],node,source["family"]);windows.append(dict(runs=runs))
                self.logical_blocks+=sum(r["count"] for r in runs);self.dynamic_events+=sum(r["count"]*self.patterns[r["event_pattern"]]["dynamic_events"] for r in runs)
            streams.append(dict(source_operator_id=source["source_operator_id"],family=source["family"],parents=source["parents"],windows=windows,
                operator_kind=node["kind"],output_shape=node["output"]["shape"],output_dtype=node["output"]["dtype"]))
        pairs=[];plan=self.program.get("block_pipeline_plan")
        if self.options.get("tile_pipeline",False):
            if not plan or compile_block_pipelines(self.program,event_slots=plan["event_slots"])["block_pipeline_plan"]!=plan:raise ValueError("full streaming pair plan does not reproduce")
            pairs=[dict(producer_source=p["producer_source"],consumer_source=p["consumer_source"],mapping=p["mapping"],event_slots=plan["event_slots"]) for p in plan["pairs"]]
        self.directory.mkdir(parents=True)
        for key,raw in self.payloads.items():(self.directory/(key+".json")).write_bytes(raw)
        return dict(schema="mlx_event_schedule_v6",source_tick_order=True,hardware=self.hardware,physical_memory=self.memory,templates=list(self.templates.values()),blocks=[],controllers=[],
            source_streams=streams,streaming_pairs=pairs,event_patterns=self.patterns,pattern_cache_bytes=64*1024*1024,block_interval_limit=0,trace_limit=0,policy="source_priority",
            max_cycles=self.options.get("max_cycles",10000000000000))
