"""Compile a whole typed graph into generic RV64 dispatch tasks and wire data."""
import hashlib
import math
import struct
from pathlib import Path

from .address_plan import iter_bindings
from .lowering import BYTES,lower_control,tensor_descriptor
from ..physical_device.lowering import geometry,lower_matrix_window
from ..physical_device.vector_lowering import lower_vector
from ..physical_device.memory_lowering import lower_memory


def compile_graph(program,life,*,device_base=2**32,device_bytes=1048576,data_offset=65536,scratch_offset=4096,scratch_bytes=16384,block_pairs=False,event_slots=32):
    if device_base%4096 or not 0<=device_base<2**40 or device_bytes%4096 or not 8192<=device_bytes<=2**40-device_base:raise ValueError("graph device mapping invalid")
    if scratch_offset<4096 or scratch_offset%8 or scratch_bytes<15872 or scratch_offset+scratch_bytes>data_offset or data_offset%64 or data_offset>device_bytes:raise ValueError("graph scratch/data partition invalid")
    if type(block_pairs) is not bool:raise ValueError("block pair selection must be boolean")
    if block_pairs:
        from .pair_graph import compile_pair_graph
        return compile_pair_graph(program,life,device_base=device_base,device_bytes=device_bytes,data_offset=data_offset,scratch_offset=scratch_offset,scratch_bytes=scratch_bytes,event_slots=event_slots)
    origin=life["initial"]["base"];relocation=device_base+data_offset-origin
    peak_end=max([a["base"]+a["reserved_bytes"] for a in life["initial"]["allocations"]]+[e["allocation"]["base"]+e["allocation"]["reserved_bytes"] for e in life["events"]]+[origin])
    required=peak_end-origin+data_offset
    if required>device_bytes:raise ValueError(f"graph requires {required} mapped bytes, provided {device_bytes}")
    blob=bytearray();tasks=[];sources=[];asset_bindings={};output_bindings={};layouts_last=None
    counts={"matrix":0,"vector":0,"memory":0,"control":0,"view":0};relocated={}
    for ordinal,(node,layouts,bindings) in enumerate(iter_bindings(program,life)):
        layouts_last=layouts
        if not relocated:relocated={name:{**binding,"base":binding["base"]+relocation} for name,binding in bindings.items()}
        root=layouts[node["id"]]["root"]
        if root not in relocated:relocated[root]={**bindings[root],"base":bindings[root]["base"]+relocation}
        if not asset_bindings:asset_bindings={name:relocated[name] for name in program["assets"]}
        kinds=[name for name in ("matrix","vector","memory","control") if name+"_program" in node]
        if len(kinds)!=1:raise ValueError("graph source has missing/ambiguous executable route")
        family=kinds[0];view=family=="memory" and node["memory_program"]["mode"]=="view";commands=[]
        if view:
            lower_memory(node,layouts,relocated);commands=[None];counts["view"]+=1
        elif family=="matrix":commands=[lower_matrix_window(node,layouts,relocated,b)[0] for b in range(geometry(node,layouts)["batches"])]
        elif family=="vector":commands=[lower_vector(node,layouts,relocated)[0]]
        elif family=="memory":commands=[lower_memory(node,layouts,relocated)[0]]
        else:commands=[lower_control(node,layouts,relocated)[0]]
        counts[family]+=1
        for batch,command in enumerate(commands):
            offset=len(blob) if command is not None else 0
            if command is not None:
                if len(blob)%8:raise ValueError("graph command alignment changed")
                blob.extend(command)
            tasks.append({"kind":0 if view else 1 if family=="control" else 2,"source_ordinal":ordinal,"source_id":node["source_operator_id"],"batch_index":batch,"batch_count":len(commands),"command_offset":offset,"bytes":len(command) if command else 0,"family":family})
        sources.append({"source_ordinal":ordinal,"source_operator_id":node["source_operator_id"],"kind":node["kind"],"family":family,"view_elided":view,"task_count":len(commands),"forward_id":node["forward_id"],"layer_idx":node["layer_idx"]})
        output_bindings[node["id"]]=relocated[layouts[node["id"]]["root"]]
    if layouts_last is None:raise ValueError("empty graph is not registered")
    outputs=[]
    for spec in program["outputs"]:
        for role in ("logits","token"):
            value=spec[role];layout=layouts_last[value];binding=output_bindings.get(value,asset_bindings.get(value))
            if binding is None:raise ValueError("graph output binding missing")
            outputs.append({"forward_id":spec["forward_id"],"role":role,"value":value,"layout":layout,"binding":binding})
    return bytes(blob),{"classification":"compiled_rv64_graph_dispatch_plan_not_execution","device_base":device_base,"device_bytes":device_bytes,"required_mapped_bytes":required,"data_offset":data_offset,"scratch_offset":scratch_offset,"scratch_bytes":scratch_bytes,
        "sources":sources,"tasks":tasks,"assets":asset_bindings,"outputs":outputs,"family_source_calls":counts,"command_bytes":len(blob),"source_calls":len(sources),"task_count":len(tasks),"model_data_executed":False,"mlx_system_verified":False,"inference_performance_eligible":False}


def literal_bytes(spec):
    values=[]
    def visit(value):
        if isinstance(value,list):
            for child in value:visit(child)
        else:values.append(value.get("float_literal") if isinstance(value,dict) else value)
    visit(spec["values"])
    if len(values)!=math.prod(spec["shape"]):raise ValueError("graph literal element count mismatch")
    dtype=spec["dtype"];output=bytearray()
    for value in values:
        if dtype=="i64":
            if type(value) not in {int,bool} or not -2**63<=value<2**63:raise ValueError("graph integer literal is not exact int64")
            output.extend(struct.pack("<q",int(value)))
        else:
            value=float(value)
            try:single=struct.pack("<f",value)
            except OverflowError:single=struct.pack("<f",math.copysign(math.inf,value))
            value=struct.unpack("<f",single)[0]
            if dtype=="bool":output.append(int(bool(value)))
            elif dtype=="f32":output.extend(single)
            else:
                try:output.extend(struct.pack("<e",value))
                except OverflowError:output.extend(struct.pack("<e",math.copysign(math.inf,value)))
    return bytes(output)


def write_payload(program,path):
    entries=[];offset=0
    with Path(path).open("wb") as output:
        for name,spec in sorted(program["assets"].items()):
            padding=(-offset)%8;output.write(bytes(padding));offset+=padding;count=math.prod(spec["shape"])*BYTES[spec["dtype"]];digest=hashlib.sha256()
            if spec["kind"]=="literal":
                data=literal_bytes(spec);output.write(data);digest.update(data)
            elif spec["kind"]=="mapped_file":
                if spec["bytes"]!=count:raise ValueError("graph mapped asset byte count mismatch")
                with Path(spec["path"]).open("rb") as source:
                    source.seek(spec["byte_offset"]);remaining=count
                    while remaining:
                        data=source.read(min(remaining,8*1024*1024))
                        if not data:raise ValueError("graph asset file ended early")
                        output.write(data);digest.update(data);remaining-=len(data)
            else:raise ValueError("graph asset kind unsupported")
            entries.append({"value":name,"offset":offset,"bytes":count,"sha256":digest.hexdigest()});offset+=count
    return entries
