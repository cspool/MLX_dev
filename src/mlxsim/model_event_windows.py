"""Route every full-program source/batch to an existing event window compiler.

Layouts and logical batch offsets are retained separately from reusable timing
patterns. This does not load a full graph into the C++ event scheduler yet.
"""
import copy
import math

from .model_event_graph_plan import graph_plan
from .model_memory_program import Planner,BYTES
from .model_value_outputs import value_outputs
from .model_matrix_events import matrix_events
from .model_vector_events import vector_events
from .model_memory_events import memory_events
from .model_control_events import control_events


class MissingWitness(ValueError):pass


def broadcast_batch(index,output_shape,input_shape):
    if type(index) is not int or not 0<=index<math.prod(output_shape) or len(input_shape)>len(output_shape):raise ValueError("batch index/rank outside output")
    position=0;stride=1
    for axis in range(len(output_shape)-1,-1,-1):
        coordinate=index%output_shape[axis];index//=output_shape[axis]
        own=axis+len(input_shape)-len(output_shape)
        if own>=0:
            extent=input_shape[own]
            if extent not in (1,output_shape[axis]):raise ValueError("input batch shape cannot broadcast")
            if extent!=1:position+=coordinate*stride
            stride*=extent
    return position


def element_address(layout,index):
    if type(index) is not int or not 0<=index<math.prod(layout["shape"]):raise ValueError("logical element outside layout")
    offset=layout["offset"]
    for extent,stride in zip(reversed(layout["shape"]),reversed(layout["strides"]),strict=True):
        if stride<0:raise ValueError("negative tensor stride")
        offset+=(index%extent)*stride;index//=extent
    if not 0<=offset<layout["storage_elements"]:raise ValueError("logical element outside root storage")
    return dict(root=layout["root"],byte_offset=offset*BYTES[layout["dtype"]],bytes=BYTES[layout["dtype"]])


class WindowCatalog:
    def __init__(self,program):
        self.program=program;self.plan=graph_plan(program);self.sources={};planner=Planner()
        for name,asset in program["assets"].items():planner.add_asset(name,asset)
        for node,row in zip(program["nodes"],self.plan["sources"],strict=True):
            inputs={name:copy.deepcopy(planner.layouts[name]) for name in row["inputs"]}
            candidate=copy.deepcopy(node);planner.register(candidate,planned=True,same_device=node.get("memory_program",{}).get("same_reference_device",True))
            if "memory_program" in node and candidate["memory_program"]!=node["memory_program"]:raise ValueError("full source layout differs from canonical memory plan")
            outputs={name:copy.deepcopy(planner.layouts[name]) for name,_ in value_outputs(node)}
            self.sources[row["source_operator_id"]]=dict(node=node,plan=row,inputs=inputs,outputs=outputs)

    def window(self,source,batch=0,witness=None):
        if type(source) is not int or source not in self.sources:raise ValueError("missing source identity")
        entry=self.sources[source];row=entry["plan"];node=copy.deepcopy(entry["node"]);family=row["family"]
        if type(batch) is not int or not 0<=batch<row["window_count"]:raise ValueError("batch outside complete source")
        requirement=row.get("required_timing_witness")
        if requirement and requirement["count"] and witness is None:raise MissingWitness(f"source {source} requires {requirement['kind']}")
        if not requirement and witness is not None:raise ValueError("unexpected timing witness")
        options=copy.deepcopy(self.program.get(family+"_schedule_options",{}));inputs=copy.deepcopy(entry["inputs"])
        binding=dict(source_operator_id=source,origin_source_operator_id=row["origin_source_operator_id"],family=family,batch_index=batch,batch_count=row["window_count"],
                     parents=row["parents"],inputs=inputs,outputs=copy.deepcopy(entry["outputs"]),release=copy.deepcopy(node.get("release",[])),
                     memory_timing="frontend_fixed_latency_not_shared_physical_memory",addresses_executed=False)
        if family=="matrix":
            geometry=row["matrix"];m,n,k=(geometry[key] for key in ("m","n","k"));linear=geometry["transposed_b"]
            a,b=(inputs[node["args"][i]["value"]] for i in (0,1));ab=0 if linear else broadcast_batch(batch,geometry["output_batch_shape"],geometry["a_batch_shape"])
            bb=0 if linear else broadcast_batch(batch,geometry["output_batch_shape"],geometry["b_batch_shape"])
            binding["matrix"]=dict(**geometry,a_batch_index=ab,b_batch_index=bb,output_batch_index=batch,a_flat_start=ab*m*k,b_flat_start=bb*k*n,output_flat_start=batch*m*n)
            job=dict(schema="mlx_matrix_window_job_v1",m=m,n=n,k=k,transposed_b=linear,program=node["matrix_program"],options=options,
                     a=dict(kind="logical_window_slice",dtype=a["dtype"],shape=[m,k]),b=dict(kind="logical_window_slice",dtype=b["dtype"],shape=[n,k] if linear else [k,n]))
            if node["matrix_program"]["has_bias"]:
                bias=inputs[node["args"][2]["value"]];job["bias"]=dict(kind="logical_window_slice",dtype=bias["dtype"],shape=[n])
        elif family=="vector":
            # Numerical identities/layouts stay in binding; the event pattern
            # depends on geometry, scalar operands and canonical code only.
            names={};assets={}
            def rename(value):
                if isinstance(value,dict) and "value" in value:
                    old=value["value"]
                    if old not in names:
                        new=f"input{len(names)}";names[old]=new;assets[new]=dict(kind="layout_only",dtype=inputs[old]["dtype"],shape=inputs[old]["shape"])
                    return dict(value=names[old])
                if isinstance(value,list):return [rename(x) for x in value]
                return value
            args=rename(node["args"])
            pattern_node={key:node[key] for key in ("kind","kwargs","output","vector_program")};pattern_node["args"]=args
            job=dict(schema="mlx_vector_window_job_v1",node=pattern_node,assets=assets,options=options);binding["pattern_inputs"]={new:old for old,new in names.items()}
        elif family=="memory":
            job=dict(schema="mlx_memory_event_window_job_v1",node=node,options=options)
            if requirement:job["predicate_choices"]=witness
        else:
            job=dict(schema="mlx_control_event_window_job_v1",node=node,input_layouts=inputs,options=options)
            if requirement:job["guard_value" if node["kind"]=="guard" else "branch_taken"]=witness
        return job,binding

    @staticmethod
    def compile(job):
        routes={"mlx_matrix_window_job_v1":matrix_events,"mlx_vector_window_job_v1":vector_events,"mlx_memory_event_window_job_v1":memory_events,"mlx_control_event_window_job_v1":control_events}
        return routes[job["schema"]](job)
