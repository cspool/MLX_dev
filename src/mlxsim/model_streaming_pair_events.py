"""Isolate one compiler-selected closed pair for event/native alignment checks.

External inputs must already be assets. This is not a full model graph linker.
"""
import copy

from .model_block_pipeline import compile_block_pipelines
from .model_event_windows import WindowCatalog
from .model_array_graph_events import array_graph_events
from .model_lazy_event_patterns import externalize
from .model_compact_event_streams import compact_streams


def streaming_pair_events(program,directory,pair_index=0,timed_templates=False,memory=None):
    plan=program["block_pipeline_plan"]
    if compile_block_pipelines(program,event_slots=plan["event_slots"])["block_pipeline_plan"]!=plan:raise ValueError("streaming pair plan does not reproduce")
    pair=plan["pairs"][pair_index];catalog=WindowCatalog(program);producer,consumer=pair["producer_source"],pair["consumer_source"]
    a,b=catalog.sources[producer],catalog.sources[consumer]
    if a["plan"]["parents"] or b["plan"]["parents"]!=[producer]:raise ValueError("isolated pair requires all external inputs to be assets")
    pj,_=catalog.window(producer);cj,_=catalog.window(consumer)
    # Canonical matrix batch windows have identical timing IR but different
    # layout/address bindings; this adapter preserves their complete count.
    for batch in range(a["plan"]["window_count"]):
        if catalog.window(producer,batch)[0]!=pj:raise ValueError("producer batch changes timing pattern")
    p=array_graph_events(dict(schema="mlx_array_event_graph_v1",windows=[pj,cj],sources=[
        dict(source_operator_id=producer,parents=[],windows=[0]),dict(source_operator_id=consumer,parents=[producer],windows=[1])]))
    p["hardware"]["template_load_timing"]=timed_templates
    c=compact_streams(externalize(p,directory));sources=c["source_streams"]
    sources[0]["windows"]=[copy.deepcopy(sources[0]["windows"][0]) for _ in range(a["plan"]["window_count"])]
    for source,entry in zip(sources,(a,b)):
        source["operator_kind"]=entry["node"]["kind"];source["output_shape"]=entry["node"]["output"]["shape"];source["output_dtype"]=entry["node"]["output"]["dtype"]
    c["streaming_pairs"]=[dict(producer_source=producer,consumer_source=consumer,mapping=pair["mapping"],event_slots=plan["event_slots"])]
    c["physical_memory"]=copy.deepcopy(memory or dict(latency=8,accept_period=1,nack_every=0))
    return c
