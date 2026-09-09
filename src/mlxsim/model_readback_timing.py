"""Compile actual output identities/shapes to the C++ post-drain readback model."""
import math

from .model_value_outputs import value_outputs
from .model_memory_program import BYTES
from .model_result_contract import result_roles, validate_result_contract


def readback_job(program, options, graph_cycles):
    if type(graph_cycles) is not int or graph_cycles < 0:
        raise ValueError('nonnegative graph cycle required')
    validate_result_contract(program)
    values = dict(program['assets'])
    for node in program['nodes']:
        for name, spec in value_outputs(node):values[name] = spec
    reads = []
    for spec in program['outputs']:
        for role in result_roles(program):
            name = spec[role];value = values[name]
            reads.append(dict(forward_id=spec['forward_id'],role=role,value=name,
                              elements=math.prod(value['shape']),element_bytes=BYTES[value['dtype']]))
    return dict(schema='mlx_post_graph_readback_v1',graph_cycles=graph_cycles,reads=reads,
                memory={k:options.get('memory',{}).get(k,v) for k,v in dict(latency=8,accept_period=1,nack_every=0).items()})
