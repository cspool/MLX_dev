"""Encode two existing operator lowerings into the bounded pair wire ABI."""
import struct
from mlxsim.model_block_pipeline import resources

from .lowering import lower_matrix_window
from .vector_lowering import lower_vector

MAGIC=0x4D4C585041493031
BYTES=8832


def lower_pair(producer,consumer,layouts,bindings,*,event_slots=32):
    if type(event_slots) is not int or not 1<=event_slots<=32:
        raise ValueError("pair event budget must be 1..32")
    if producer["source_operator_id"]>=consumer["source_operator_id"]:
        raise ValueError("pair source priority must be forward")
    if "vector_program" not in consumer or consumer["kind"] in {"mean","softmax"}:
        raise ValueError("pair consumer must be pointwise vector")
    if producer["output"]["shape"]!=consumer["output"]["shape"]:
        raise ValueError("pair pending input requires the same shape")
    if "matrix_program" in producer:
        first,first_route=lower_matrix_window(producer,layouts,bindings,0);kind=1
    elif "vector_program" in producer:
        first,first_route=lower_vector(producer,layouts,bindings);kind=2
    else:
        raise ValueError("pair producer lacks an array lowering")
    mask=0
    for index,arg in enumerate(consumer["args"][:2]):
        if isinstance(arg,dict) and arg.get("value")==producer["id"]:mask|=1<<index
    if not mask:raise ValueError("pair consumer has no direct producer operand")
    second,second_route=lower_vector(consumer,layouts,bindings)
    p,c=resources(producer),resources(consumer)
    if p["rf"]+c["rf"]>16 or p["spm"]+c["spm"]>128 or p["rom"]+c["rom"]>32:
        raise ValueError("pair exceeds joint RF/SPM/ROM capacity")
    words=[MAGIC,1,kind,len(first),producer["source_operator_id"],consumer["source_operator_id"],event_slots,mask,0]+[0]*23
    blob=struct.pack("<32Q",*words)+first+bytes(4288-len(first))+second
    if len(blob)!=BYTES:raise ValueError("pair wire size changed")
    return blob,{"profile":"mlx-pair-wire-v1","producer_source":producer["source_operator_id"],"consumer_source":consumer["source_operator_id"],
                 "event_slots":event_slots,"input_mask":mask,"producer":first_route,"consumer":second_route,
                 "command_bytes":len(blob),"model_execution_verified":False,"mlx_system_verified":False}
