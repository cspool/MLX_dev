import copy
import pytest
import torch

from scripts.replay_mlx_predicate_closure import closure
from mlxsim.model_control_program import control_program
from test_model_tensor_semantics import node,ref
from test_physical_model import compiled_nodes


def program():
    a=torch.arange(4);m=a<=2
    nodes=[node(0,'arange',[4],a),node(1,'le',[ref('v0'),2],m),
           node(2,'where',[ref('v1'),1.,0.],m.float())]
    nodes[0]['control_program']=control_program('arange')
    nodes[1]['control_program']=control_program('le')
    return compiled_nodes({},nodes,'v2','v0')


def test_preserves_all_computed_dependencies_and_changes_no_semantics():
    p=program();old=copy.deepcopy(p);subset,requests=closure(p)
    assert p==old and [n['source_operator_id'] for n in subset['nodes']]==[0,1]
    assert subset['assets']=={} and subset['outputs']==[]
    for n,original in zip(subset['nodes'],p['nodes']):
        assert {k:v for k,v in n.items() if k!='release'}=={k:v for k,v in original.items() if k!='release'}
    assert requests==[p['nodes'][2]]


@pytest.mark.parametrize('damage',['external','cycle','operation','route','profile'])
def test_refuses_unproven_predicate_closure(damage):
    p=program()
    if damage=='external':p['nodes'][1]['args'][0]=ref('asset')
    elif damage=='cycle':p['nodes'][0]['args']=[ref('v1')]
    elif damage=='operation':p['nodes'][0]['kind']='argmax'
    elif damage=='route':p['nodes'][0].pop('control_program')
    else:p['schema']='mlx_tensor_semantics_v4'
    with pytest.raises(RuntimeError):closure(p)
