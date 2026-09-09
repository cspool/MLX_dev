import copy
import pytest

from scripts.verify_mlx_replayed_model_windows import check_argmax_binding
from scripts.mlx_system_attempt import digest


@pytest.mark.parametrize('damage',[None,'node','operand','offset','file','order','missing','duplicate'])
def test_argmax_job_and_order_are_bound_to_target(tmp_path,damage):
    path=tmp_path/'logits.bin';path.write_bytes(b'\x00'*6)
    node=dict(kind='argmax',args=[dict(value='logits'),-1])
    job=dict(node=copy.deepcopy(node),assets=dict(logits=dict(kind='mapped_file',path=str(path),byte_offset=0,shape=[1,3])))
    report=dict(trace=[dict(event='instruction_issue',phase='select',pc=0,row=0,column=c) for c in (1,2)])
    sha=digest(path)
    if damage=='node':job['node']['args'][1]=0
    elif damage=='operand':job['assets']['extra']={}
    elif damage=='offset':job['assets']['logits']['byte_offset']=2
    elif damage=='file':path.write_bytes(b'\x01'*6)
    elif damage=='order':report['trace'].reverse()
    elif damage=='missing':report['trace'].pop()
    elif damage=='duplicate':report['trace'].append(copy.deepcopy(report['trace'][0]))
    if damage:
        with pytest.raises(RuntimeError):check_argmax_binding(node,job,report,sha)
    else:check_argmax_binding(node,job,report,sha)
