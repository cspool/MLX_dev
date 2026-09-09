import copy
import hashlib

import numpy as np
import pytest

from scripts.verify_mlx_bert_system_model import compare_framework
from scripts.verify_mlx_system_model import Evidence


def records(tmp_path):
    span=dict(start=0,end=0,text="answer")
    expected={};actual={}
    for role in ("start_logits","end_logits"):
        ref=tmp_path/f"ref-{role}.bin";ref.write_bytes(np.array([1,2,3],dtype='<f4').tobytes())
        got=tmp_path/f"cpu-{role}.bin";got.write_bytes(ref.read_bytes())
        expected[role]=dict(file=str(ref),sha256=hashlib.sha256(ref.read_bytes()).hexdigest())
        actual[role]=dict(file=str(got),sha256=hashlib.sha256(got.read_bytes()).hexdigest())
    return {"reference_checks":[dict(forward_id=0,context_mask=[True]*3,span=span,outputs=expected)]}, \
           {"outputs":[dict(forward_id=0,span=copy.deepcopy(span),outputs=actual)]}


@pytest.mark.parametrize("damage",[None,"small_rounding","numeric","answer","nonfinite","extent","reference"])
def test_framework_comparison_accepts_tolerance_but_not_false_system_results(tmp_path,damage):
    from pathlib import Path
    inventory,actual=records(tmp_path)
    own=actual["outputs"][0]["outputs"]["start_logits"];path=Path(own["file"])
    if damage in {"small_rounding","numeric","nonfinite","extent"}:
        data=np.array([1,2,3],dtype='<f4')
        if damage=="small_rounding":data[0]+=1e-6
        elif damage=="numeric":data[0]=200
        elif damage=="nonfinite":data[0]=np.nan
        else:data=data[:2]
        path.write_bytes(data.tobytes());own["sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
    if damage=="answer":actual["outputs"][0]["span"]["text"]="wrong"
    if damage=="reference":Path(inventory["reference_checks"][0]["outputs"]["start_logits"]["file"]).write_bytes(bytes(12))
    if damage in {"nonfinite","extent","reference"}:
        with pytest.raises(RuntimeError):compare_framework(inventory,actual,Evidence())
    else:
        rows=compare_framework(inventory,actual,Evidence())
        assert rows[0]["major_correctness_passed"]==(damage in {None,"small_rounding"})
