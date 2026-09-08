"""Single-file and explicit bijective parameter names, without dtype conversion."""
import copy
import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from mlxsim.model_execution_inventory import ModelExecutionInventory
from mlxsim.model_tensor_compiler import compile_inventory
from scripts.run_mlx_tensor_semantics import sha
from test_physical_model import binary as physical_binary, run as run_physical


class Parameters(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm=torch.nn.Module()
        self.norm.weight=torch.nn.Parameter(torch.tensor([2.,3.,4.]))
        self.norm.bias=torch.nn.Parameter(torch.tensor([-1.,.5,7.]))
    def forward(self,x):
        return x*self.norm.weight+self.norm.bias


def inventory(tmp_path,renamed):
    model=Parameters().eval();names={"norm.weight":"norm.gamma","norm.bias":"norm.beta"} if renamed else None
    file=tmp_path/"model.safetensors"
    save_file({names.get(k,k) if names else k:v.detach() for k,v in model.state_dict().items()},str(file))
    x=torch.tensor([[1.,2.,3.]]);trace=ModelExecutionInventory(model,capture_bindings=True);trace.bind_input("x",x)
    with trace.attach(),torch.inference_mode(),trace.step(0,"prefill"):
        logits=model(x);trace.phase="token_selection";token=logits.argmax(-1)
    report=trace.report();report.update(model_identity={"family":"checkpoint-binding-component","variant":"not-full-model","parameters":6,"path":str(tmp_path)},
        files={str(file):{"sha256":sha(file),"bytes":file.stat().st_size}},reference_checks=[{"forward_id":0}],input={"x":x.tolist()})
    if names:report["model_identity"]["checkpoint_name_by_model_parameter"]=names
    return report,logits.numpy(),token.tolist()


def compile(report):
    return compile_inventory(report,matrix_backend="scheduled",vector_backend="scheduled",memory_backend="scheduled",control_backend="scheduled")[0]


@pytest.mark.parametrize("renamed",[False,True])
def test_single_checkpoint_exact_f32_bytes_reach_cpp(physical_binary,tmp_path,renamed):
    report,expected,tokens=inventory(tmp_path,renamed);program=compile(report)
    mapped=[a for a in program["assets"].values() if a["kind"]=="mapped_file"]
    assert len(mapped)==2 and all(a["dtype"]=="f32" and a["bytes"]==12 for a in mapped)
    assert {a["parameter_name"] for a in mapped}==({"norm.gamma","norm.beta"} if renamed else {"norm.weight","norm.bias"})
    if renamed:assert {a["model_parameter_name"] for a in mapped}=={"norm.weight","norm.bias"}
    actual=run_physical(physical_binary,tmp_path/"native",program)
    np.testing.assert_array_equal(np.fromfile(actual["outputs"][0]["logits_file"],dtype=np.float32).reshape(expected.shape),expected)
    assert actual["outputs"][0]["tokens"]==tokens


@pytest.mark.parametrize("damage",["collision","missing","unknown","fingerprint","shape","dtype"])
def test_single_checkpoint_rejects_incomplete_or_reinterpreted_bindings(tmp_path,damage):
    report,_,_=inventory(tmp_path,True)
    if damage=="collision":report["model_identity"]["checkpoint_name_by_model_parameter"]["norm.bias"]="norm.gamma"
    if damage=="missing":del report["model_identity"]["checkpoint_name_by_model_parameter"]["norm.bias"]
    if damage=="unknown":report["model_identity"]["checkpoint_name_by_model_parameter"]["norm.bias"]="foreign"
    if damage=="fingerprint":report["files"]={}
    if damage in {"shape","dtype"}:
        meta=next(t for t in report["tensors"].values() if t["parameter_or_buffer_names"]==["norm.weight"])
        meta[damage]=[4] if damage=="shape" else "torch.float16"
    with pytest.raises(ValueError):compile(report)


def test_checkpoint_shard_cannot_escape_model_directory(tmp_path):
    report,_,_=inventory(tmp_path,False)
    (tmp_path/"model.safetensors.index.json").write_text(json.dumps({"weight_map":{"norm.weight":"../outside.safetensors"}}))
    with pytest.raises(ValueError,match="outside the model directory"):compile(report)
