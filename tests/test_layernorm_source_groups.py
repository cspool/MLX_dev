"""Full-shape LayerNorm recipes execute through real C++ primitives and memory."""
import copy
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from mlxsim.model_execution_inventory import ModelExecutionInventory
from mlxsim.model_tensor_compiler import compile_inventory
from mlxsim.model_composites import layer_norm_contract
from mlxsim.model_numeric_reference import pairwise_last
from mlxsim.model_source_groups import validate_source_groups,verify_source_groups
from mlxsim.model_physical_evidence import verify_physical_execution
from mlxsim.model_block_pipeline import compile_block_pipelines
from mlxsim.model_ready_evidence import verify_ready_execution
from scripts.run_mlx_tensor_semantics import sha
from test_model_tensor_semantics import native
from test_ready_graph import binary as graph_binary
from test_physical_model import binary as physical_binary


def shifted_reference(x,weight,bias,eps):
    dtype=x.numpy().dtype;values=x.numpy().astype(np.float32)
    shifted=np.subtract(values,values[...,:1],dtype=np.float32)
    mean=np.divide(pairwise_last(shifted),np.float32(values.shape[-1]),dtype=np.float32)
    center=np.subtract(shifted,mean,dtype=np.float32)
    variance=np.divide(pairwise_last(np.multiply(center,center,dtype=np.float32)),np.float32(values.shape[-1]),dtype=np.float32)
    inverse=np.divide(np.float32(1),np.sqrt(np.add(variance,np.float32(eps),dtype=np.float32)),dtype=np.float32)
    value=np.multiply(center,inverse,dtype=np.float32)
    if weight is not None:value=np.multiply(value,weight.detach().numpy().astype(np.float32),dtype=np.float32)
    if bias is not None:value=np.add(value,bias.detach().numpy().astype(np.float32),dtype=np.float32)
    return value.astype(dtype)


def capture(tmp_path,shape=(2,17),dtype=torch.float32,affine=True,large=False,split=False):
    torch.manual_seed(801)
    x=torch.randn(shape,dtype=dtype)
    if large:x=(torch.arange(int(np.prod(shape))).reshape(shape)%7).float()*.125+1e6
    model=torch.nn.LayerNorm(shape[-1],eps=1e-12,elementwise_affine=affine,dtype=dtype).eval()
    if affine:
        with torch.no_grad():model.weight.copy_(torch.linspace(.7,1.3,shape[-1],dtype=dtype));model.bias.copy_(torch.linspace(-.2,.3,shape[-1],dtype=dtype))
    file=tmp_path/"model.safetensors";save_file({k:v.detach() for k,v in model.state_dict().items()},str(file))
    trace=ModelExecutionInventory(model,capture_bindings=True);trace.bind_input("x",x)
    with trace.attach(),torch.inference_mode(),trace.step(0,"prefill"):
        logits=model(x)
        if split:
            left,right=logits.split(shape[-1]//2,-1);logits=left+right
        trace.phase="token_selection";token=logits.argmax(-1)
    report=trace.report();report.update(model_identity={"family":"LayerNorm-component","variant":"not-full-model","parameters":sum(p.numel() for p in model.parameters()),"path":str(tmp_path),"files":{str(file):{"sha256":sha(file),"bytes":file.stat().st_size}}},reference_checks=[{"forward_id":0}],input={"shape":shape})
    program,coverage=compile_inventory(report,matrix_backend="scheduled",vector_backend="scheduled",memory_backend="scheduled",control_backend="scheduled")
    expected=shifted_reference(x,model.weight,model.bias,model.eps)
    if split:expected=np.add(expected[...,:shape[-1]//2],expected[...,shape[-1]//2:],dtype=expected.dtype)
    return program,coverage,report,expected,logits.numpy(),x


def run(binary,path,program,*,kind="physical",error=None):
    path.mkdir();p=path/"program.json";p.write_text(json.dumps(program));options={"base":2**32,"bytes":16*1024*1024,"max_cycles":100000000}
    if kind=="paired":options.update(tile_pipeline=True,template_load_timing=True)
    o=path/"options.json";o.write_text(json.dumps(options))
    command=[str(binary),str(p),str(path/"out"),"none","1"] if kind=="tensor" else [str(binary),str(p),str(o),str(path/"out")]
    process=subprocess.run(command,capture_output=True,text=True,timeout=180)
    (path/"run.log").write_text(process.stdout+process.stderr)
    if error:
        assert process.returncode!=0 and error in process.stderr,process.stdout+process.stderr
        assert not (path/"out/result.json").exists();return
    assert process.returncode==0,process.stdout+process.stderr
    result=json.loads((path/"out/result.json").read_text())
    verify_source_groups(program,result)
    if kind=="physical":verify_physical_execution(program,result,options)
    if kind=="paired":verify_ready_execution(program,result,options)
    dtype=np.float16 if result["outputs"][0]["dtype"]=="f16" else np.float32
    actual=np.fromfile(result["outputs"][0]["logits_file"],dtype=dtype).reshape(result["outputs"][0]["shape"])
    return result,actual,options


@pytest.mark.parametrize("shape,dtype,affine",[((2,17),torch.float32,True),((2,33),torch.float16,True),((3,768),torch.float32,False),((1,28,768),torch.float32,True)])
def test_layernorm_full_shapes_run_as_grouped_cpp_stages(physical_binary,tmp_path,shape,dtype,affine):
    program,coverage,inventory,expected,framework,_=capture(tmp_path,shape,dtype,affine)
    assert program["schema"]=="mlx_tensor_semantics_v3" and len(inventory["operations"])==coverage["source_calls"]==2
    assert coverage["lowered_calls"]>coverage["source_calls"] and len(program["nodes"])==coverage["lowered_calls"]
    actual,values,_=run(physical_binary,tmp_path/"physical",program)
    np.testing.assert_array_equal(values.view(np.uint16 if dtype==torch.float16 else np.uint32),expected.view(np.uint16 if dtype==torch.float16 else np.uint32))
    np.testing.assert_allclose(values,framework,atol=1e-3 if dtype==torch.float16 else 3e-6,rtol=2e-3 if dtype==torch.float16 else 3e-5)
    assert actual["executed_source_calls"]==2 and actual["executed_lowered_calls"]==len(program["nodes"])
    assert actual["functional_entry_calls"]==actual["blas_calls"]==0 and actual["arena_drained"]["reserved_bytes"]==0
    assert all(n["vector_program"]["rom_words"]==32 and len(n["vector_program"]["rom"])<=32 for n in program["nodes"] if "vector_program" in n)


def test_shifted_statistics_handle_large_offsets_without_widening_hardware(physical_binary,tmp_path):
    program,_,_,expected,framework,x=capture(tmp_path,(2,35),large=True)
    _,values,_=run(physical_binary,tmp_path/"shifted",program)
    np.testing.assert_array_equal(values.view(np.uint32),expected.view(np.uint32))
    a=x.numpy().astype(np.float64);center=a-a.mean(-1,keepdims=True);ref=center/np.sqrt((center*center).mean(-1,keepdims=True)+1e-12)
    ref=ref*np.linspace(.7,1.3,35,dtype=np.float32)+np.linspace(-.2,.3,35,dtype=np.float32)
    np.testing.assert_allclose(values,ref,rtol=3e-5,atol=3e-6)
    (tmp_path/"large-offset-comparison.json").write_text(json.dumps({"target_vs_framework_max_abs":float(np.max(np.abs(values-framework))),"target_vs_float64_max_abs":float(np.max(np.abs(values-ref))),"framework_gate_overridden":False})+"\n")


def test_grouped_recipe_in_plain_and_shared_paired_engines(native,graph_binary,tmp_path):
    program,_,_,expected,_,_=capture(tmp_path)
    for binary,kind in ((native[0],"tensor"),(graph_binary,"paired")):
        p=compile_block_pipelines(program,event_slots=4) if kind=="paired" else program
        result,actual,options=run(binary,tmp_path/kind,p,kind=kind)
        np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
        bad=copy.deepcopy(result);bad["executed_source_calls"]=bad["executed_lowered_calls"]
        with pytest.raises(RuntimeError,match="accounting mismatch"):verify_source_groups(p,bad)
        bad=copy.deepcopy(result);bad["source_group_events"][0]["lowered_ids"].pop()
        with pytest.raises(RuntimeError,match="completion differs"):verify_source_groups(p,bad)


def test_composite_source_groups_keep_all_split_results(graph_binary,tmp_path):
    program,coverage,inventory,expected,_,_=capture(tmp_path,(1,8),split=True)
    assert coverage["source_calls"]==len(inventory["operations"])==4
    program=compile_block_pipelines(program,event_slots=4)
    result,actual,_=run(graph_binary,tmp_path/"tuple",program,kind="paired")
    np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
    split=next(n for n in program["nodes"] if n["kind"]=="split")
    event=next(e for e in result["events"] if e["source_operator_id"]==split["source_operator_id"])
    assert event["produced_values"]==[o["id"] for o in split["split_outputs"]]


@pytest.mark.parametrize("damage",["stage","profile","origin","kind","outputs","schema","source_dtype","source_shape","epsilon","affine_dtype","direct_label"])
def test_source_group_cannot_hide_missing_or_relabelled_stages(graph_binary,tmp_path,damage):
    program,_,_,_,_,_=capture(tmp_path)
    group=program["source_groups"][0]
    if damage=="stage":group["lowered_ids"].pop()
    if damage=="profile":group["lowering_profile"]="direct"
    if damage=="origin":program["nodes"][0]["origin_source_operator_id"]=8
    if damage=="kind":program["nodes"][3]["kind"]="mul";group["lowered_kinds"][3]="mul"
    if damage=="outputs":group["output_values"]=[program["nodes"][0]["id"]]
    if damage=="schema":program["schema"]="mlx_tensor_semantics_v1"
    if damage=="source_dtype":group["layer_norm_config"]["input_dtype"]="torch.float64"
    if damage=="source_shape":group["layer_norm_config"]["shape"]=[1,34]
    if damage=="epsilon":group["layer_norm_config"]["epsilon"]=.01
    if damage=="affine_dtype":group["layer_norm_config"]["weight_dtype"]="torch.float64"
    if damage=="direct_label":program["source_groups"][-1]["source_operator"]="aten.mul.Tensor"
    with pytest.raises(ValueError):validate_source_groups(program)
    run(graph_binary,tmp_path/"invalid",program,kind="paired",error="source" if damage in {"stage","origin","outputs","source_dtype","source_shape","epsilon","affine_dtype","direct_label"} else "recipe" if damage in {"kind","profile"} else "contract")


@pytest.mark.parametrize("damage",["axis","epsilon","dtype","affine","mutable"])
def test_layernorm_rejects_unregistered_contracts(tmp_path,damage):
    _,_,inventory,_,_,_=capture(tmp_path)
    event=copy.deepcopy(inventory["operations"][0])
    if damage=="axis":event["inputs"][1]=[2,17]
    if damage=="epsilon":event["inputs"][4]=-1
    if damage=="dtype":event["inputs"][0]["dtype"]="torch.bfloat16"
    if damage=="affine":event["inputs"][2]["shape"]=[16]
    if damage=="mutable":event["mutable"]=True
    with pytest.raises(ValueError):layer_norm_contract(event)


def test_composite_expansion_does_not_promote_an_unregistered_inventory(tmp_path):
    _,_,inventory,_,_,_=capture(tmp_path)
    inventory["classification"]="not_a_real_reference_inventory"
    with pytest.raises(ValueError,match="real model execution inventory"):
        compile_inventory(inventory,matrix_backend="scheduled",vector_backend="scheduled",memory_backend="scheduled",control_backend="scheduled")
