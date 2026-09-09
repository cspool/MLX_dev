"""GELU erf-form primitive execution and numerical limits, not model acceptance."""
import copy
import json
import math
import subprocess

import numpy as np
import pytest
import torch

from mlxsim.model_execution_inventory import ModelExecutionInventory
from mlxsim.model_tensor_compiler import compile_inventory
from mlxsim.model_gelu_program import gelu_contract
from mlxsim.model_numeric_reference import NumericReferenceMode
from mlxsim.model_source_groups import validate_source_groups,verify_source_groups
from mlxsim.model_physical_evidence import verify_physical_execution
from mlxsim.model_block_pipeline import compile_block_pipelines
from mlxsim.model_ready_evidence import verify_ready_execution
from test_model_tensor_semantics import native
from test_ready_graph import binary as graph_binary
from test_physical_model import binary as physical_binary

# Frozen component limits, not a replacement for any model/GPU tolerance.
F32_ABS_LIMIT = 3e-6
F16_ABS_LIMIT = 2e-3


def recipe_reference(x):
    """Independent vector control; the atomic expf is explicitly shared libm."""
    a=x.numpy().astype(np.float32);maximum=lambda x,y:np.where(x<y,y,x).astype(np.float32)
    clipped=maximum(-maximum(-a,np.float32(-8)),np.float32(-8))
    scaled=np.multiply(clipped,np.float32(0.7071067811865476),dtype=np.float32)
    absolute=maximum(scaled,-scaled)
    sign=np.divide(scaled,maximum(absolute,np.float32(2.0**-126)),dtype=np.float32)
    t=np.divide(np.float32(1),np.add(np.multiply(absolute,np.float32(.3275911),dtype=np.float32),np.float32(1),dtype=np.float32),dtype=np.float32)
    poly=np.multiply(np.float32(1.061405429),t,dtype=np.float32)
    poly=np.add(poly,np.float32(-1.453152027),dtype=np.float32)
    for coefficient in (1.421413741,-.284496736,.254829592):
        poly=np.add(np.multiply(poly,t,dtype=np.float32),np.float32(coefficient),dtype=np.float32)
    poly=np.multiply(poly,t,dtype=np.float32)
    atomic=NumericReferenceMode()
    decay=atomic.unary("expf",-np.multiply(scaled,scaled,dtype=np.float32))
    positive=np.add(-np.multiply(poly,decay,dtype=np.float32),np.float32(1),dtype=np.float32)
    factor=np.add(np.multiply(positive,sign,dtype=np.float32),np.float32(1),dtype=np.float32)
    return np.multiply(np.multiply(a,np.float32(.5),dtype=np.float32),factor,dtype=np.float32).astype(x.numpy().dtype)


def mathematical_reference(x):
    return np.fromiter((.5*float(v)*math.erfc(-float(v)/math.sqrt(2)) for v in x.numpy().flat),dtype=np.float64,count=x.numel()).reshape(x.shape)


def capture(tmp_path,x,scheduled=False):
    (tmp_path/"model.safetensors.index.json").write_text(json.dumps({"weight_map":{}}))
    trace=ModelExecutionInventory(torch.nn.Identity(),capture_bindings=True)
    if x.numel()<=65536:trace.bind_input("x",x)
    else:
        # Explicit large initial input for this operator test. This is not an
        # observation or an intermediate binding fed back into model execution.
        identifier=trace.tensor(x)["tensor_id"]
        trace.bindings[identifier]={"kind":"input","name":"x","values":x.tolist()}
    with torch.inference_mode(),trace.step(0,"prefill"):
        actual=torch.nn.functional.gelu(x,approximate="none")
        trace.phase="token_selection";actual.argmax(-1)
    inventory=trace.report();inventory.update(model_identity={"family":"GELU-component","variant":"not-full-model","parameters":0,"path":str(tmp_path),"files":{}},reference_checks=[{"forward_id":0}],input={"shape":list(x.shape)})
    program,coverage=compile_inventory(inventory,matrix_backend="scheduled" if scheduled else "microcode",vector_backend="scheduled" if scheduled else "microcode",memory_backend="scheduled" if scheduled else "planned",control_backend="scheduled" if scheduled else "rv64_leaf")
    return program,coverage,inventory,actual.numpy()


def run(binary,path,program,mode="tensor",error=None):
    path.mkdir();p=path/"program.json";p.write_text(json.dumps(program))
    options={"base":2**32,"bytes":64*1024*1024,"max_cycles":1000000000}
    if mode=="paired":options.update(tile_pipeline=True,template_load_timing=True)
    o=path/"options.json";o.write_text(json.dumps(options))
    command=[str(binary),str(p),str(path/"out"),"none","1"] if mode=="tensor" else [str(binary),str(p),str(o),str(path/"out")]
    result=subprocess.run(command,capture_output=True,text=True,timeout=600)
    (path/"run.log").write_text(result.stdout+result.stderr)
    if error:
        assert result.returncode!=0 and error in result.stderr,result.stdout+result.stderr
        assert not (path/"out/result.json").exists();return
    assert result.returncode==0,result.stdout+result.stderr
    report=json.loads((path/"out/result.json").read_text());verify_source_groups(program,report)
    if mode=="physical":verify_physical_execution(program,report,options)
    if mode=="paired":verify_ready_execution(program,report,options)
    output=report["outputs"][0];dtype=np.float16 if output["dtype"]=="f16" else np.float32
    values=np.fromfile(output["logits_file"],dtype=dtype).reshape(output["shape"])
    assert values.argmax(-1).reshape(-1).tolist()==output["tokens"]
    return report,values


@pytest.mark.parametrize("dtype",[torch.float16,torch.float32])
def test_gelu_tail_boundaries_and_signed_zero_match_explicit_recipe(native,tmp_path,dtype):
    x=torch.tensor([[-65504.,-8.,-4.,-1.,-.01,-0.,0.,.01,1.,4.,8.,65504.]],dtype=dtype)
    p,coverage,_,framework=capture(tmp_path,x)
    report,actual=run(native[0],tmp_path/"native",p)
    expected=recipe_reference(x);unsigned=np.uint16 if dtype==torch.float16 else np.uint32
    np.testing.assert_array_equal(actual.view(unsigned),expected.view(unsigned))
    assert coverage["source_calls"]==report["executed_source_calls"]==2
    assert report["executed_lowered_calls"]==len(p["nodes"])==(34 if dtype==torch.float16 else 32)
    limit=F16_ABS_LIMIT if dtype==torch.float16 else F32_ABS_LIMIT
    np.testing.assert_allclose(actual,mathematical_reference(x),rtol=0,atol=limit)
    np.testing.assert_allclose(actual,framework,rtol=0,atol=limit)


def test_every_finite_fp16_encoding_is_evaluated_in_cpp(native,tmp_path):
    bits=np.arange(65536,dtype=np.uint16);values=bits.view(np.float16);values=values[np.isfinite(values)].copy()
    assert values.size==63488
    x=torch.from_numpy(values.reshape(1,-1));p,_,_,framework=capture(tmp_path,x)
    _,actual=run(native[0],tmp_path/"all-f16",p)
    expected=recipe_reference(x);np.testing.assert_array_equal(actual.view(np.uint16),expected.view(np.uint16))
    reference=mathematical_reference(x);error=float(np.max(np.abs(actual.astype(np.float64)-reference)))
    assert error<=F16_ABS_LIMIT
    framework_error=float(np.max(np.abs(actual.astype(np.float64)-framework.astype(np.float64))))
    assert framework_error<=F16_ABS_LIMIT
    (tmp_path/"numeric-domain.json").write_text(json.dumps({"domain":"all_63488_finite_fp16_encodings","elements":63488,"math_max_abs":error,"framework_max_abs":framework_error,"limit":F16_ABS_LIMIT,"all_fp32_inputs_verified":False})+"\n")


def test_fp32_dense_and_bitpattern_samples_have_bounded_error(native,tmp_path):
    rng=np.random.default_rng(3919)
    raw=rng.integers(0,2**32,size=32768,dtype=np.uint32).view(np.float32)
    raw=raw[np.isfinite(raw)]
    values=np.concatenate([np.linspace(-10,10,32768,dtype=np.float32),raw])[:65536].reshape(1,-1).copy()
    x=torch.from_numpy(values);p,_,_,framework=capture(tmp_path,x)
    _,actual=run(native[0],tmp_path/"f32-sample",p)
    expected=recipe_reference(x);np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
    reference=mathematical_reference(x);error=float(np.max(np.abs(actual.astype(np.float64)-reference)))
    assert error<=F32_ABS_LIMIT
    finite=np.isfinite(framework)
    framework_error=float(np.max(np.abs(actual[finite].astype(np.float64)-framework[finite].astype(np.float64))))
    nonfinite=int((~finite).sum())
    assert framework_error<=F32_ABS_LIMIT
    (tmp_path/"numeric-domain.json").write_text(json.dumps({"domain":"dense_interval_plus_seeded_finite_fp32_bitpatterns","elements":values.size,"math_max_abs":error,
        "finite_framework_max_abs":framework_error,"framework_nonfinite_outputs":nonfinite,"framework_comparison_passed":nonfinite==0 and framework_error<=F32_ABS_LIMIT,
        "framework_nonfinite_input_examples":values[~finite][:8].tolist(),"limit":F32_ABS_LIMIT,"all_fp32_inputs_verified":False,"framework_gate_overridden":False})+"\n")


@pytest.mark.parametrize("shape",[(2,33),(1,28,3072)])
def test_gelu_actual_operator_shapes_execute_through_scheduled_physical_memory(physical_binary,tmp_path,shape):
    x=torch.linspace(-5,5,int(np.prod(shape)),dtype=torch.float32).reshape(shape)
    p,_,_,framework=capture(tmp_path,x,True)
    report,actual=run(physical_binary,tmp_path/"physical",p,"physical")
    np.testing.assert_array_equal(actual.view(np.uint32),recipe_reference(x).view(np.uint32))
    np.testing.assert_allclose(actual,framework,rtol=0,atol=F32_ABS_LIMIT)
    assert report["blas_calls"]==report["functional_entry_calls"]==0 and report["arena_drained"]["reserved_bytes"]==0


def test_gelu_shared_pair_execution_keeps_all_original_source_groups(graph_binary,tmp_path):
    x=torch.linspace(-5,5,64).reshape(2,32);p,_,_,_=capture(tmp_path,x,True)
    p=compile_block_pipelines(p,event_slots=4)
    _,actual=run(graph_binary,tmp_path/"paired",p,"paired")
    np.testing.assert_array_equal(actual.view(np.uint32),recipe_reference(x).view(np.uint32))


@pytest.mark.parametrize("damage",["coefficient","literal","wiring","clip","dtype","profile","stage"])
def test_gelu_recipe_cannot_be_relabelled_or_silently_changed(graph_binary,tmp_path,damage):
    p,_,_,_=capture(tmp_path,torch.tensor([[-1.,1.]]),True);g=p["source_groups"][0]
    if damage=="coefficient":g["gelu_config"]["coefficients"][0]=.3
    if damage=="literal":next(n for n in p["nodes"] if n["lowering_stage_name"]=="poly_add1")["args"][1]=.3
    if damage=="wiring":next(n for n in p["nodes"] if n["lowering_stage_name"]=="half_input")["args"][0]={"value":p["nodes"][3]["id"]}
    if damage=="clip":g["gelu_config"]["clip"]=4
    if damage=="dtype":g["gelu_config"]["output_dtype"]="torch.float16"
    if damage=="profile":g["lowering_profile"]="direct"
    if damage=="stage":g["lowered_ids"].pop();g["stage_names"].pop();g["lowered_kinds"].pop()
    with pytest.raises(ValueError):validate_source_groups(p)
    run(graph_binary,tmp_path/"bad",p,"paired",error="recipe" if damage in {"profile","stage"} else "GELU")


@pytest.mark.parametrize("damage",["tanh","dtype","shape","mutable"])
def test_gelu_rejects_unregistered_source_contracts(tmp_path,damage):
    _,_,inventory,_=capture(tmp_path,torch.tensor([[1.,2.]]));event=copy.deepcopy(inventory["operations"][0])
    if damage=="tanh":event["kwargs"]["approximate"]="tanh"
    if damage=="dtype":event["inputs"][0]["dtype"]="torch.bfloat16"
    if damage=="shape":event["outputs"]["shape"]=[2,1]
    if damage=="mutable":event["mutable"]=True
    with pytest.raises(ValueError):gelu_contract(event)
