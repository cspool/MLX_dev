import json
from pathlib import Path
import struct

import pytest
import torch

from scripts.audit_mlx_model_materials import inspect,safe_shard,safetensors_metadata


def config(root,kind='internlm2'):
    (root/'config.json').write_text(json.dumps({'model_type':kind,'num_hidden_layers':1,'architectures':['ExampleModel']}))


def test_bf16_metadata_stays_bf16_and_does_not_execute_custom_code(tmp_path):
    config(tmp_path);tensor=torch.ones(2,3,dtype=torch.bfloat16)
    torch.save({'model.layers.0.weight':tensor},tmp_path/'weights.bin')
    (tmp_path/'pytorch_model.bin.index.json').write_text(json.dumps({'metadata':{'total_size':12},'weight_map':{'model.layers.0.weight':'weights.bin'}}))
    (tmp_path/'modeling_example.py').write_text("raise RuntimeError('must never execute model code')\n")
    result=inspect(tmp_path)
    assert result['tensor_elements']==6 and result['tensor_dtypes']=={'torch.bfloat16':1}
    assert any(g['kind']=='unsupported_tensor_precision' for g in result['gaps'])
    assert not result['custom_model_code_executed'] and not result['model_loaded_for_inference'] and not result['full_model_execution_verified']


@pytest.mark.parametrize('damage',['index','payload','layers','lfs'])
def test_indexed_checkpoint_rejects_false_metadata_and_pointer_files(tmp_path,damage):
    config(tmp_path);torch.save({'model.layers.0.weight':torch.ones(2)},tmp_path/'weights.bin')
    index={'metadata':{'total_size':8},'weight_map':{'model.layers.0.weight':'weights.bin'}}
    if damage=='index':index['weight_map']={'missing.weight':'weights.bin'}
    elif damage=='payload':index['metadata']['total_size']=4
    elif damage=='layers':(tmp_path/'config.json').write_text(json.dumps({'model_type':'internlm2','num_hidden_layers':2}))
    else:(tmp_path/'weights.bin').write_text('version https://git-lfs.github.com/spec/v1\noid sha256:abc\n')
    (tmp_path/'pytorch_model.bin.index.json').write_text(json.dumps(index))
    with pytest.raises(ValueError):inspect(tmp_path)


def test_shards_cannot_escape_the_declared_model_directory(tmp_path):
    model=tmp_path/'model';model.mkdir();outside=tmp_path/'outside.bin';outside.write_bytes(b'not a shard')
    with pytest.raises(ValueError):safe_shard(model.resolve(),'../outside.bin')
    with pytest.raises(ValueError):safe_shard(model.resolve(),'missing.bin')


def write_header(file,header,payload=16):
    data=json.dumps(header).encode();file.write_bytes(struct.pack('<Q',len(data))+data+bytes(payload))


def test_structured_factors_are_not_relabelled_as_dense_weights(tmp_path):
    config(tmp_path,'bert');write_header(tmp_path/'model.safetensors',{'bert.encoder.layer.0.query.factors':{'dtype':'F32','shape':[2,2],'data_offsets':[0,16]}})
    result=inspect(tmp_path)
    assert result['tensor_elements']==4 and any(g['kind']=='structured_constructor_required' for g in result['gaps'])
    assert not result['author_hybrid_identity_verified'] and not result['operator_trace_captured']


@pytest.mark.parametrize('damage',['shape','dtype','offset','bool_offset','overlap','header'])
def test_bad_safetensors_metadata_is_rejected(tmp_path,damage):
    file=tmp_path/'model.safetensors';entry={'dtype':'F32','shape':[2,2],'data_offsets':[0,16]};header={'x':entry}
    if damage=='shape':entry['shape']=[True,4]
    elif damage=='dtype':entry['dtype']='UNKNOWN'
    elif damage=='offset':entry['data_offsets']=[0,20]
    elif damage=='bool_offset':entry['data_offsets']=[False,16]
    elif damage=='overlap':header['y']=dict(entry)
    write_header(file,header)
    if damage=='header':file.write_bytes(struct.pack('<Q',2**40))
    with pytest.raises(ValueError):safetensors_metadata(file)
