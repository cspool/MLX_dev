"""Inspect local checkpoint metadata without executing custom model code."""
import argparse
import ast
from collections import Counter
import importlib
import json
import math
from pathlib import Path
import re
import struct

import torch
import transformers

from mlxsim.model_tensor_compiler import DTYPES,ROUTES
from scripts.run_mlx_tensor_semantics import ROOT,sha

WIDTH={'F16':2,'BF16':2,'F32':4,'F64':8,'I64':8,'I32':4,'I16':2,'I8':1,'U8':1,'BOOL':1}
TORCH_DTYPE={'F16':'torch.float16','BF16':'torch.bfloat16','F32':'torch.float32','F64':'torch.float64','I64':'torch.int64','I32':'torch.int32','I16':'torch.int16','I8':'torch.int8','U8':'torch.uint8','BOOL':'torch.bool'}


def reject_pointer(file):
    with file.open('rb') as stream:prefix=stream.read(64)
    if prefix.startswith(b'version https://git-lfs.github.com/spec/v1'):raise ValueError('checkpoint is an LFS pointer, not tensor data')


def safe_shard(root,name):
    file=(root/name).resolve()
    if not file.is_relative_to(root) or not file.is_file():raise ValueError('checkpoint shard is missing or outside model directory')
    reject_pointer(file);return file


def safetensors_metadata(file):
    reject_pointer(file);size=file.stat().st_size
    with file.open('rb') as stream:
        raw=stream.read(8)
        if len(raw)!=8:raise ValueError('truncated safetensors header')
        length=struct.unpack('<Q',raw)[0]
        if not 0<length<=16*2**20 or length+8>size:raise ValueError('invalid safetensors header extent')
        header=json.loads(stream.read(length))
    result={};ranges=[]
    for key,value in header.items():
        if key=='__metadata__':continue
        shape=value['shape'];dtype=value['dtype'];start,end=value['data_offsets']
        if dtype not in WIDTH or any(type(n) is not int or n<0 for n in shape) or type(start) is not int or type(end) is not int or not 0<=start<=end<=size-8-length or end-start!=math.prod(shape)*WIDTH[dtype]:raise ValueError('invalid safetensors tensor extent/type')
        result[key]={'shape':shape,'dtype':TORCH_DTYPE[dtype],'elements':math.prod(shape),'bytes':end-start}
        if end>start:ranges.append((start,end))
    ranges.sort()
    if any(a[1]>b[0] for a,b in zip(ranges,ranges[1:])):raise ValueError('overlapping safetensors tensor data')
    return result


def dependencies(file):
    """Only inspect symbols in installed trusted frameworks; never import model code."""
    rows=[]
    for node in ast.parse(file.read_text()).body:
        if not isinstance(node,ast.ImportFrom) or node.level or not node.module:continue
        if node.module.split('.')[0] not in {'torch','transformers','einops'}:continue
        try:
            module=importlib.import_module(node.module);rows.append({'module':node.module,'required':[n.name for n in node.names],'missing':[n.name for n in node.names if not hasattr(module,n.name)]})
        except Exception as error:rows.append({'module':node.module,'import_error':type(error).__name__+': '+str(error)})
    return rows


def inspect(root):
    root=Path(root).resolve();config_file=root/'config.json';config=json.loads(config_file.read_text());files={str(config_file):sha(config_file)};tensors={};shards=[]
    index_file=root/'pytorch_model.bin.index.json'
    if index_file.exists():
        index=json.loads(index_file.read_text());files[str(index_file)]=sha(index_file)
        for name in sorted(set(index['weight_map'].values())):
            file=safe_shard(root,name);files[str(file)]=sha(file)
            values=torch.load(file,map_location='meta',weights_only=True,mmap=True)
            expected={key for key,value in index['weight_map'].items() if value==name}
            if set(values)!=expected or set(tensors)&set(values):raise ValueError('checkpoint tensor keys do not match the shard index')
            for key,value in values.items():
                if not isinstance(value,torch.Tensor) or value.device.type!='meta':raise ValueError('checkpoint metadata is not a meta tensor')
                tensors[key]={'shape':list(value.shape),'dtype':str(value.dtype),'elements':value.numel(),'bytes':value.numel()*value.element_size()}
            shards.append({'file':name,'file_bytes':file.stat().st_size,'tensor_count':len(values)})
        if sum(t['bytes'] for t in tensors.values())!=index['metadata']['total_size']:raise ValueError('checkpoint payload size differs from index')
        format_name='pytorch_weights_only_meta'
    else:
        file=safe_shard(root,'model.safetensors');files[str(file)]=sha(file);tensors=safetensors_metadata(file)
        shards=[{'file':file.name,'file_bytes':file.stat().st_size,'tensor_count':len(tensors)}];format_name='safetensors_header_only'
    code=[]
    for file in sorted(root.glob('*.py')):
        files[str(file)]=sha(file);code.append({'file':file.name,'top_level_framework_symbols':dependencies(file)})
    for name in ('tokenizer.json','tokenizer.model','tokenizer_config.json','special_tokens_map.json','generation_config.json'):
        file=root/name
        if file.exists():files[str(file)]=sha(file)
    dtypes=Counter(t['dtype'] for t in tensors.values());layers=sorted({int(m.group(1)) for key in tensors if (m:=re.search(r'(?:layers|layer)\.(\d+)\.',key))})
    if layers!=list(range(config['num_hidden_layers'])):raise ValueError('checkpoint metadata does not cover every configured layer')
    gaps=[]
    unsupported=sorted(set(dtypes)-set(DTYPES))
    if unsupported:gaps.append({'kind':'unsupported_tensor_precision','dtypes':unsupported,'action':'define and validate a backend precision path or explicitly selected conversion contract; do not silently cast'})
    if config['model_type']!='llama':gaps.append({'kind':'reference_entry_not_registered','action':'current capture_mlx_model_reference only registers full Llama2; add a separately bound model/task entry'})
    structured=sorted(key for key in tensors if key.endswith('.factors'))
    if structured:gaps.append({'kind':'structured_constructor_required','tensors':structured,'action':'bind the training/reconstruction module before strict checkpoint loading'})
    missing=[item for module in code for item in module['top_level_framework_symbols'] if item.get('missing') or item.get('import_error')]
    if missing:gaps.append({'kind':'framework_import_symbols_missing','items':missing})
    if any(sha(Path(file))!=digest for file,digest in files.items()):raise ValueError('model materials changed during inspection')
    return {'classification':'local_checkpoint_metadata_and_entry_gap_audit_not_model_execution','path':str(root),'model_type':config['model_type'],'architectures':config.get('architectures'),
            'config':config,'format':format_name,'shards':shards,'tensor_count':len(tensors),'tensor_elements':sum(t['elements'] for t in tensors.values()),'payload_bytes':sum(t['bytes'] for t in tensors.values()),
            'tensor_dtypes':dict(dtypes),'layer_indices':layers,'tensors':tensors,'custom_code':code,'gaps':gaps,'files':files,'compiler_dtypes':DTYPES,'compiler_direct_operator_routes':sorted(ROUTES),
            'torch':torch.__version__,'transformers':transformers.__version__,'custom_model_code_executed':False,'model_loaded_for_inference':False,'operator_trace_captured':False,
            'author_hybrid_identity_verified':False,'full_model_execution_verified':False,'inference_performance_eligible':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--model',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise RuntimeError('choose a fresh materials audit file')
    sources={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__).resolve(),ROOT/'src/mlxsim/model_tensor_compiler.py',ROOT/'scripts/capture_mlx_model_reference.py')}
    report=inspect(args.model)
    if any(sha(ROOT/p)!=h for p,h in sources.items()):raise RuntimeError('materials audit source changed during inspection')
    report['audit_sources']=sources
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n');print(f'MODEL_MATERIALS_INSPECTED {args.output}')


if __name__=='__main__':main()
