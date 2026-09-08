"""Same-input GPU split-K policy probes; no model inference or gate changes."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from scripts.diagnose_mlx_exact_projections import compare
from scripts.run_mlx_tensor_semantics import ROOT,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('gpu','program','exact','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',default='cuda:1');args=parser.parse_args();out=args.output.resolve()
    if out.exists():raise RuntimeError('choose a fresh split-K diagnostic directory')
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';out.mkdir(parents=True);files={}
    def bind(file,expected=None):
        file=Path(file).resolve();digest=files.get(str(file)) or sha(file)
        if expected is not None and digest!=expected:raise RuntimeError('split-K diagnostic input fingerprint mismatch')
        files[str(file)]=digest
    for file in (Path(__file__).resolve(),ROOT/'scripts/diagnose_mlx_exact_projections.py',Path(torch.backends.cuda.__file__),args.program,args.gpu/'inventory.json',args.gpu/'diagnostics/observations.json',args.exact/'report.json'):bind(file)
    program=json.loads(args.program.read_text());inventory=json.loads((args.gpu/'inventory.json').read_text());exact=json.loads((args.exact/'report.json').read_text());observations={r['source_operator_id']:r for r in json.loads((args.gpu/'diagnostics/observations.json').read_text())['observations']}
    for file in (args.program,args.gpu/'inventory.json',args.gpu/'diagnostics/observations.json'):
        bind(file,exact['files'][str(file.resolve())])
    if inventory['runtime']['torch']!=torch.__version__ or inventory['runtime']['tf32'] or inventory['runtime']['fp16_reduced_precision_reduction']:raise RuntimeError('historical GPU environment/precision contract differs')
    torch.cuda.set_device(args.device);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_fp16_accumulation=False
    a=observations[49];bind(a['file'],a['sha256']);x=torch.from_numpy(np.fromfile(a['file'],dtype=np.float16).reshape(a['shape'])).to(args.device)
    results=[]
    for sid,name in ((50,'q'),(53,'k'),(56,'v')):
        node=program['nodes'][sid];weight=program['assets'][node['args'][1]['value']]
        if node['source_operator_id']!=sid or node['kind']!='linear' or node['args'][0]!={'value':'v49'} or weight['parameter_name']!=f'model.layers.0.self_attn.{name}_proj.weight' or weight['shape']!=[4096,4096] or weight['dtype']!='f16':raise RuntimeError('split-K probe changed actual projection operands')
        bind(weight['path'],weight['file_sha256']);original=observations[sid];bind(original['file'],original['sha256']);oracle=args.exact/name/'exact/exact.f16.bin';bind(oracle,exact['files'][str(oracle.resolve())])
        w=torch.from_numpy(np.fromfile(weight['path'],dtype=np.float16,count=4096*4096,offset=weight['byte_offset']).reshape(weight['shape'])).to(args.device)
        for backend,split in (('cublas',True),('cublaslt',True),('cublaslt',False)):
            torch.backends.cuda.preferred_blas_library(backend)
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=(False,split)
            if torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction or torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction_split_k!=split:raise RuntimeError('requested GPU reduction policy was not effective')
            target=torch.nn.functional.linear(x,w);torch.cuda.synchronize(args.device);path=out/f'{name}-{backend}-splitk-{int(split)}.f16.bin';path.write_bytes(target.cpu().numpy().tobytes());bind(path)
            results.append({'source_operator_id':sid,'projection':name,'requested_backend':backend,'reported_backend':str(torch.backends.cuda.preferred_blas_library()),'allow_reduced_precision':False,'allow_splitk':split,'allow_fp16_accumulation':torch.backends.cuda.matmul.allow_fp16_accumulation,
                            'vs_historical_gpu':compare(path,original['file']),'vs_exact':compare(path,oracle)})
        del w
    if any(sha(Path(file))!=digest for file,digest in files.items()):raise RuntimeError('split-K diagnostic inputs/outputs changed')
    properties=torch.cuda.get_device_properties(args.device)
    report={'classification':'same_input_gpu_projection_policy_diagnosis_not_model_execution_or_acceptance','files':files,'torch':torch.__version__,'cuda':torch.version.cuda,'device':args.device,'gpu_name':properties.name,'compute_capability':[properties.major,properties.minor],
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'tf32':torch.backends.cuda.matmul.allow_tf32,'cublas_workspace_config':os.environ['CUBLAS_WORKSPACE_CONFIG'],'results':results,
            'all_historical_outputs_reproduced_by_cublas':all(r['vs_historical_gpu']['bitwise_changed']==0 for r in results if r['requested_backend']=='cublas'),
            'original_gpu_reference_replaced':False,'original_gpu_tolerance_changed':False,'model_outputs_injected':False,'full_model_execution_verified':False,'inference_performance_eligible':False}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(f'GPU_SPLITK_DIAGNOSIS_COMPLETE {out/"report.json"}')


if __name__=='__main__':main()
