"""Own a complete audited C++ event-graph run, separate from numerical/system acceptance."""
import argparse
import json
from pathlib import Path
import shutil

from scripts.mlx_system_attempt import digest, record, run_process, linked_libraries
from scripts.verify_mlx_event_schedule import require, audit_trace


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('linked','audit','binary','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--host-timeout',type=int,default=172800)
    args=parser.parse_args();out=args.output.resolve();linked=args.linked.resolve()
    require(not out.exists(),'choose a fresh full event attempt')
    audit=json.loads(args.audit.read_text());inputs=dict(audit['inputs'])
    inputs[str(args.audit.resolve())]=digest(args.audit)
    require(audit['classification']=='independent_full_event_link_audit_not_execution','missing full link audit')
    require(inputs.get(str(linked/'graph.json'))==digest(linked/'graph.json'),'different graph audit')
    require(all(digest(Path(p))==h for p,h in inputs.items()),'audited input changed')
    root=Path(__file__).resolve().parents[1]
    require(audit['auditor_sha256']==digest(root/'scripts/verify_mlx_full_event_link.py'),'auditor changed')
    sources={str(p.relative_to(root)):digest(p) for p in (root/'simulator_ext/event_schedule').iterdir() if p.is_file()}
    for name in ('simulator_ext/model_events/completion_window.cc','simulator_ext/model_events/completion_window.h',
                 'scripts/run_mlx_full_event_graph.py','scripts/mlx_system_attempt.py','scripts/verify_mlx_event_schedule.py'):
        sources[name]=digest(root/name)
    out.mkdir(parents=True);binary=out/'event-schedule';shutil.copy2(args.binary,binary)
    sha=digest(binary);libraries=linked_libraries(binary)
    run_process([str(binary),str(linked/'graph.json'),str(out/'result.json')],out/'run.log',out/'execution.json',
                timeout=args.host_timeout,metadata=dict(mode='full_model_concurrent_event_graph_only',binary_sha256=sha,
                inputs=inputs,sources=sources,runtime_libraries=libraries,final_readback_included=False,
                full_model_numerical_execution=False,mlx_system_verified=False))
    require(digest(binary)==sha and all(digest(Path(p))==h for p,h in inputs.items()) and
            all(digest(root/p)==h for p,h in sources.items()) and all(digest(Path(p))==h for p,h in libraries.items()),'run provenance changed')
    result=json.loads((out/'result.json').read_text());graph=json.loads((linked/'graph.json').read_text())
    require(result['source_graph_completed'] and result['all_resources_drained'],'event graph not drained')
    require(result['blocks']==audit['logical_blocks'] and result['events']==audit['dynamic_events'],'complete event work differs')
    require(len(result['source_intervals'])==audit['sources'] and len(result['pipeline_groups'])==audit['streaming_pairs'],'source/pair coverage differs')
    audit_trace(graph,result)
    record(out/'report.json',dict(classification='complete_concurrent_model_event_graph_not_numerical_or_system_acceptance',
           cycles=result['cycles'],logical_blocks=result['blocks'],dynamic_events=result['events'],
           event_graph_execution_verified=True,final_readback_included=False,model_performance_error_available=False,
           full_model_numerical_execution=False,mlx_system_verified=False,result_sha256=digest(out/'result.json')))
    print('FULL_MODEL_EVENT_GRAPH_COMPLETED cycles='+str(result['cycles']),flush=True)


if __name__=='__main__':main()
