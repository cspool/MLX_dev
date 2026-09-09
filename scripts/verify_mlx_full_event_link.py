"""Audit compact full-model linkage against independently rebuilt window files.

Does not call the linker, expand model-wide blocks, or claim execution success.
"""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.mlx_system_attempt import digest, record
from scripts.verify_mlx_event_schedule import require


def shape(events):
    """Discard local names only; preserve every instruction field and loop."""
    result = []
    for item in events:
        row = dict(item)
        if 'repeat' in row:
            row['body'] = shape(row['body'])
        else:
            require(not row['dependencies'], 'unlinked event dependency')
            row.pop('id')
        result.append(row)
    return result


def signature(events):
    return hashlib.sha256(json.dumps(shape(events), sort_keys=True).encode()).hexdigest()


def count(events):
    return sum(e['repeat']*count(e['body']) if 'repeat' in e else 1 for e in events)


def operations(events):
    for e in events:
        if 'repeat' in e:yield from operations(e['body'])
        else:yield e['op']


def audit_graph(graph, compiled, program, read_window, read_block):
    rows = compiled['windows']
    require(compiled['all_window_patterns_compiled'] and not compiled['blocked_windows'], 'incomplete windows')
    bindings = {(r['binding']['source_operator_id'],r['binding']['batch_index']):r for r in rows}
    require(len(bindings)==len(rows), 'duplicate window')
    nodes = {n['source_operator_id']:n for n in program['nodes']}
    streams = graph['source_streams']
    require([s['source_operator_id'] for s in streams]==list(nodes), 'source order/coverage')
    require(graph['schema']=='mlx_event_schedule_v6' and graph['source_tick_order'] is True, 'execution profile')
    require(not graph['blocks'] and not graph['controllers'], 'unexpected eager work')
    templates = {t['id']:t for t in graph['templates']}
    require(len(templates)==len(graph['templates']), 'duplicate templates')
    blocks = {}; used_templates=set(); seen=set(); checked={}; total_blocks=total_events=0
    for key, meta in graph['event_patterns'].items():
        raw = read_block(meta)
        require(raw['schema']=='mlx_block_event_pattern_v1', 'block schema')
        require(count(raw['events'])==meta['dynamic_events'], 'block dynamic count')
        blocks[key]=(signature(raw['events']),meta['dynamic_events'])
    windows=0; code_identity={}
    for source in streams:
        sid=source['source_operator_id'];node=nodes[sid]
        require((source['operator_kind'],source['output_shape'],source['output_dtype'])==
                (node['kind'],node['output']['shape'],node['output']['dtype']), 'source output binding')
        for batch, window in enumerate(source['windows']):
            require((sid,batch) in bindings, 'foreign window')
            row=bindings[(sid,batch)];binding=row['binding'];windows+=1
            require(source['parents']==binding['parents'] and source['family']==binding['family'] and
                    len(source['windows'])==binding['batch_count'], 'source dependency/batch binding')
            phase = ([len(node['matrix_program'][k]) for k in ('prologue','body','epilogue')]
                     if 'matrix_program' in node else node.get('vector_program',{}).get('phases'))
            cache=(row['pattern'],json.dumps(window,sort_keys=True),json.dumps(phase,sort_keys=True))
            if cache not in checked:
                original=read_window(row['pattern']);old=original['blocks']+original.get('controllers',[])
                hardware=original['hardware'];target_hardware=graph['hardware']
                if source['family'] in ('matrix','vector'):
                    fields=['rows','columns','contexts','rf_vectors_per_pe','spm_vectors_total','rom_words_per_pe',
                            'spm_port_period','writeback_period','dma_request_period','dma_response_period','compute_ii']
                    if source['family']=='vector':fields.append('sfu_ii')
                    require(all(hardware[k]==target_hardware[k] for k in fields),'shared hardware changed')
                else:
                    field=source['family']+'_controller'
                    require(hardware[field]==target_hardware[field],'private controller changed')
                old_templates={t['id']:t for t in original['templates']}
                cursor=events=0
                for run in window['runs']:
                    require(type(run['count']) is int and run['count']>0, 'invalid run count')
                    require(cursor+run['count']<=len(old), 'extra logical blocks')
                    key=run['event_pattern'];require(key in blocks, 'missing block pattern');seen.add(key)
                    for offset in range(run['count']):
                        block=old[cursor+offset]
                        for op in set(operations(block['events'])):
                            if op not in ('dma_read','dma_write','memory_index_read','memory_predicate_read'):
                                require(hardware['latencies'][op]==target_hardware['latencies'][op],'execution latency changed')
                        require(signature(block['events'])==blocks[key][0], 'block instructions/loops changed')
                        require(block['source_operator_id']==0 and not block['admission_dependencies'], 'window ownership')
                        if source['family'] in ('matrix','vector'):
                            pe=(run['pe_base']+offset*run['pe_stride'])%(graph['hardware']['rows']*graph['hardware']['columns'])
                            require(pe==block['pe'], 'PE mapping changed')
                            target=templates[run['template']];origin=old_templates[block['template']]
                            require(all(target[k]==origin[k] for k in ('words','rf_vectors','spm_vectors')), 'template words/resources')
                            identity=json.dumps([source['family'],phase,origin['words']],sort_keys=True)
                            require(target['id'] not in code_identity or code_identity[target['id']]==identity, 'incompatible code sharing')
                            code_identity[target['id']]=identity;used_templates.add(target['id'])
                    cursor+=run['count'];events+=run['count']*blocks[key][1]
                require(cursor==len(old), 'omitted logical blocks')
                checked[cache]=(cursor,events)
            n,e=checked[cache];total_blocks+=n;total_events+=e
    require(windows==len(rows), 'omitted windows')
    require(seen==set(blocks) and used_templates==set(templates), 'unreferenced code/patterns')
    plan=program['block_pipeline_plan']
    expected=[{k:p[k] for k in ('producer_source','consumer_source','mapping')}|dict(event_slots=plan['event_slots']) for p in plan['pairs']]
    require(graph['streaming_pairs']==expected, 'original streaming pairs changed')
    return dict(windows=windows,sources=len(streams),logical_blocks=total_blocks,dynamic_events=total_events,
                independently_checked_window_mappings=len(checked),streaming_pairs=len(expected))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('linked','compiled','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();require(not args.output.exists(),'choose fresh audit output')
    linked=args.linked.resolve();compiled=args.compiled.resolve()
    meta=json.loads((linked/'manifest.json').read_text());m=json.loads((compiled/'manifest.json').read_text())
    inputs=dict(meta['inputs']);inputs[str(linked/'manifest.json')]=digest(linked/'manifest.json')
    inputs[str(linked/'graph.json')]=meta['graph_sha256']
    require(all(digest(Path(p))==h for p,h in inputs.items()),'link inputs changed')
    require(inputs.get(str(compiled/'manifest.json'))==digest(compiled/'manifest.json'),'different window catalogue')
    graph=json.loads((linked/'graph.json').read_text());program=json.loads(Path(m['program_path']).read_text())
    options=meta['runtime_options']
    require(options.get('tile_pipeline',False) is True and options.get('overlap',True) is True and
            options.get('pipeline_whole_source_barrier',False) is False,'different concurrency mode')
    require(graph['physical_memory']=={k:options.get('memory',{}).get(k,v) for k,v in
            dict(latency=8,accept_period=1,nack_every=0).items()},'physical memory profile changed')
    require(graph['hardware']['template_load_timing']==options.get('template_load_timing',False) and
            graph['hardware']['source_window_limit']==options.get('max_active_nodes',32) and
            graph['max_cycles']==options.get('max_cycles',10000000000000),'runtime timing/cap changed')
    def read_window(key):return json.loads((compiled/'patterns'/(key+'.json')).read_text())
    def read_block(meta):
        p=Path(meta['path']);require(digest(p)==meta['sha256'],'block file changed');inputs[str(p)]=meta['sha256']
        return json.loads(p.read_text())
    result=audit_graph(graph,m,program,read_window,read_block)
    require(all(result[k]==meta[k] for k in ('windows','logical_blocks','dynamic_events','streaming_pairs')),'link summary differs')
    require(all(digest(Path(p))==h for p,h in inputs.items()),'audit inputs changed')
    record(args.output,dict(classification='independent_full_event_link_audit_not_execution',**result,inputs=inputs,
                           auditor_sha256=digest(Path(__file__)),full_model_execution_verified=False,final_readback_included=False))
    print('FULL_EVENT_LINK_AUDIT_PASS '+json.dumps(result),flush=True)


if __name__=='__main__':main()
