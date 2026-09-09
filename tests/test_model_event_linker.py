"""Full catalogue linking fixtures; constant token is not a model acceptance test."""
import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from mlxsim.model_event_linker import ModelEventLinker, canonical
from mlxsim.model_event_windows import WindowCatalog
from mlxsim.model_block_pipeline import compile_block_pipelines
from test_streaming_pair_events import extra_program
from test_model_tensor_semantics import literal
from test_model_tensor_semantics import node, ref
from test_physical_model import outputs, compiled_nodes
from mlxsim.model_matrix_program import matrix_program
from mlxsim.model_vector_program import vector_program
from mlxsim.model_control_program import control_program
from test_ready_graph import binary as native_binary, execute
from test_event_schedule import binary
from test_event_loops import run
from scripts.verify_mlx_full_event_link import audit_graph
import json


def fixture_program(kind):
    p, expected, _ = extra_program(kind)
    # Deliberately graph-only fixture: do not invent an argmax branch witness.
    p['nodes'].pop()
    p['assets']['fixture_token'] = literal(torch.tensor(0))
    p['outputs'][0]['token'] = 'fixture_token'
    p.pop('block_pipeline_plan')
    return compile_block_pipelines(p, event_slots=2), expected


def catalogue(p):
    c = WindowCatalog(p)
    patterns, rows = {}, []
    for s in c.plan['sources']:
        for batch in range(s['window_count']):
            job, binding = c.window(s['source_operator_id'], batch)
            key = hashlib.sha256(canonical(job).encode()).hexdigest()
            if key not in patterns:
                patterns[key] = c.compile(job)
            rows.append(dict(pattern=key, binding=binding))
    return dict(all_window_patterns_compiled=True, blocked_windows=[], windows=rows), patterns


@pytest.mark.parametrize('kind', ['batched', 'neg', 'mean', 'softmax'])
@pytest.mark.parametrize('timed', [False, True])
def test_linked_complete_fixture_matches_native_graph(binary, native_binary, tmp_path, kind, timed):
    p, expected = fixture_program(kind)
    manifest, patterns = catalogue(p)
    options = dict(tile_pipeline=True, template_load_timing=timed,
                   memory=dict(latency=8, accept_period=1, nack_every=0))
    linker = ModelEventLinker(p, manifest, lambda k: copy.deepcopy(patterns[k]), tmp_path/'patterns', options)
    graph = linker.link()
    native = execute(native_binary, tmp_path/'native', p, options)
    actual, tokens = outputs(native)[0]
    np.testing.assert_allclose(actual, expected, rtol=5e-6, atol=5e-6)
    assert tokens == [0]
    event = run(binary, tmp_path/'event', graph)
    assert event['cycles'] == native['graph_cycles']
    assert len(graph['source_streams']) == len(p['nodes'])
    assert len(graph['streaming_pairs']) == len(p['block_pipeline_plan']['pairs'])
    assert graph['source_streams'][1]['parents'] == [0]
    assert sum(len(s['windows']) for s in graph['source_streams']) == len(manifest['windows'])
    assert linker.logical_blocks == sum(r['count'] for s in graph['source_streams'] for w in s['windows'] for r in w['runs'])
    if kind == 'batched':
        assert len(graph['source_streams'][0]['windows']) == 2
        assert len(linker.window_runs) < len(manifest['windows'])


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'parents', 'latency', 'resources', 'retry', 'pair'])
def test_linker_rejects_incomplete_or_incompatible_catalogue(tmp_path, damage):
    p, _ = fixture_program('neg')
    manifest, patterns = catalogue(p)
    options = dict(tile_pipeline=True)
    if damage == 'missing': manifest['all_window_patterns_compiled'] = False
    elif damage == 'duplicate': manifest['windows'].append(copy.deepcopy(manifest['windows'][0]))
    elif damage == 'parents': manifest['windows'][1]['binding']['parents'] = []
    elif damage == 'latency':
        for pattern in patterns.values(): pattern['hardware']['latencies']['neg'] = 99
    elif damage == 'resources':
        for pattern in patterns.values(): pattern['hardware']['rf_vectors_per_pe'] = 99
    elif damage == 'retry': options['memory'] = dict(nack_every=2)
    else: p['block_pipeline_plan']['pairs'] = []
    with pytest.raises(ValueError):
        ModelEventLinker(p, manifest, lambda k: copy.deepcopy(patterns[k]), tmp_path/'patterns', options).link()
    assert not (tmp_path/'patterns').exists()


@pytest.mark.parametrize('timed', [False, True])
def test_all_four_backends_keep_external_pair_parents(binary, native_binary, tmp_path, timed):
    a = torch.arange(6).float().reshape(2,3)
    b = torch.arange(15).float().reshape(3,5)
    mask = torch.tensor([True, False, True])
    nodes = [node(0, 'bitwise_and', [ref('mask'), ref('mask')], mask),
             node(1, 'cast', [ref('a'), 'torch.float32', False, True], a.clone()),
             node(2, 'matmul', [ref('v1'), ref('b')], a@b),
             node(3, 'neg', [ref('v2')], -(a@b)),
             node(4, 'all', [ref('v0')], mask.all())]
    nodes[0]['control_program'] = control_program('bitwise_and')
    nodes[2]['matrix_program'] = matrix_program('f32', 'f32')
    nodes[3]['vector_program'] = vector_program('neg', ['f32'], 'f32')
    nodes[4]['control_program'] = control_program('all')
    p = compiled_nodes({k:literal(v) for k,v in dict(a=a,b=b,mask=mask,token=torch.tensor(0)).items()}, nodes, 'v3', 'token')
    p = compile_block_pipelines(p, event_slots=2)
    manifest, patterns = catalogue(p)
    options = dict(tile_pipeline=True, template_load_timing=timed)
    graph = ModelEventLinker(p, manifest, lambda k:copy.deepcopy(patterns[k]), tmp_path/'patterns', options).link()
    assert [s['parents'] for s in graph['source_streams']] == [[],[],[1],[2],[0]]
    assert len(graph['streaming_pairs']) == 1
    native = execute(native_binary, tmp_path/'native', p, options)
    np.testing.assert_array_equal(outputs(native)[0][0], -(a@b).numpy())
    event = run(binary, tmp_path/'event', graph)
    assert event['cycles'] == native['graph_cycles']
    native_sources = {s['source_operator_id']:s for s in native['events']}
    for s in event['source_intervals']:
        n = native_sources[s['source_operator_id']]
        assert (s['begin_cycle'],s['publish_cycle']) == (n['start_cycle'],n['publish_cycle'])


@pytest.mark.parametrize('damage', [None,'parents','pe','count','events','template','pair','batch','latency','hardware'])
def test_independent_link_audit_checks_work_and_mapping(tmp_path, damage):
    p, _ = fixture_program('batched')
    manifest, patterns = catalogue(p)
    linker=ModelEventLinker(p,manifest,lambda k:copy.deepcopy(patterns[k]),tmp_path/'patterns',dict(tile_pipeline=True))
    graph=linker.link()
    if damage=='parents':graph['source_streams'][1]['parents']=[]
    elif damage=='pe':graph['source_streams'][0]['windows'][0]['runs'][0]['pe_base']+=1
    elif damage=='count':graph['source_streams'][0]['windows'][0]['runs'][0]['count']+=1
    elif damage=='template':graph['templates'][0]['rf_vectors']+=1
    elif damage=='pair':graph['streaming_pairs']=[]
    elif damage=='batch':graph['source_streams'][0]['windows'].pop()
    elif damage=='latency':graph['hardware']['latencies']['mul']+=1
    elif damage=='hardware':graph['hardware']['spm_port_period']+=1
    def read_block(meta):
        result=json.loads(Path(meta['path']).read_text())
        if damage=='events':result['events']=[]
        return result
    def check():return audit_graph(graph,manifest,p,lambda k:copy.deepcopy(patterns[k]),read_block)
    if damage:
        with pytest.raises(RuntimeError):check()
    else:
        result=check()
        assert result['logical_blocks']==linker.logical_blocks
        assert result['dynamic_events']==linker.dynamic_events
