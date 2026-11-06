"""Uncached serialized graph execution over synthetic local shards only."""
import asyncio
import json
import pytest
from sciona import video_multilabel_sharded_training as lifecycle
from sciona.atoms.ml.masked_attention import masked_attention_pool, sigmoid_context_gate
from sciona.atoms.ml.model_selection.materialized_prediction import average
from sciona.atoms.ml.sklearn.feature_extraction.sparse_numeric import fit_numeric_vocabulary, transform_numeric_vocabulary
from sciona.atoms.ml.calibration.empirical_f1 import fit_empirical_f1_thresholds, apply_column_thresholds
from sciona.decomposed_video_multilabel_graph import build_decomposed_video_multilabel_graph
from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph
from sciona.video_multilabel_contract import prepare, execute as reference_execute
from sciona.video_multilabel_shards import FeatureShards
from sciona.visualizer import runner
from scripts.video_multilabel_synthetic import payload


@pytest.mark.parametrize('seed,epochs,batch_size', [(12,80,4),(3,5,3),(29,3,16)])
def test_serialized_native_graph_full_lifecycle(tmp_path, monkeypatch, seed, epochs, batch_size):
    p=payload();p['controls'].update(seed=seed,epochs=epochs,batch_size=batch_size)
    inputs={'controls':p['controls']}
    for name in ('training','calibration','query'):
        part=p[name];paths=[]
        for index,frames in enumerate(part['videos']):
            row=dict(frames=frames,sparse=part['sparse'][index],group=part['groups'][index])
            if 'labels' in part:row['labels']=part['labels'][index]
            path=tmp_path/f'{name}-{index}.jsonl'
            path.write_text(json.dumps(row)+'\n');paths.append(path)
        inputs[name]=FeatureShards(paths,labelled=name!='query')
    expected=reference_execute(prepare(p))
    functions=dict(pool_function=masked_attention_pool,gate_function=sigmoid_context_gate,
        mean_function=average,
        vocabulary_fit_function=fit_numeric_vocabulary,vocabulary_transform_function=transform_numeric_vocabulary,
        threshold_fit_function=fit_empirical_f1_thresholds,threshold_apply_function=apply_column_thresholds)
    bindings=[];observed=[];run=lifecycle.execute
    def checked(*args,**kwargs):
        assert kwargs==functions
        bindings.append(True)
        return run(*args,**kwargs)
    def capture(directory,node,name,value):
        if node=='execute' and name=='out_result':observed.append(value)
    monkeypatch.setattr(lifecycle,'execute',checked)
    monkeypatch.setattr(runner,'RUNS_DIR',tmp_path/'runs')
    monkeypatch.setattr(runner,'save_intermediate_value',capture)
    digest,nodes,edges=encode_execution_graph(build_decomposed_video_multilabel_graph())
    graph=decode_execution_graph(nodes,edges,digest)
    assert encode_execution_graph(graph)[0]==digest
    runner._ensure_atoms_imported()
    result=asyncio.run(runner.CDGExecutionSession(None,'synthetic-sharded-video','case').execute(inputs,cdg=graph))
    assert result['status']=='completed'
    traces={entry['node_id']:entry for entry in result['trace']}
    assert set(traces)=={'pool','gate','mean','vocabulary_fit','vocabulary_transform','threshold_fit','threshold_apply','execute'}
    assert all(not entry['cached'] for entry in traces.values())
    assert all(traces[name].get('callable_reference') is True
               for name in ('pool','gate','mean','vocabulary_fit','vocabulary_transform','threshold_fit','threshold_apply'))
    assert bindings==[True] and observed==[expected]
