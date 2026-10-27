"""Complete graph wiring; learner stand-ins do not claim native model fits."""
import asyncio
import json

import numpy as np
import pytest

from scripts.otto_synthetic import payload
from sciona import stacking_first_operations as first
from sciona import stacking_meta_operations as meta
from sciona import otto_tsne
from sciona.stacking_complete_graph import build_complete_graph
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
from sciona.visualizer import runner


def data():
    return payload(r_library='synthetic-runtime-r',libfm='synthetic-runtime-fm')


def test_complete_graph_has_one_input_and_separate_fold_routes():
    graph=build_complete_graph()
    assert len(graph.nodes)==62
    digest,nodes,edges=encode_execution_graph(graph)
    restored=decode_execution_graph(nodes,edges,digest)
    assert encode_execution_graph(restored)[0]==digest
    fed={(e.target_id,e.input_name) for e in restored.edges}
    assert len(fed)==len(restored.edges)
    assert {(n.node_id,p.name) for n in restored.nodes for p in n.inputs if (n.node_id,p.name) not in fed}=={('unpack','payload')}
    routing={(e.target_id,e.input_name):(e.source_id,e.output_name) for e in restored.edges}
    assert routing[('first_prepare','folds')]==('unpack','first_folds')
    assert routing[('supplemental_prepare','folds')]==('unpack','first_folds')
    assert routing[('layout','folds')]==('unpack','first_folds')
    assert routing[('meta_prepare','folds')]==('unpack','meta_folds')


def test_complete_runner_all_nodes_and_feature_views(monkeypatch,tmp_path):
    values=data();calls=[];captured={}
    def embedding(x,ids,**kwargs):
        assert len(x)==1359 and ids==values['training']['identities']+values['query']['identities']
        calls.append('embedding')
        return otto_tsne.PopulationEmbedding(tuple(ids),np.zeros((len(ids),3)),0.)
    def entry(inputs,index):
        calls.append(('entry',index))
        x,y,f,ids,q,qids=inputs.args
        if index==10:return first._block(inputs,index,np.zeros((len(x),5)),np.zeros((len(q),5)))
        return first._block(inputs,index,np.eye(9)[y],np.eye(9))
    def bank(inputs,representative):
        calls.append('bank')
        x,y,f,ids,q,qids=inputs.args
        return first.NeighborBank(inputs,json.dumps(inputs.controls[representative],sort_keys=True),
            dict(oof=np.repeat(np.eye(9)[y][:,None,:],10,axis=1),query=np.repeat(np.eye(9)[:,None,:],10,axis=1)))
    def tune(x,y,f,ids,*,family,seed,candidates):
        assert x.shape==(1350,354 if family=='lasagne_neural' else 341)
        np.testing.assert_array_equal(y,values['training']['labels'])
        np.testing.assert_array_equal(f,values['training']['meta_folds'])
        assert ids==values['training']['identities']
        assert candidates==values['controls']['meta_candidates'][family] and seed==12
        calls.append(('select',family))
        return dict(selected_index=1,log_losses=[1.,.1],models_per_candidate=4*meta.RUNS[family])
    def learner(family):
        def fit(x,y,q,*,seed,controls):
            assert x.shape==(1350,354 if family=='lasagne_neural' else 341)
            assert q.shape==(9,x.shape[1])
            assert controls==values['controls']['meta_candidates'][family][1]
            calls.append(('refit',family))
            return np.repeat(np.eye(9)[None,:,:],meta.RUNS[family],axis=0)
        return fit,meta.RUNS[family]
    monkeypatch.setattr(otto_tsne,'fit_population',embedding)
    monkeypatch.setattr(first,'fit_entry',entry)
    monkeypatch.setattr(first,'fit_neighbor_bank',bank)
    monkeypatch.setattr(meta,'tune',tune)
    monkeypatch.setattr(meta,'_learner',learner)
    def capture(directory,node,port,value):
        if port.startswith('out_'):captured[(node,port)]=value
    monkeypatch.setattr(runner,'RUNS_DIR',tmp_path)
    monkeypatch.setattr(runner,'save_intermediate_value',capture)
    digest,nodes,edges=encode_execution_graph(build_complete_graph())
    graph=decode_execution_graph(nodes,edges,digest)
    result=asyncio.run(runner.CDGExecutionSession(None,'synthetic-complete','case').execute({'payload':values},cdg=graph))
    assert result['status']=='completed'
    assert {n for n,p in captured}=={n.node_id for n in graph.nodes}
    assert sorted(c[1] for c in calls if isinstance(c,tuple) and c[0]=='entry')==list(range(1,24))
    assert calls.count('bank')==calls.count('embedding')==1
    final=captured[('meta_result','out_result')]
    assert final['blend']['classes']==list(range(9))
    assert sum(s['tuning_model_fits'] for s in final['selection'].values())==8800
    assert sum(s['refit_models'] for s in final['selection'].values())==1100
    json.dumps(final,allow_nan=False)


def test_neighbor_group_guard_and_alternate_compilation():
    from sciona.atoms.ml.stacking_complete import unpack_payload,validate_neighbor_sharing,UNPACK_PORTS
    values=data();values['controls']['first_level']['25']['metric']='cityblock'
    fields=dict(zip((n for n,t in UNPACK_PORTS),unpack_payload(values)))
    with pytest.raises(ValueError,match='compiled sharing'):
        validate_neighbor_sharing(fields['first_controls'],*[24]*10)
    representatives=[25 if i==25 else 24 for i in range(24,34)]
    assert validate_neighbor_sharing(fields['first_controls'],*representatives)==fields['first_controls']
    graph=build_complete_graph([[i for i in range(24,34) if i!=25],[25]])
    assert len(graph.nodes)==63


def test_boundary_detaches_values_and_rejects_query_labels():
    from sciona.atoms.ml.stacking_complete import unpack_payload,UNPACK_PORTS
    values=data()
    fields=dict(zip((n for n,t in UNPACK_PORTS),unpack_payload(values)))
    original=fields['training'][0][0]
    values['training']['values'][0][0]=99999
    assert fields['training'][0][0]==original
    values['query']['labels']=[0]*9
    with pytest.raises(ValueError):unpack_payload(values)
