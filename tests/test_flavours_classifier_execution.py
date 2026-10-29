"""Full-budget classifier CDG and explicit input-boundary validation."""
import asyncio
import json
import numpy as np
import pytest
from threadpoolctl import threadpool_limits
from sciona.flavours_classifier_execution import build_classifier_graph
from sciona.flavours_classifiers import train_classifiers
from sciona.atoms.ml.flavours_classifier_configuration import classifier_view_inputs
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.visualizer import runner


def fixture(width,seed):
    rng=np.random.default_rng(seed)
    # Enough rows to permit nontrivial splits under the source child-Hessian
    # threshold of 15; smaller fixtures can only produce constant trees.
    x=rng.normal(size=(243,width));q=rng.normal(size=(31,width))
    target=(x[:,0]+x[:,1]**2>0.8).astype(np.int64)
    def views(a):
        return dict(base=a,restricted=a[:,:-1],proxy=np.column_stack([a,a[:,0]*a[:,1]]),
                    corrected=np.column_stack([a,np.sin(a[:,0]),a[:,1]**2]))
    train,query=views(x),views(q)
    names={k:[f'feature_{i}' for i in range(v.shape[1])] for k,v in train.items()}
    return train,query,names,target


@pytest.mark.parametrize('width,seed',[(3,801),(5,918)])
def test_all_classifier_graph_fits_match_retained_reference(tmp_path,monkeypatch,width,seed):
    graph=build_classifier_graph();digest,nodes,edges=encode_execution_graph(graph)
    restored=_artifact_document_to_cdg(dict(cdg_nodes=[dict(n,version_id='check') for n in nodes],
        cdg_edges=[dict(e,version_id='check') for e in edges]),version_id='check',content_hash=digest,
        require_execution_envelope=True)
    assert restored==graph and len(graph.nodes)==116
    bound={(e.target_id,e.input_name) for e in graph.edges}
    assert len(bound)==len(graph.edges)
    assert {p.name for n in graph.nodes for p in n.inputs if (n.node_id,p.name) not in bound}=={
        'training_views','query_views','feature_names','targets'}
    train,query,names,targets=fixture(width,seed)
    monkeypatch.setattr(runner,'RUNS_DIR',tmp_path)
    with threadpool_limits(limits=1):
        expected=train_classifiers(train,targets,query)
        result=asyncio.run(runner.CDGExecutionSession(None,'synthetic-classifier-stage','check').execute(
            dict(training_views=train,query_views=query,feature_names=names,targets=targets),cdg=restored))
    assert result['status']=='completed' and len(result['trace'])==116
    assert all(not n['cached'] for n in result['trace'])
    for branch,rounds in enumerate((5,35,700,3,1500),1):
        name=f'xgb{branch}'
        np.testing.assert_array_equal(np.load(tmp_path/'check'/f'{name}_oof'/'out_predictions.npy'),expected['oof'][name])
        actual=np.load(tmp_path/'check'/f'{name}_full_prediction'/'out_predictions.npy')
        np.testing.assert_array_equal(actual,expected['query'][name])
        assert np.ptp(actual)>0
        for fold in list(range(5))+['full']:
            folder=tmp_path/'check'/f'{name}_{fold}_fit'
            assert json.loads((folder/'out_actual_rounds.json').read_text())['value']==rounds
            if fold!='full':
                fit_mask=np.load(folder/'in_selection.npy')
                held_mask=np.load(tmp_path/'check'/f'{name}_{fold}_contribution'/'in_selection.npy')
                assert np.all(fit_mask^held_mask)
            else:
                np.testing.assert_array_equal(np.load(folder/'in_targets.npy'),targets)
        # Full-refit query prediction must consume the full model, never a
        # fold-averaged ensemble masquerading as source-equivalent prediction.
        assert any(e.source_id==name+'_full_fit' and e.target_id==name+'_full_prediction'
                   and e.input_name=='state' for e in graph.edges)


@pytest.mark.parametrize('mutation',['missing_view','row_mismatch','width_mismatch','duplicate_names','infinite','float32'])
def test_classifier_boundary_rejects_invalid_views(mutation):
    train,query,names,_=fixture(3,801)
    if mutation=='missing_view':del train['base']
    elif mutation=='row_mismatch':query['base']=query['base'][:-1]
    elif mutation=='width_mismatch':query['base']=query['base'][:,:-1]
    elif mutation=='duplicate_names':names['base']=['x']*3
    elif mutation=='infinite':train['base'][0,0]=np.inf
    else:train['base']=train['base'].astype(np.float32)
    with pytest.raises(ValueError):classifier_view_inputs(train,query,names)
