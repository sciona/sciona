import hashlib
import json
import numpy as np
import pytest
from scripts.otto_synthetic import payload
from sciona.otto_execution import Prepared,prepare,execute
from sciona.otto_graph import build_otto_graph
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph


def fixture():return payload(r_library='synthetic-runtime-r',libfm='synthetic-runtime-fm')


def test_prepared_input_is_detached_canonical_and_has_private_repr():
    p=fixture();prepared=prepare(p)
    p['training']['values'][0][0]=999
    assert json.loads(prepared.configuration)['training']['values'][0][0]!=999
    assert prepared==prepare(json.loads(prepared.configuration))
    assert 'configuration' not in repr(prepared) and 'synthetic-runtime' not in repr(prepared)


@pytest.mark.parametrize('problem',['query_labels','overlap','duplicate','first_folds','meta_folds','class_partition','references','nan','seed','missing_entry','empty_candidates','embedding','width'])
def test_invalid_input_rejected(problem):
    p=fixture();t=p['training'];q=p['query'];c=p['controls']
    if problem=='query_labels':q['labels']=list(range(9))
    elif problem=='overlap':q['identities'][0]=t['identities'][0]
    elif problem=='duplicate':t['identities'][1]=t['identities'][0]
    elif problem=='first_folds':t['first_folds'][0]=True
    elif problem=='meta_folds':t['meta_folds'][0]=4
    elif problem=='class_partition':t['meta_folds']=[0 if y==8 else f for y,f in zip(t['labels'],t['meta_folds'])]
    elif problem=='references':
        for key in t:t[key]=t[key][:1000]
    elif problem=='nan':q['values'][0][0]=np.nan
    elif problem=='seed':c['seed']=True
    elif problem=='missing_entry':del c['first_level']['33']
    elif problem=='empty_candidates':c['meta_candidates']['lasagne_neural']=[]
    elif problem=='embedding':c['embedding']['perplexity']=10000
    else:q['values']=[r[:-1] for r in q['values']]
    with pytest.raises(ValueError):prepare(p)


@pytest.mark.parametrize('problem',['fingerprint','noncanonical','invalid_json','forged_partition'])
def test_changed_prepared_input_cannot_execute(problem,monkeypatch):
    import sciona.otto_tsne as tsne
    def forbidden(*args,**kwargs):raise AssertionError('Fitting started')
    monkeypatch.setattr(tsne,'fit_population',forbidden)
    p=prepare(fixture());encoded=p.configuration
    if problem=='fingerprint':changed=Prepared(encoded,'bad')
    else:
        if problem=='noncanonical':encoded=json.dumps(json.loads(encoded),indent=2)
        elif problem=='invalid_json':encoded='invalid'
        else:
            data=json.loads(encoded);data['training']['meta_folds'][0]=4
            encoded=json.dumps(data,sort_keys=True,separators=(',',':'))
        changed=Prepared(encoded,hashlib.sha256(encoded.encode()).hexdigest())
    with pytest.raises(ValueError):execute(changed)


def test_graph_codec_and_provider_port_witnesses():
    import inspect
    import sciona.atoms.ml.otto_execution as provider
    graph=build_otto_graph();digest,nodes,edges=encode_execution_graph(graph)
    restored=decode_execution_graph(nodes,edges,digest)
    assert encode_execution_graph(restored)[0]==digest
    witness={}
    for node in restored.nodes:
        function=getattr(provider,'otto_'+node.node_id)
        assert list(inspect.signature(function).parameters)==[p.name for p in node.inputs]
        witness=getattr(provider,'witness_otto_'+node.node_id)(witness)
    assert witness=={'kind':'Otto.Result'}
    with pytest.raises(ValueError):provider.witness_otto_execute({})


def test_json_interaction_triples_reach_real_transform_without_changing_values():
    from sciona.otto_execution import _first_controls
    from sciona.otto_interactions import InteractionSelection
    data=json.loads(prepare(fixture()).configuration)
    controls=_first_controls(data['controls']['first_level'])
    x=np.arange(1,27,dtype=float).reshape(2,13)
    selector=InteractionSelection(13,tuple(range(13)))
    for entry in (12,13):
        actual=selector.transform(x,triples=controls[entry]['triples'])
        np.testing.assert_array_equal(actual,np.column_stack((x[:,0]*x[:,1]*x[:,2],x[:,3]*x[:,4]*x[:,5])))
        assert all(type(t) is list for t in data['controls']['first_level'][str(entry)]['triples'])


@pytest.mark.parametrize('triples',[[[0,0,1]],[[0,1,True]],[[0,1,13]],[[0,1]],[[0,1,2],[2,0,1]],[]])
def test_invalid_serialized_triples_rejected_before_fitting(triples):
    p=fixture();p['controls']['first_level']['12']['triples']=triples
    with pytest.raises(ValueError):prepare(p)
