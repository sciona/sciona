import hashlib
import json
import inspect
import pytest
from scripts.santander_synthetic import payload
from sciona.santander_execution import prepare,execute,Prepared
from sciona.santander_graph import build_santander_graph
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph


def test_detached_canonical_private_preparation():
    p=payload();prepared=prepare(p);p['training']['values'][0][0]=12345
    assert json.loads(prepared.configuration)['training']['values'][0][0]!=12345
    assert prepare(json.loads(prepared.configuration))==prepared
    assert 'configuration' not in repr(prepared)


@pytest.mark.parametrize('problem',['overlap','duplicate','query_labels','small_query','class_count','folds','epochs','seed','policy','nan','width'])
def test_invalid_payload(problem):
    p=payload();t=p['training'];q=p['query'];c=p['controls']
    if problem=='overlap':q['identities'][0]=t['identities'][0]
    elif problem=='duplicate':t['identities'][0]=t['identities'][1]
    elif problem=='query_labels':q['labels']=[]
    elif problem=='small_query':q['values']=q['values'][:7999];q['identities']=q['identities'][:7999]
    elif problem=='class_count':t['labels']=[0]*len(t['labels'])
    elif problem=='folds':c['tree']['folds']=2
    elif problem=='epochs':c['neural']['epochs']=1
    elif problem=='seed':c['tree']['seeds']=[True]
    elif problem=='policy':c['reference_policy']='unknown'
    elif problem=='nan':q['values'][0][0]=float('nan')
    else:q['values']=[r[:-1] for r in q['values']]
    with pytest.raises(ValueError):prepare(p)


@pytest.mark.parametrize('problem',['fingerprint','noncanonical','invalid_json'])
def test_modified_preparation_rejected_before_fitting(problem,monkeypatch):
    from sciona import santander_lifecycle as lifecycle
    def forbidden(*args,**kwargs):raise AssertionError('Training started')
    monkeypatch.setattr(lifecycle,'fit',forbidden)
    original=prepare(payload());text=original.configuration
    if problem=='noncanonical':text=json.dumps(json.loads(text),indent=2)
    elif problem=='invalid_json':text='invalid'
    fingerprint='bad' if problem=='fingerprint' else hashlib.sha256(text.encode()).hexdigest()
    with pytest.raises(ValueError):execute(Prepared(text,fingerprint))


def test_graph_roundtrip_and_provider_port_witnesses():
    import sciona.atoms.ml.santander_execution as provider
    digest,nodes,edges=encode_execution_graph(build_santander_graph());graph=decode_execution_graph(nodes,edges,digest)
    assert encode_execution_graph(graph)[0]==digest
    witness={}
    for node in graph.nodes:
        assert list(inspect.signature(getattr(provider,'santander_'+node.node_id)).parameters)==[p.name for p in node.inputs]
        witness=getattr(provider,'witness_santander_'+node.node_id)(witness)
    assert witness=={'kind':'Santander.Result'}
    with pytest.raises(ValueError):provider.witness_santander_execute({})
