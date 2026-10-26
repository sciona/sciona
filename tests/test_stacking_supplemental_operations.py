"""Native synthetic parity and fold-isolation checks for supplemental stages."""
import numpy as np
import pytest

from sciona import stacking_supplemental_operations as current
from sciona import otto_supplemental as historical
from sciona.otto_tsne import PopulationEmbedding


def fixture():
    rng=np.random.default_rng(21)
    x=rng.integers(0,12,(90,13)).astype(float)
    q=rng.integers(0,12,(3,13)).astype(float)
    y=np.arange(90)%9;f=np.arange(90)%5
    ids=[f'synthetic-training-{i}' for i in range(90)]
    qids=[f'synthetic-query-{i}' for i in range(3)]
    embedding=PopulationEmbedding(tuple(ids+qids),rng.normal(size=(93,3)),0.)
    options=dict(embedding=embedding,seed=12,
        raw_metrics=['euclidean','cityblock','braycurtis'],tfidf_metrics=['euclidean','cityblock'],
        embedding_metrics=['euclidean','cityblock'],
        tfidf_controls=dict(smooth_idf=True,sublinear_tf=False,norm='l2'),
        cluster_controls=dict(kind='raw',standardize=False,ddof=1,cluster_counts=[2,3],n_init=2,max_iter=100))
    return (x,y,f,ids,q,qids),options


def test_all_native_blocks_match_retained_source():
    args,options=fixture()
    expected=historical.build(*args,**options)
    actual=current.build_decomposed(*args,**options)
    assert set(actual['supplemental'])==set(range(1,8))
    for old,new in [(expected['supplemental'][i],actual['supplemental'][i]) for i in range(1,8)]+[
                    (expected['raw_neural'],actual['raw_neural'])]:
        np.testing.assert_array_equal(old.training,new.training)
        np.testing.assert_array_equal(old.query,new.query)
        np.testing.assert_array_equal(old.folds,new.folds)
        assert old.training_ids==new.training_ids and old.query_ids==new.query_ids


def test_held_fold_labels_do_not_influence_its_features():
    args,options=fixture()
    original=current.build_decomposed(*args,**options)
    x,y,f,ids,q,qids=args
    changed=y.copy();held=f==0;changed[held]=(changed[held]+1)%9
    altered=current.build_decomposed(x,changed,f,ids,q,qids,**options)
    for entry in range(1,6):
        np.testing.assert_array_equal(original['supplemental'][entry].training[held],
                                      altered['supplemental'][entry].training[held])


def test_cluster_fit_uses_only_training_and_count_is_independently_computed(monkeypatch):
    args,options=fixture();seen=[];original=current.fit_clusters
    def capture(x,**kwargs):
        seen.append(x.copy())
        return original(x,**kwargs)
    monkeypatch.setattr(current,'fit_clusters',capture)
    result=current.build_decomposed(*args,**options)
    assert len(seen)==1
    np.testing.assert_array_equal(seen[0],args[0])
    for values,counts in ((args[0],result['supplemental'][7].training),(args[4],result['supplemental'][7].query)):
        assert counts[:,0].tolist()==[sum(value!=0 for value in row) for row in values]


@pytest.mark.parametrize('bad',['duplicate_metric','unsupported_embedding_metric','identity_overlap',
    'bad_fold','class_partition','seed','embedding_shape'])
def test_invalid_context_precedes_native_work(monkeypatch,bad):
    args,options=fixture()
    if bad=='duplicate_metric':options['raw_metrics']=['euclidean','euclidean']
    elif bad=='unsupported_embedding_metric':options['embedding_metrics']=['braycurtis']
    elif bad=='identity_overlap':args[5][0]=args[3][0]
    elif bad=='bad_fold':args[2][0]=5
    elif bad=='class_partition':args[2][args[1]==8]=0
    elif bad=='seed':options['seed']=True
    else:options['embedding']=PopulationEmbedding(tuple(args[3]+args[5]),np.zeros((93,2)),0.)
    def forbidden(*a,**k):raise AssertionError('Native computation started')
    monkeypatch.setattr(current,'raw_class_distances',forbidden)
    with pytest.raises(ValueError):current.build_decomposed(*args,**options)


def test_native_serialized_graph_matches_source_and_independent_distance_oracle(monkeypatch,tmp_path):
    import asyncio
    from sciona.stacking_supplemental_graph import build_supplemental_graph
    from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
    from sciona.visualizer import runner
    args,options=fixture()
    expected=historical.build(*args,**options)
    digest,nodes,edges=encode_execution_graph(build_supplemental_graph())
    graph=decode_execution_graph(nodes,edges,digest)
    assert encode_execution_graph(graph)[0]==digest
    assert len(graph.nodes)==9 and len(graph.edges)==16
    captured={}
    def capture(directory,node,port,value):
        if port.startswith('out_'):captured[(node,port)]=value
    monkeypatch.setattr(runner,'RUNS_DIR',tmp_path)
    monkeypatch.setattr(runner,'save_intermediate_value',capture)
    inputs=dict(zip(('training','labels','folds','training_ids','query','query_ids'),args))
    inputs.update(options)
    result=asyncio.run(runner.CDGExecutionSession(None,'synthetic-supplemental','native').execute(inputs,cdg=graph))
    assert result['status']=='completed'
    assert {n for n,p in captured}=={n.node_id for n in graph.nodes}
    blocks=captured[('collect','out_supplemental')]
    for entry in range(1,8):
        np.testing.assert_array_equal(blocks[entry].training,expected['supplemental'][entry].training)
        np.testing.assert_array_equal(blocks[entry].query,expected['supplemental'][entry].query)
    np.testing.assert_array_equal(captured[('collect','out_raw_neural')].training,args[0])
    x,y,f,ids,q,qids=args
    for row in (0,17):
        for label in range(9):
            reference=x[(f!=f[row])&(y==label)]
            distances=sorted(sum(float(a-b)**2 for a,b in zip(x[row],other))**.5 for other in reference)
            for entry,count in ((1,1),(2,2),(3,4)):
                assert blocks[entry].training[row,label]==pytest.approx(sum(distances[:count]),rel=1e-12,abs=1e-12)


@pytest.mark.parametrize('bad',['identities','folds','unsupervised_folds','shape','negative','distance_width'])
def test_collector_rejects_misaligned_blocks(bad):
    from dataclasses import replace
    args,options=fixture()
    result=current.build_decomposed(*args,**options)
    values=[result['supplemental'][i] for i in range(1,8)]+[result['raw_neural']]
    if bad=='identities':values[1]=replace(values[1],training_ids=values[1].training_ids[::-1])
    elif bad=='folds':values[1]=replace(values[1],folds=np.roll(values[1].folds,1))
    elif bad=='unsupervised_folds':values[5]=replace(values[5],folds=values[1].folds)
    elif bad=='shape':values[5]=replace(values[5],query=values[5].query[:-1])
    elif bad=='negative':values[1].training[0,0]=-1
    else:values[1]=replace(values[1],training=values[1].training[:,:10],query=values[1].query[:,:10])
    with pytest.raises(ValueError):current.collect_supplemental(*values)
