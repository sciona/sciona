"""Uncached serialized native execution with tuple routing and query reuse."""
import asyncio
import warnings
import numpy as np
import pytest
from scipy import sparse
from sciona.sparse_projection_graph import build_sparse_projection_graph
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
from sciona.visualizer import runner
from sciona.atoms.ml.sklearn.decomposition import sparse_projection as provider


@pytest.mark.parametrize('rows,features,seed',[(9,19,42),(19,9,3),(23,23,12)])
def test_native_projection_and_frozen_query(tmp_path,monkeypatch,rows,features,seed):
    rng=np.random.default_rng(seed)
    a=rng.normal(size=(rows,features));a[rng.random(a.shape)<.6]=0
    matrix=sparse.csr_matrix(a);before=matrix.copy()
    query=sparse.csr_matrix(rng.normal(size=(4,features)))
    inputs=dict(matrix=matrix,rank=2,iterations=5,oversamples=10,seed=seed)
    graph=build_sparse_projection_graph()
    digest,nodes,edges=encode_execution_graph(graph)
    restored=decode_execution_graph(nodes,edges,digest)
    assert encode_execution_graph(restored)[0]==digest
    captured={}
    def capture(directory,node,name,value):captured[node,name]=value
    monkeypatch.setattr(runner,'RUNS_DIR',tmp_path)
    monkeypatch.setattr(runner,'save_intermediate_value',capture)
    runner._ensure_atoms_imported()
    with warnings.catch_warnings():
        warnings.simplefilter('error',RuntimeWarning)
        result=asyncio.run(runner.CDGExecutionSession(None,'synthetic-projection','fit').execute(inputs,cdg=restored))
        assert result['status']=='completed'
        assert {t['node_id'] for t in result['trace']}=={'fit','project'}
        assert all(not t['cached'] for t in result['trace'])
        components=captured['fit','out_components'];singular=captured['fit','out_singular_values']
        expected_components,expected_singular=provider.fit_sparse_projection(matrix,2,5,10,seed)
        np.testing.assert_array_equal(components,expected_components)
        np.testing.assert_array_equal(singular,expected_singular)
        assert captured['project','in_components'] is components
        np.testing.assert_array_equal(captured['project','out_projected'],provider.transform_sparse_projection(matrix,components))
        frozen=components.copy()
        # A separate serialized transform graph reuses factors without fitting query rows.
        query_graph=graph.model_copy(deep=True)
        query_graph.nodes=[n for n in query_graph.nodes if n.node_id=='project'];query_graph.edges=[]
        qhash,qnodes,qedges=encode_execution_graph(query_graph)
        query_graph=decode_execution_graph(qnodes,qedges,qhash)
        result=asyncio.run(runner.CDGExecutionSession(None,'synthetic-projection','query').execute(
            dict(matrix=query,components=components),cdg=query_graph))
        assert result['status']=='completed'
        assert len(result['trace'])==1 and not result['trace'][0]['cached']
        np.testing.assert_array_equal(captured['project','out_projected'],provider.transform_sparse_projection(query,frozen))
        np.testing.assert_array_equal(components,frozen)
    assert (matrix!=before).nnz==0
