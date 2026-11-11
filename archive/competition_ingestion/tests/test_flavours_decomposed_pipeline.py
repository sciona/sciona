"""Full-budget integrated training/prediction graph on synthetic physical inputs."""
import asyncio
import json
import numpy as np
from threadpoolctl import threadpool_limits
from sciona.flavours_pipeline import execute
from sciona.flavours_decomposed_pipeline import build_pipeline_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.visualizer import runner


def synthetic_inputs():
    rng=np.random.default_rng(624)
    def physical(rows):
        base=rng.normal(size=(rows,46));p=rng.uniform(1,8,(rows,3));pt=p*rng.uniform(.1,.8,(rows,3))
        return [base,p,pt,rng.uniform(.2,3,rows),rng.uniform(1,5,rows),rng.uniform(.1,2,rows)]
    train=physical(600);query=physical(17)
    labels=np.tile(np.array([0,1],dtype=np.int64),300);rng.shuffle(labels)
    train[0][:,0]+=labels*.8
    mass=3+train[0][:,1]**2+.5*train[0][:,2]
    return dict(training_inputs=train,labels=labels,mass=mass,query_inputs=query,excluded_column=17)


def test_integrated_all_115_fits_match_retained_pipeline(tmp_path,monkeypatch):
    graph=build_pipeline_graph();digest,nodes,edges=encode_execution_graph(graph)
    restored=_artifact_document_to_cdg(dict(cdg_nodes=[dict(n,version_id='check') for n in nodes],
        cdg_edges=[dict(e,version_id='check') for e in edges]),version_id='check',content_hash=digest,require_execution_envelope=True)
    assert restored==graph and len(nodes)==834
    bound={(e.target_id,e.input_name) for e in graph.edges}
    assert len(bound)==len(edges)
    roots={(n.node_id,p.name) for n in graph.nodes for p in n.inputs if (n.node_id,p.name) not in bound}
    assert roots=={('inputs',key) for key in ('training_inputs','labels','mass','query_inputs','excluded_column')}
    inputs=synthetic_inputs();monkeypatch.setattr(runner,'RUNS_DIR',tmp_path)
    with threadpool_limits(limits=1):
        expected=execute(**inputs)
        result=asyncio.run(runner.CDGExecutionSession(None,'synthetic-integrated-pipeline','check').execute(inputs,cdg=restored))
    assert result['status']=='completed' and len(result['trace'])==834
    assert all(not item['cached'] for item in result['trace'])
    for population in ('oof','query'):
        actual=np.load(tmp_path/'check'/f'{population}_blend'/'out_scores.npy')
        np.testing.assert_array_equal(actual,expected[population+'_scores'])
        assert np.ptp(actual)>0
    for output,key in [('held_mean','oof_mass'),('query_mean','query_mass')]:
        np.testing.assert_array_equal(np.load(tmp_path/'check'/('mass_'+output)/'out_predictions.npy'),expected['mass_correction'][key])
    for record in expected['classifiers']['fits']:
        value=json.loads((tmp_path/'check'/f"trees_{record['branch']}_{record['fold']}_fit"/'out_actual_rounds.json').read_text())['value']
        assert value==record['rounds']
    for record in expected['neural']['fits']:
        prefix=f"neural_f{record['fold']}_s{record['seed']}"
        for output,key in [('epochs','epochs'),('updates','optimizer_updates')]:
            assert json.loads((tmp_path/'check'/(prefix+'_fit')/f'out_{output}.json').read_text())['value']==record[key]
        assert json.loads((tmp_path/'check'/(prefix+'_validation')/'out_loss.json').read_text())['value']==record['validation_loss']
    for repeat in range(7):
        for fold in range(5):
            assert json.loads((tmp_path/'check'/f'mass_r{repeat}_f{fold}_fit'/'out_actual_rounds.json').read_text())['value']==2800
