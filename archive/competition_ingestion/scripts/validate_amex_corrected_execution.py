"""Execute the stored corrected original graph with full reference comparison."""
import asyncio
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import torch
import lightgbm as lgb
from scripts.amex_synthetic import payload
from sciona.amex_pipeline import blend
from sciona.amex_execution import prepare, execute
from scripts import stage_amex_corrected_intake as intake
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
from sciona.visualizer import runner
import sciona.atoms.ml.decomposed_amex_execution as provider
import sciona.atoms.numerical.statistics.ordered_sum as numerical_provider

ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    parity_path=ROOT/'docs/reviews/decomposed_amex_pipeline.json'
    parity=json.loads(parity_path.read_text())
    assert parity['passed'] and parity['exact_branch_probabilities'] and parity['exact_final_output']
    assert parity['complete_reference_models']==parity['complete_candidate_models']==25
    for name,digest in parity['implementation_sha256'].items():assert sha(ROOT/name)==digest
    assert sha(Path(inspect.getfile(numerical_provider)))==parity['generic_provider_sha256']
    plan,providers=intake.qualified_plan()
    with intake.connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        graph=intake.check_candidate(db,plan,providers)
    digest,nodes,edges=encode_execution_graph(graph)
    assert digest==plan['graph_sha256']
    graph=decode_execution_graph(nodes,edges,digest)
    assert encode_execution_graph(graph)[0]==digest
    paths=sorted((ROOT/'sciona').glob('amex_*.py'))+[
        ROOT/'sciona/decomposed_amex_pipeline.py',ROOT/'sciona/decomposed_amex_graph.py',
        ROOT/'sciona/visualizer/runner.py',ROOT/'sciona/services/execution_graph_codec.py',
        ROOT/'scripts/amex_synthetic.py',ROOT/'scripts/stage_amex_corrected_intake.py',
        ROOT/'scripts/amex_approved_provider_state.py',intake.PLAN,
        intake.provider_publication.PLAN,Path(__file__).resolve(),parity_path]
    before={str(p.relative_to(ROOT)):sha(p) for p in paths}
    provider_hash=sha(Path(inspect.getfile(provider)));generic_hash=sha(Path(inspect.getfile(numerical_provider)))
    runner._ensure_atoms_imported();captured={};p=payload();original=json.dumps(p,sort_keys=True)
    rng=torch.get_rng_state().clone()
    original_train=lgb.train
    training_records={'reference':[],'graph':[]}
    phase='reference'
    def observe_train(*args,**kwargs):
        assert kwargs['num_boost_round']==4500
        iterations=[]
        def observe_iteration(environment):iterations.append(environment.iteration)
        kwargs['callbacks']=[*(kwargs.get('callbacks') or []),observe_iteration]
        model=original_train(*args,**kwargs)
        assert iterations==list(range(4500))
        training_records[phase].append(dict(evaluated_rounds=len(iterations),retained_iterations=model.current_iteration()))
        print(json.dumps(dict(phase=phase,completed_dart_models=len(training_records[phase]),**training_records[phase][-1])),flush=True)
        return model
    print('Checking unchanged full reference with observed training rounds',flush=True)
    with patch.object(lgb,'train',side_effect=observe_train):expected=execute(prepare(p))
    phase='graph'
    def capture(directory,node,name,value):
        if name.startswith('out_'):captured[node,name]=value
    print('Starting full serialized ten-node Amex graph',flush=True)
    with tempfile.TemporaryDirectory(prefix='amex-synthetic-decomposed-') as directory:
        with patch.object(runner,'RUNS_DIR',Path(directory)),patch.object(runner,'save_intermediate_value',side_effect=capture),patch.object(lgb,'train',side_effect=observe_train):
            status=asyncio.run(runner.CDGExecutionSession(None,'synthetic-amex','full').execute({'payload':p},cdg=graph))
    assert status['status']=='completed'
    trace=status['trace'];assert len(trace)==10 and all(not t['cached'] for t in trace)
    assert {t['node_id'] for t in trace}=={n.node_id for n in graph.nodes}
    row=captured['row_models','out_row_result'];trees=captured['tree_models','out_tree_result'];neural=captured['neural_models','out_neural_result']
    assert len(training_records['reference'])==len(training_records['graph'])==15
    assert training_records['graph']==training_records['reference']
    retained=[r['retained_iterations'] for r in training_records['graph']]
    assert row['models']==5 and row['configured_rounds']==4500 and row['iterations']==retained[:5]
    assert trees['models']==10 and trees['configured_rounds']==4500
    for i,name in enumerate(('manual','manual_with_slots')):
        assert trees['variants'][name]['models']==5 and trees['variants'][name]['iterations']==retained[5+5*i:10+5*i]
    assert neural['models']==10 and neural['epochs']==10 and neural['hidden_width']==128
    for name in ('series','combined'):
        variant=neural['variants'][name];assert variant['models']==5 and len(variant['folds'])==5
        for fold in variant['folds']:
            assert len(fold['history'])==10 and fold['best_metric']>0
            assert [h['learning_rate'] for h in fold['history']]==[.001]*5+[.0001]*4+[.00001]
            assert all(h['training_rows']==1280 for h in fold['history'])
    values=captured['combination_inputs','out_values'];weights=captured['combination_inputs','out_coefficients']
    expected_branches=np.stack([trees['variants']['manual']['query_predictions'],trees['variants']['manual_with_slots']['query_predictions'],
        neural['variants']['series']['query_predictions'],neural['variants']['combined']['query_predictions']])
    np.testing.assert_array_equal(values,expected_branches);np.testing.assert_array_equal(weights,[.3,.35,.15,.1])
    np.testing.assert_array_equal(captured['combine','out_scores'],blend(*values))
    output=json.loads(json.dumps(captured['result','out_result'],allow_nan=False))
    np.testing.assert_array_equal(output['scores'],blend(*values))
    assert output['models']==25 and output['model_counts']==dict(row=5,downstream_tree=10,neural=10)
    assert output==expected
    assert original==json.dumps(p,sort_keys=True) and torch.equal(rng,torch.get_rng_state())
    assert before=={str(path.relative_to(ROOT)):sha(path) for path in paths}
    assert sha(Path(inspect.getfile(provider)))==provider_hash and sha(Path(inspect.getfile(numerical_provider)))==generic_hash
    report=dict(format='amex-corrected-execution-validation.v1',passed=True,approved=False,catalog_mutations=0,
        stored_candidate=True,candidate_version_id=plan['version_id'],original_artifact_id=plan['artifact_id'],
        plan_sha256=intake.sha(intake.PLAN),actual_runner_nodes=10,edges=16,all_nodes_uncached=True,models=25,dart_models=15,
        rounds_per_dart_model=4500,neural_models=10,epochs_per_neural_model=10,
        exact_reference_output=True,training_records=training_records,
        exact_source_blend=True,strict_json=True,input_and_torch_rng_preserved=True,
        serialized_graph_sha256=digest,implementation_sha256=before,
        provider_sha256=provider_hash,generic_provider_sha256=generic_hash,
        limits='Synthetic complete CPU execution; no historical accuracy or GPU equivalence. Evaluated boosting rounds are observed separately from retained tree iterations, which can be lower when the source learner cannot split. All ten providers are approved and served at Tier 3; original CDG publication and served selection remain pending.')
    (ROOT/'docs/reviews/amex_corrected_intake_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='implementation_sha256'}),flush=True)


if __name__=='__main__':main()
