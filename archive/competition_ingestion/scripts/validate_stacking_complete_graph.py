"""Full native decomposed Otto run, retaining aggregate synthetic evidence only."""
import asyncio
from decimal import Decimal, localcontext
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch
import uuid

import numpy as np
from threadpoolctl import threadpool_info

from sciona import stacking_meta_operations as operations
from sciona import otto_meta_tuning as tuning
from sciona.otto_final_blend import blend as retained_blend
from sciona.stacking_complete_graph import build_complete_graph
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
from sciona.visualizer import runner
from scripts.otto_synthetic import payload
from scripts.review_otto_candidate_runtime import review as runtime_review
from scripts.validate_stacking_first_graph import JAVA_FLAGS

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def decimal_blend(bags):
    with localcontext() as context:
        context.prec=80
        means=[]
        for bag in bags:
            means.append([[sum(Decimal.from_float(float(run[row,column])) for run in bag)/len(bag)
                           for column in range(9)] for row in range(9)])
        weight,extra_weight,left_exp,right_exp=map(Decimal.from_float,(.85,.15,.65,.35))
        raw=[];probabilities=[]
        for tree,neural,extra in zip(*means):
            scores=[weight*x**left_exp*y**right_exp+extra_weight*z for x,y,z in zip(tree,neural,extra)]
            raw.append([float(x) for x in scores])
            probabilities.append([float(x/sum(scores)) for x in scores])
        return np.asarray(raw),np.asarray(probabilities)


def main():
    if not __debug__:raise RuntimeError('Validation assertions required')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
        if os.environ.get(key)!='1':raise ValueError('Explicit one-thread native pools required')
    if os.environ.get('JAVA_TOOL_OPTIONS')!=JAVA_FLAGS:raise ValueError('Bounded JVM configuration required')
    runtime=runtime_review()
    paths=sorted(set(list((ROOT/'sciona').glob('otto_*.py'))+list((ROOT/'sciona').glob('otto_*.R'))+
        list((ROOT/'sciona').glob('stacking_*.py'))+[ROOT/'sciona/probability_ensemble.py',
        ROOT/'scripts/otto_synthetic.py',ROOT/'scripts/validate_stacking_first_graph.py',Path(__file__).resolve()]))
    before={str(p.relative_to(ROOT)):sha(p) for p in paths}
    import sciona.atoms.ml.stacking_first as first_provider
    import sciona.atoms.ml.stacking_supplemental as supplemental_provider
    import sciona.atoms.ml.stacking_meta as meta_provider
    import sciona.atoms.ml.stacking_complete as complete_provider
    provider_paths={module.__name__:Path(inspect.getfile(module)) for module in
                    (first_provider,supplemental_provider,meta_provider,complete_provider)}
    provider_hashes={name:sha(path) for name,path in provider_paths.items()}
    if any(p['num_threads']!=1 for p in threadpool_info()):raise ValueError('Loaded numerical pool exceeds allocation')
    digest,nodes,edges=encode_execution_graph(build_complete_graph())
    graph=decode_execution_graph(nodes,edges,digest)
    assert encode_execution_graph(graph)[0]==digest
    captured={};calls=[];active={'node':None}
    original=tuning._learner

    def instrumented(family):
        learner,runs=original(family)
        def fit(x,y,q,*,seed,controls):
            bag=learner(x,y,q,seed=seed,controls=controls)
            assert bag.shape==(runs,len(q),9)
            calls.append(dict(family=family,fitted_models=runs,fitting_rows=len(x),prediction_rows=len(q)))
            print(json.dumps(dict(completed_native_meta_bag=len(calls),**calls[-1])),flush=True)
            return bag
        return fit,runs

    def capture(directory,node,port,value):
        active['node']=node
        if port.startswith('out_'):
            captured[(node,port)]=value
            if node not in ('unpack','first_policy'):print('Completed full-stack node: '+node,flush=True)

    try:
        with tempfile.TemporaryDirectory(prefix='synthetic-stacking-complete-') as temporary:
            with patch.object(runner,'RUNS_DIR',Path(temporary)),patch.object(runner,'save_intermediate_value',side_effect=capture),\
                 patch.object(tuning,'_learner',side_effect=instrumented),patch.object(operations,'_learner',side_effect=instrumented):
                result=asyncio.run(runner.CDGExecutionSession(None,'synthetic-complete','native').execute(
                    {'payload':payload(r_library=str(ROOT/'.venv/otto-r-library'),libfm=str(ROOT/'.venv/otto-libfm/bin/libFM'))},cdg=graph))
        assert result['status']=='completed'
        assert {n for n,p in captured}=={n.node_id for n in graph.nodes}
        first=captured[('first_collect','out_first_level')]
        supplemental=captured[('supplemental_collect','out_supplemental')]
        layout=captured[('layout','out_layout')]
        final=captured[('meta_result','out_result')]
        assert set(first)==set(range(1,34)) and set(supplemental)==set(range(1,8))
        assert layout['first_level_width']==293
        assert layout['tree_training'].shape==(1350,341) and layout['neural_training'].shape==(1350,354)
        for family in operations.FAMILIES:
            observed=[(c['fitting_rows'],c['prediction_rows']) for c in calls if c['family']==family]
            assert observed==[(1012,338),(1012,338),(1013,337),(1013,337)]*2+[(1350,9)]
        assert sum(c['fitted_models'] for c in calls)==9900
        assert sum(s['tuning_model_fits'] for s in final['selection'].values())==8800
        assert sum(s['refit_models'] for s in final['selection'].values())==1100
        bags=[captured[('meta_refit_'+name,'out_bag')] for name in ('xgboost','neural','extratrees')]
        assert retained_blend(*bags)==final['blend']
        raw,probabilities=decimal_blend(bags)
        np.testing.assert_allclose(final['blend']['raw_scores'],raw,rtol=2e-13,atol=1e-15)
        np.testing.assert_allclose(final['blend']['probabilities'],probabilities,rtol=2e-13,atol=1e-15)
        np.testing.assert_array_equal(final['blend']['classes'],np.arange(9))
        json.dumps(final,allow_nan=False)
        assert before=={str(p.relative_to(ROOT)):sha(p) for p in paths}
        assert provider_hashes=={name:sha(path) for name,path in provider_paths.items()}
        assert runtime_review()==runtime
    except Exception as error:
        failure=dict(status='failed',approved=False,catalog_mutations=0,failed_node=active['node'],
                     error_type=type(error).__name__,completed_nodes=len({n for n,p in captured}),
                     completed_native_meta_bags=len(calls),serialized_graph_sha256=digest,
                     implementation_sha256=before,provider_sha256=provider_hashes)
        name='competition_otto_decomposed_complete_failure_'+uuid.uuid4().hex+'.json'
        (ROOT/'docs/reviews'/name).write_text(json.dumps(failure,indent=2)+'\n')
        raise
    report=dict(status='passed',approved=False,catalog_mutations=0,serialized_graph_sha256=digest,
        actual_runner_nodes=len(graph.nodes),actual_graph_edges=len(graph.edges),source_entries=33,supplemental_blocks=7,
        first_level_columns=293,tree_columns=341,neural_columns=354,synthetic_queries_correct=9,
        native_meta_tuning_fits=8800,native_meta_refits=1100,completed_native_meta_bags=calls,
        selection=final['selection'],retained_blend_exact_match=True,decimal_oracle_precision=80,
        decimal_oracle_rtol=2e-13,decimal_oracle_atol=1e-15,strict_json_result=True,
        implementation_sha256=before,provider_sha256=provider_hashes,
        candidate_runtime_evidence_sha256=runtime['evidence_sha256'],java_resource_flags=JAVA_FLAGS,
        limits=['Complete independent source realization on synthetic inputs only; no winning accuracy or historical code equivalence claim.',
                'Provisioned CPU in-process uncached execution; static shape proof, clean installation and persistent model serving unqualified.',
                'Original intake and new providers still require semantic, catalog corruption, rollback and served-selection gates before approval.'])
    (ROOT/'docs/reviews/competition_otto_decomposed_complete_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='passed',actual_runner_nodes=len(graph.nodes),source_entries=33,native_meta_fits=9900)),flush=True)


if __name__=='__main__':main()
