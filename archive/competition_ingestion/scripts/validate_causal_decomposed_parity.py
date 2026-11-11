"""Qualify explicit training/prediction against pinned upstream on synthetic pairs."""
import argparse
import asyncio
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import tempfile
import warnings

import numpy as np
from threadpoolctl import threadpool_limits
from sciona.causal_decomposed_prediction import build_explicit_causal_prediction_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.visualizer import runner
from scripts.validate_causal_feature_parity import load_reference,HASHES,COMMIT
from scripts.validate_causal_execution import reference_system,synthetic_pairs


async def validate(reference_dir):
    reference=load_reference(reference_dir)
    graph=build_explicit_causal_prediction_graph()
    digest,nodes,edges=encode_execution_graph(graph)
    restored=_artifact_document_to_cdg({'cdg_nodes':[{**n,'version_id':'validation'} for n in nodes],
        'cdg_edges':[{**e,'version_id':'validation'} for e in edges]},version_id='validation',content_hash=digest,require_execution_envelope=True)
    cases=[]
    with tempfile.TemporaryDirectory(prefix='sciona-causal-explicit-') as temp,threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.simplefilter('ignore')
        previous=runner.RUNS_DIR
        runner.RUNS_DIR=Path(temp)
        try:
            for seed,estimators in [(731,12),(732,12),(733,500)]:
                rng=np.random.default_rng(seed)
                train,train_types=synthetic_pairs(rng,18)
                query,query_types=synthetic_pairs(rng,9)
                labels=np.tile([-1,0,1],6)
                def matrix(pairs,types):
                    return np.stack([reference(a,ta,b,tb) for (x,y),(tx,ty) in zip(pairs,types)
                        for a,ta,b,tb in [(x,tx,y,ty),(y,ty,x,tx)]])
                model=reference_system(reference_dir,estimators)
                model.fit(matrix(train,train_types),np.column_stack((labels,-labels)).reshape(-1))
                expected=model.predict(matrix(query,query_types))
                run='case-'+str(seed)
                result=await runner.CDGExecutionSession(None,'synthetic-explicit-reference',run).execute({
                    'training_pairs':train,'prediction_pairs':query,'training_types':train_types,
                    'prediction_types':query_types,'pair_labels':labels,'weights':np.array([.383,.370,.247]),
                    'n_estimators':estimators},cdg=restored)
                if (result['status']!='completed' or len(result['trace'])!=len(graph.nodes) or
                        any(item['cached'] for item in result['trace']) or
                        {item['node_id'] for item in result['trace']}!={n.node_id for n in graph.nodes}):
                    raise ValueError('Complete uncached graph execution required')
                actual=np.load(Path(temp)/run/'ensemble'/'out_causal_scores.npy')
                np.testing.assert_allclose(actual,expected,rtol=1e-9,atol=1e-10)
                cases.append({'seed':seed,'estimators_per_fit':estimators,'uncached_nodes':len(result['trace']),
                    'maximum_absolute_error':float(np.max(np.abs(actual-expected)))})
        finally:runner.RUNS_DIR=previous
    import sys
    modules=[module for name,module in sorted(sys.modules.items()) if
        (name.startswith('sciona.atoms.causal_inference.') or name.startswith('sciona.atoms.ml.sklearn.ensemble.gradient_boosting.'))
        and getattr(module,'__file__',None)]
    paths=['sciona/causal_decomposed_prediction.py','sciona/causal_decomposed_training.py','sciona/causal_prediction_execution.py',
        'scripts/validate_causal_decomposed_parity.py','scripts/validate_causal_execution.py','scripts/validate_causal_feature_parity.py',
        'sciona/services/execution_graph_codec.py','sciona/visualizer/runner.py']
    root=Path(__file__).resolve().parents[1]
    return {'passed':True,'synthetic_only':True,'source_commit':COMMIT,'source_hashes':HASHES,
        'graph_hash':digest,'nodes':len(graph.nodes),'edges':len(graph.edges),'cases':cases,
        'implementation_hashes':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        'provider_hashes':{m.__name__:hashlib.sha256(Path(inspect.getfile(m)).read_bytes()).hexdigest() for m in modules},
        'library_versions':{n:importlib.metadata.version(n) for n in ['numpy','scipy','scikit-learn']},
        'port':['Python 2 collection materialization','selected-column shell accepts preselected reference features','deviance renamed log_loss'],
        'limitations':['Raw feature recipe remains inside one orchestration node','No catalog promotion performed',
            'Synthetic same-library parity; no historical-library reproduction, real-data accuracy or causal-validity claim'],
        'resource_policy':{'serial':True,'native_threads':1}}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=asyncio.run(validate(args.reference_dir))
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':report['passed'],'cases':report['cases']},indent=2))
