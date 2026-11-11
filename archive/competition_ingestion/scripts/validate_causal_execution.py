#!/usr/bin/env python3
"""Compare full CDG execution with pinned upstream training and prediction."""
import argparse
import ast
import asyncio
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import tempfile
import types
from uuid import uuid5
import warnings
import numpy as np
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from sciona.causal_prediction_execution import build_raw_pair_causal_training_prediction_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.visualizer import runner
from scripts.validate_causal_feature_parity import load_reference,Python2Collections,HASHES,COMMIT
from scripts.import_residual_execution_drafts import ensure_row
from sciona.physics_ingest.pdg_evidence import _digest


def reference_system(directory,estimators):
    # The reference receives the already selected feature matrix from its own
    # feature definitions. Replace only its DataFrame column-selection shell.
    class SelectedColumns:
        def __init__(self,features):self.features=features
        def fit(self,X,y=None):return self
        def fit_transform(self,X,y=None):return X
        def transform(self,X):return X
    f=types.SimpleNamespace(FeatureMapper=SelectedColumns,extract_features=lambda X,**kwargs:X)
    tree=ast.parse((directory/'estimator.py').read_text())
    tree.body=[n for n in tree.body if not (isinstance(n,ast.Import) and any(a.name=='features' for a in n.names))]
    tree=ast.fix_missing_locations(Python2Collections().visit(tree))
    ns={'f':f,'map':lambda fn,values:list(map(fn,values))};exec(compile(tree,'ported_pinned_estimator.py','exec'),ns)
    ns['gbc_params']['loss']='log_loss'
    ns['gbc_params']['n_estimators']=estimators
    return ns['CauseEffectSystemCombination'](weights=np.array([.383,.370,.247]),n_jobs=1)


def synthetic_pairs(rng,count):
    choices=['Numerical','Categorical','Binary'];pairs=[];types_out=[]
    for i in range(count):
        tx=choices[i%3];ty=choices[(i//3)%3];n=48+i%4*8
        x=rng.normal(size=n) if tx=='Numerical' else rng.integers(0,2 if tx=='Binary' else 4,size=n).astype(float)
        y=.25*x+rng.normal(size=n) if ty=='Numerical' else rng.integers(0,2 if ty=='Binary' else 3,size=n).astype(float)
        pairs.append((x,y));types_out.append((tx,ty))
    return pairs,types_out


async def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir',required=True,type=Path)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args();reference=load_reference(args.reference_dir)
    graph=build_raw_pair_causal_training_prediction_graph()
    digest,nodes,edges=encode_execution_graph(graph)
    restored=_artifact_document_to_cdg({'cdg_nodes':[{**n,'version_id':'validation'} for n in nodes],'cdg_edges':[{**e,'version_id':'validation'} for e in edges]},version_id='validation',content_hash=digest,require_execution_envelope=True)
    reports=[]
    with tempfile.TemporaryDirectory(prefix='sciona-causal-e2e-') as temp,warnings.catch_warnings():
        warnings.simplefilter('ignore')
        previous=runner.RUNS_DIR;runner.RUNS_DIR=Path(temp)
        try:
            for seed,estimators in [(731,12),(732,12),(733,500)]:
                rng=np.random.default_rng(seed)
                train,train_types=synthetic_pairs(rng,18);test,test_types=synthetic_pairs(rng,9)
                labels=np.tile([-1,0,1],6);augmented=np.column_stack((labels,-labels)).reshape(-1)
                def matrix(pairs,types_list):
                    return np.stack([reference(a,ta,b,tb) for (x,y),(tx,ty) in zip(pairs,types_list) for a,ta,b,tb in [(x,tx,y,ty),(y,ty,x,tx)]])
                expected_train=matrix(train,train_types);expected_test=matrix(test,test_types)
                model=reference_system(args.reference_dir,estimators)
                model.fit(expected_train,augmented)
                expected=model.predict(expected_test)
                run='case-'+str(seed)
                inputs={'training_pairs':train,'prediction_pairs':test,'training_types':train_types,'prediction_types':test_types,'pair_labels':labels,'weights':np.array([.383,.370,.247]),'n_estimators':estimators}
                result=await runner.CDGExecutionSession(None,'synthetic-causal-reference',run).execute(inputs,cdg=restored)
                if result['status']!='completed':raise ValueError('CDG execution failed')
                actual=np.load(Path(temp)/run/'ensemble'/'out_causal_scores.npy')
                np.testing.assert_allclose(actual,expected,rtol=1e-9,atol=1e-10)
                reports.append({'seed':seed,'n_estimators':estimators,'training_pairs':18,'prediction_pairs':9,'maximum_absolute_error':float(np.max(np.abs(actual-expected)))})
        finally:runner.RUNS_DIR=previous
    root=Path(__file__).resolve().parents[1]
    paths=['sciona/causal_prediction_execution.py','sciona/services/execution_graph_codec.py','sciona/visualizer/runner.py','scripts/validate_causal_execution.py','scripts/validate_causal_feature_parity.py']
    hashes={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}
    # Include transitive project-provider dependencies used by the graph.
    modules=[name for name in __import__('sys').modules if name.startswith('sciona.atoms.causal_inference.') or name.startswith('sciona.atoms.ml.sklearn.ensemble.gradient_boosting.')]
    provider_hashes={name:hashlib.sha256(Path(inspect.getfile(__import__('sys').modules[name])).read_bytes()).hexdigest() for name in sorted(modules) if getattr(__import__('sys').modules[name],'__file__',None)}
    details={'scope':'Synthetic full execution parity against pinned upstream feature definitions and estimator classes using installed scientific libraries; no real competition performance claim.','synthetic_only':True,'source_commit':COMMIT,'source_hashes':HASHES,'reference_port':['Python 2 collection materialization','selected-column shell accepts preselected reference features','deviance renamed log_loss'],'execution_graph':graph.model_dump(mode='json'),'execution_graph_sha256':digest,'implementation_hashes':hashes,'provider_hashes':provider_hashes,'library_versions':{n:importlib.metadata.version(n) for n in ['numpy','scipy','scikit-learn']},'cases':reports}
    with psycopg.connect(os.environ['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row) as db:
        records=db.execute("SELECT * FROM artifact_audit_evidence WHERE runner_version='competition-intake.v1' FOR SHARE").fetchall()
        records=[r for r in records if 'igci_asymmetry_score' in {s['stage_id'] for s in r['details']['snapshot']['template']['stages']}]
        if len(records)!=1:raise ValueError('unique causal source intake required')
        source=records[0];details['source_intake_sha256']=_digest(source['details']['snapshot'])
        evidence_id=uuid5(source['version_id'],'causal-full-execution.v1:'+_digest(details))
        created=ensure_row(db,'artifact_audit_evidence',{'evidence_id':evidence_id},{'evidence_id':evidence_id,'artifact_id':source['artifact_id'],'version_id':source['version_id'],'audit_type':'regression_test','passed':True,'status':'completed','source_kind':'automated','runner_version':'causal-full-execution.v1','details':Jsonb(details)})
        if not args.apply:db.rollback()
    print(json.dumps({'applied':args.apply,'evidence_created':created,'cases':reports},indent=2))


if __name__=='__main__':asyncio.run(main())
