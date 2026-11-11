"""Execute all five served quantile stages against the frozen synthetic lifecycle."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
            'NUMEXPR_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
os.environ.setdefault('TF_USE_LEGACY_KERAS','1')
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
import importlib
import inspect
import json
import numpy as np
import subprocess
from unittest.mock import patch
from sciona import osic_quantile_training as training
from scripts.validate_osic_quantile_features import synthetic_tables
from scripts import promote_osic_quantile_providers as publisher


def verify():
    plan=publisher.qualify_plan()
    bindings=[]
    with publisher.connect() as db:
        db.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        publisher.check(db,plan,published=True)
        for record in plan['records']:
            module,symbol=record['fqdn'].rsplit('.',1)
            expected=dict(fqdn=record['fqdn'],import_module=module,source_symbol=symbol,
                version_id=record['version_id'],content_hash=record['content_hash'],trust_tier=3)
            for prefix,key in [('atom','atom_id'),('artifact','artifact_id')]:
                rows=db.execute(f'''SELECT s.fqdn,p.import_module,p.source_symbol,v.version_id::text,v.content_hash,v.trust_tier
                    FROM catalog_{prefix}s_served s JOIN {prefix}_versions v USING({key})
                    JOIN atoms p ON p.atom_id=s.{key}
                    WHERE s.{key}=%s AND v.is_latest''',(record['artifact_id'],)).fetchall()
                if not rows or any(row!=expected for row in rows):
                    raise ValueError('Exact served binding required in both catalogs')
            bindings.append(expected)
        original=db.execute('SELECT status,is_publishable FROM artifacts WHERE fqdn=%s',
            ('cdg.competition.solution.kaggle.osic_pulmonary_1st',)).fetchone()
        if original!=dict(status='draft',is_publishable=False):
            raise ValueError('Original workflow state differs from pending scope')
    functions={}
    for binding in bindings:
        module=importlib.import_module(binding['import_module'])
        if publisher.sha(inspect.getfile(module))!=plan['qualification']['software_sha256']['provider']:
            raise ValueError('Served implementation differs')
        functions[binding['source_symbol']]=getattr(module,binding['source_symbol'])
    previous=subprocess.check_output(['git','show','760fc106:sciona/osic_quantile_training.py'],cwd=publisher.ROOT)
    namespace={}
    exec(compile(previous,'<frozen-quantile-lifecycle>','exec'),namespace)
    train,baseline,query=synthetic_tables()
    coordinates=dict(feature_coordinate='synthetic-features.v1',response_coordinate='synthetic-response.v1')
    expected=namespace['train_quantile_branch'](train,baseline,query,
        **coordinates,seeds=(910,911),allow_transductive=True)
    prepared=functions['fit_joint_quantile_features'](
        functions['assemble_quantile_populations'](train,baseline,query),True)
    x=prepared['populations']['training']['features'].astype(np.float32)
    y=prepared['populations']['training']['targets'].astype(np.float32).reshape(-1,1)
    q=prepared['populations']['query']['features'].astype(np.float32)
    columns=[json.dumps(column,separators=(',', ':')) for column in prepared['feature_columns']]
    assert columns==expected['feature_columns']
    def forbidden(*args,**kwargs):
        raise AssertionError('Served numeric stage crossed its preparation or training boundary')
    with patch.object(training,'prepare_features',forbidden):
        actual=functions['train_prepared_quantile_folds'](x,y,q,columns,**coordinates,seeds=(910,911))
    assert actual['fold_states']==expected['fold_states']
    np.testing.assert_array_equal(actual['oof_predictions'],expected['oof_predictions'])
    np.testing.assert_array_equal(q,expected['query_features'])
    with patch.object(training,'prepare_features',forbidden), \
         patch.object(training,'train_quantile_branch',forbidden), \
         patch.object(training,'train_prepared_quantile_folds',forbidden):
        replay=functions['replay_prepared_quantile_folds'](actual['fold_states'],q,columns,**coordinates)
    np.testing.assert_array_equal(replay,actual['fold_predictions'])
    predictions=functions['reduce_quantile_predictions'](replay,expected['observed_mask'],expected['observed_values'])
    np.testing.assert_array_equal(predictions,expected['predictions'])
    checks=dict(frozen_lifecycle_commit='760fc106',full_schedule_fits=4,epochs_per_fit=600,
        exact_saved_states=True,exact_out_of_fold_predictions=True,exact_query_features=True,
        exact_raw_fold_replay=True,replay_fitting_forbidden=True,exact_reduced_predictions=True,
        synthetic_only=True,executed_served_providers=len(functions))
    with publisher.connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        publisher.check(db,plan,published=True)
    for name,digest in plan['qualification']['software_sha256'].items():
        path=inspect.getfile(module) if name=='provider' else publisher.ROOT/name
        if publisher.sha(path)!=digest:raise ValueError('Implementation changed during served execution')
    return dict(passed=True,read_only=True,catalog_mutations=0,approved=True,trust_tier=3,
        served_catalogs=2,bindings=bindings,fresh_resolved_execution=checks,
        original_workflow_approved=False,history_and_publication_evidence_preserved=True,
        verifier_sha256=publisher.sha(__file__),plan_sha256=publisher.sha(publisher.PLAN),
        implementation_sha256=plan['qualification']['software_sha256'],
        limitations=['Five source-specific served Python stages in the provisioned runtime; no HTTP transport, semantic ranking or whole-CDG execution qualification.',
                    'Synthetic numerical checks only. The complete original OSIC workflow remains draft.'])


if __name__=='__main__':
    result=verify()
    (publisher.ROOT/'docs/reviews/osic_quantile_served_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,providers=len(result['bindings']),full_schedule_fits=result['fresh_resolved_execution']['full_schedule_fits'])))
