"""Check numeric training decomposition against the frozen preceding lifecycle."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
            'NUMEXPR_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('TF_USE_LEGACY_KERAS','1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
import hashlib
import json
from pathlib import Path
import subprocess
from unittest.mock import patch
import numpy as np
from sciona import osic_quantile_training as candidate
from scripts.validate_osic_quantile_features import synthetic_tables

ROOT=Path(__file__).resolve().parents[1]
PREVIOUS='760fc106'


def validate():
    path=ROOT/'sciona/osic_quantile_training.py'
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    previous=subprocess.check_output(['git','show',PREVIOUS+':sciona/osic_quantile_training.py'],cwd=ROOT)
    namespace={}
    exec(compile(previous,'<frozen-quantile-lifecycle>','exec'),namespace)
    train,baseline,query=synthetic_tables()
    kwargs=dict(feature_coordinate='synthetic-features.v1',response_coordinate='synthetic-response.v1',
                seeds=(910,911),allow_transductive=True)
    expected=namespace['train_quantile_branch'](train,baseline,query,**kwargs)
    actual=candidate.train_quantile_branch(train,baseline,query,**kwargs)
    for key in ('predictions','oof_predictions','query_features','observed_mask','observed_values'):
        np.testing.assert_array_equal(actual[key],expected[key])
    for key in ('fold_states','folds','feature_columns'):
        assert actual[key]==expected[key]
    prepared=actual['preparation']
    x=prepared['populations']['training']['features'].astype(np.float32)
    y=prepared['populations']['training']['targets'].astype(np.float32).reshape(-1,1)
    q=actual['query_features']
    columns=actual['feature_columns']
    def forbidden(*args,**kwargs):
        raise AssertionError('Numeric fold boundary must not prepare domain features')
    with patch.object(candidate,'prepare_features',forbidden):
        numeric=candidate.train_prepared_quantile_folds(x,y,q,columns,
            feature_coordinate=kwargs['feature_coordinate'],response_coordinate=kwargs['response_coordinate'],
            seeds=kwargs['seeds'])
    assert numeric['fold_states']==actual['fold_states']
    np.testing.assert_array_equal(numeric['oof_predictions'],actual['oof_predictions'])
    output=candidate.quantile_predictions(numeric['fold_predictions'],actual['observed_mask'],actual['observed_values'])
    np.testing.assert_array_equal(output,actual['predictions'])
    with patch.object(candidate,'prepare_features',forbidden), \
         patch.object(candidate,'train_quantile_branch',forbidden), \
         patch.object(candidate,'train_prepared_quantile_folds',forbidden):
        replay=candidate.replay_prepared_quantile_folds(numeric['fold_states'],q,columns,
            kwargs['feature_coordinate'],kwargs['response_coordinate'])
    np.testing.assert_array_equal(replay,numeric['fold_predictions'])
    rejected=[]
    invalid=[('dtype',x.astype(np.float64),y,q,columns),
             ('target_rank',x,y[:,0],q,columns),('rows',x,y[:-1],q,columns),
             ('query_width',x,y,q[:,:-1],columns),('empty_query',x,y,q[:0],columns),
             ('nonfinite',x*np.nan,y,q,columns),('column_count',x,y,q,columns[:-1]),
             ('duplicate_columns',x,y,q,[columns[0]]*len(columns))]
    with patch.object(candidate,'build_model',side_effect=AssertionError('Invalid contract reached model construction')):
        for label,a,b,c,d in invalid:
            try:
                candidate.train_prepared_quantile_folds(a,b,c,d,feature_coordinate=kwargs['feature_coordinate'],
                    response_coordinate=kwargs['response_coordinate'],seeds=kwargs['seeds'])
            except ValueError:
                rejected.append(label)
            else:
                raise AssertionError('Invalid training contract accepted')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
        raise ValueError('Training implementation changed')
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,
        frozen_lifecycle_commit=PREVIOUS,frozen_lifecycle_sha256=hashlib.sha256(previous).hexdigest(),
        training_sha256=digest,validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        lifecycle_outputs_exact=True,private_fold_states_exact=True,numeric_boundary_without_preparation=True,
        raw_fold_replay_exact=True,replay_training_and_preparation_disabled=True,
        full_schedule_fits=6,epochs_per_fit=600,rejected_before_model_construction=rejected,
        limitations=['The prepared trainer retains the fixed OSIC architecture and schedule; this is decomposition, not a general model trainer.',
                    'Synthetic regression against the preceding qualified lifecycle only; no empirical accuracy or full original CDG approval.'])


if __name__=='__main__':
    result=validate()
    (ROOT/'docs/reviews/osic_training_boundary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,full_schedule_fits=result['full_schedule_fits'],rejected=len(result['rejected_before_model_construction']))))
