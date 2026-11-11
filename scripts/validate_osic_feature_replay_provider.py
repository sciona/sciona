"""Qualify frozen feature replay independently of model construction or training."""
import copy
import hashlib
import inspect
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sciona import osic_feature_state as implementation
from sciona import osic_quantile_features as features
from sciona.atoms.ml.domain_adapters import osic_feature_replay as provider
from scripts.validate_osic_quantile_features import synthetic_tables

ROOT=Path(__file__).resolve().parents[1]
FEATURE='synthetic-feature-coordinate.v1'
RESPONSE='synthetic-response-coordinate.v1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate():
    files=['sciona/osic_feature_state.py','sciona/osic_quantile_features.py',
           'scripts/validate_osic_quantile_features.py','scripts/validate_osic_feature_replay_provider.py']
    before={name:sha(ROOT/name) for name in files}
    cases=[]
    for dtype in (np.float32,np.float64):
        train,baseline,queries=synthetic_tables()
        queries=pd.concat([queries,baseline[['entity','time']]],ignore_index=True)
        for frame in (train,baseline,queries):
            for field in ('time','target','numeric'):
                if field in frame:frame[field]=frame[field].astype(dtype)
        expected=features.prepare_features(train,baseline,queries,allow_transductive=True)
        state=json.loads(json.dumps(implementation.export_preparation_state(train,baseline,queries,
            feature_coordinate=FEATURE,response_coordinate=RESPONSE,allow_transductive=True),allow_nan=False))
        original=copy.deepcopy(state);original_queries=queries.copy(deep=True)
        def forbidden(*args,**kwargs):raise AssertionError('Feature replay attempted fitting')
        with patch.object(implementation,'assemble_populations',forbidden), \
             patch.object(implementation,'joint_population_features',forbidden), \
             patch.object(features,'prepare_features',forbidden):
            result=provider.transform_frozen_queries(state,queries,FEATURE,RESPONSE)
        assert set(result)=={'features','feature_columns','observed_mask','observed_values'}
        np.testing.assert_array_equal(result['features'],expected['populations']['query']['features'])
        assert result['feature_columns']==expected['feature_columns']
        np.testing.assert_array_equal(result['observed_mask'],[False,False,False,True,True])
        np.testing.assert_array_equal(result['observed_values'],[0.,0.,0.,999.,50.])
        assert state==original
        pd.testing.assert_frame_equal(queries,original_queries)
        # Output metadata is a separate copy; downstream graph conversion cannot
        # alter the saved feature-order declaration through this result.
        result['feature_columns'][0].append('synthetic-mutation')
        assert state==original
        cases.append(dict(arithmetic_dtype=dtype.__name__,queries=len(queries),
            exact_fitted_features=True,exact_observation_mask=True,exact_observation_values=True,
            state_and_queries_unchanged=True,output_columns_independent=True,fitting_disabled=True))
    rejected=[]
    mutations=[('format',lambda s:s.update(format='invalid')),
        ('extra_field',lambda s:s.update(import_path='forbidden')),
        ('feature_coordinate',lambda s:s.update(feature_coordinate='wrong')),
        ('response_coordinate',lambda s:s.update(response_coordinate='wrong')),
        ('query_scope',lambda s:s['query_scope'][0].update(time=1234.)),
        ('column_order',lambda s:s['feature_columns'].reverse()),
        ('missing_observation',lambda s:s['observations'].pop()),
        ('duplicate_observation',lambda s:s['observations'].append(copy.deepcopy(s['observations'][0]))),
        ('nonfinite_observation',lambda s:s['observations'][0].update(baseline_target=float('nan'))),
        ('degenerate_range',lambda s:s['ranges'].update(numeric=[1.,1.])),
        ('object_dtype',lambda s:s['dtypes'].update(numeric='object'))]
    for label,mutation in mutations:
        changed=copy.deepcopy(state);mutation(changed)
        try:provider.transform_frozen_queries(changed,queries,FEATURE,RESPONSE)
        except (ValueError,TypeError):rejected.append(label)
        else:raise AssertionError('Invalid feature state accepted: '+label)
    try:provider.transform_frozen_queries(state,queries.iloc[::-1],FEATURE,RESPONSE)
    except ValueError:rejected.append('query_order')
    else:raise AssertionError('Changed query order accepted')
    try:provider.witness_feature_replay()
    except ValueError:rejected.append('abstract_execution')
    else:raise AssertionError('Abstract execution accepted')
    assert len(rejected)==13
    if before!={name:sha(ROOT/name) for name in files}:raise ValueError('Qualification software changed')
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,cases=cases,
        rejected_contracts=rejected,provider_sha256=sha(inspect.getfile(provider)),software_sha256=before,
        runtime=dict(numpy=np.__version__,pandas=pd.__version__),
        limitations=[
            'Frozen source-specific transductive preparation only; exact declared query population and order required.',
            'Compared against qualified fitted feature preparation, not a new independent source-notebook replay.',
            'Input dtype conversion, column serialization, model replay, reduction and final blending remain separate graph operations.',
            'Private state contains caller observations and query scope; structural checks do not establish authenticity or empirical effectiveness.',
            'Provider qualification only; catalog approval and original-CDG approval remain pending.'])


if __name__=='__main__':
    result=validate()
    (ROOT/'docs/reviews/osic_feature_replay_provider.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,cases=len(result['cases']),rejected=len(result['rejected_contracts']),approved=False)))
