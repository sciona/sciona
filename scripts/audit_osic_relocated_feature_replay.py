"""Read-only synthetic smoke check of the relocated feature replay imports."""
import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

import numpy as np
import pandas as pd
from sciona._runtime.ml import osic_feature_state as state_runtime
from sciona._runtime.ml import osic_quantile_features as features
from sciona.atoms.ml.domain_adapters import osic_feature_replay as provider

ROOT=Path(__file__).resolve().parents[1]
FIXTURE_COMMIT='5ed15bda'


def audit():
    source=subprocess.check_output(['git','show',FIXTURE_COMMIT+':scripts/validate_osic_quantile_features.py'],cwd=ROOT)
    fixture=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='synthetic_tables')
    scope={'pd':pd,'FIELDS':features.FIELDS}
    exec(compile(ast.Module(body=[fixture],type_ignores=[]),'<frozen-synthetic-fixture>','exec'),scope)
    paths={module.__name__:Path(inspect.getfile(module)) for module in (provider,state_runtime,features)}
    paths['validator']=Path(__file__)
    def hashes():return {name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in paths.items()}
    before=hashes()
    cases=[]
    for dtype in ('float32','float64'):
        train,base,query=scope['synthetic_tables']()
        query=pd.concat([query,base[['entity','time']]],ignore_index=True)
        for frame in (train,base,query):
            for field in ('time','target','numeric'):
                if field in frame:frame[field]=frame[field].astype(dtype)
        coordinates=dict(feature_coordinate='synthetic-joint-features.v1',response_coordinate='synthetic-response.v1')
        state=state_runtime.export_preparation_state(train,base,query,allow_transductive=True,**coordinates)
        state=json.loads(json.dumps(state,allow_nan=False))
        original=copy.deepcopy(state)
        expected=features.prepare_features(train,base,query,allow_transductive=True)['populations']['query']['features']
        def forbidden(*args,**kwargs):raise AssertionError('Replay attempted fitting')
        with patch.object(features,'prepare_features',forbidden), \
             patch.object(state_runtime,'assemble_populations',forbidden), \
             patch.object(state_runtime,'joint_population_features',forbidden):
            actual=provider.transform_frozen_queries(state,query,**coordinates)
        np.testing.assert_array_equal(actual['features'],expected)
        np.testing.assert_array_equal(actual['observed_mask'],[False,False,False,True,True])
        np.testing.assert_array_equal(actual['observed_values'],[0,0,0,999,50])
        assert state==original
        for label,queries,coords in [('order',query.iloc[::-1],coordinates),
            ('coordinate',query,dict(coordinates,response_coordinate='wrong'))]:
            try:provider.transform_frozen_queries(state,queries,**coords)
            except ValueError:pass
            else:raise AssertionError('Invalid relocated replay accepted: '+label)
        cases.append(dict(dtype=dtype,queries=len(query),features_exact=True,observation_policy_exact=True,
                          state_unchanged=True,fitting_disabled=True,rejections=2))
    assert hashes()==before
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,
        fixture_commit=FIXTURE_COMMIT,cases=cases,software_sha256=before,
        limitations=['Import resolution and synthetic feature replay smoke check only; not full migration qualification.',
            'Current provider bytes differ from prior approved versions; catalog history and promotion require separate review.',
            'No image or quantile model execution, training continuation or combined graph verification.'])


if __name__=='__main__':
    result=audit()
    (ROOT/'docs/reviews/osic_relocated_feature_replay_smoke.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,dtype_cases=len(result['cases']),approved=False)))
