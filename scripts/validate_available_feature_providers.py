"""Cold discovery and materialized execution of reusable feature atom providers."""
import contextlib
import hashlib
import importlib
import inspect
import io
import json
from pathlib import Path
import runpy

import numpy as np
import pandas as pd

from sciona.ghost.registry import REGISTRY
from sciona.visualizer.runner import _ensure_atoms_imported

ROOT=Path(__file__).resolve().parents[1]
MODULE='sciona.atoms.ml.tabular.available_features'


def same(actual,expected):
    if isinstance(expected,pd.DataFrame):pd.testing.assert_frame_equal(actual,expected)
    elif isinstance(expected,dict):
        assert actual.keys()==expected.keys()
        for key in expected:same(actual[key],expected[key])
    elif isinstance(expected,tuple):
        assert len(actual)==len(expected)
        for left,right in zip(actual,expected):same(left,right)
    else:np.testing.assert_array_equal(actual,expected)


def validate():
    names=['available_indices','configuration_features','estimate_features','forecast_features','event_counts',
        'event_delays','calendar_features','entity_attributes','identifier_features','assemble_tables',
        'prepare_features','estimate_fallback','clip_truncate']
    required={MODULE+'.'+name for name in names}
    if required&REGISTRY.keys():raise ValueError('Cold process without provider preimports required')
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):_ensure_atoms_imported()
    if required-REGISTRY.keys():raise ValueError('Standard loader missed reusable providers')
    module=importlib.import_module(MODULE)
    def fixture(name):return runpy.run_path(str(ROOT/'tests'/('test_'+name+'.py')))
    configuration=fixture('configuration_state_features');estimate=fixture('estimate_history_features')
    forecast=fixture('forecast_history_features');counts=fixture('event_count_windows')
    delays=fixture('event_delay_windows');identifier=fixture('structured_identity_features');fallback=fixture('asof_estimate_fallback')
    cases={
        'available_indices':('_asof',[np.array([10,20],dtype=np.int64),np.array([0,10,20],dtype=np.int64)],{}),
        'configuration_features':('_configuration',configuration['args'](),{}),
        'estimate_features':('_estimates',estimate['inputs'](),{}),
        'forecast_features':('_forecasts',forecast['args'](),forecast['policy']()),
        'event_counts':('_counts',counts['inputs'](),{}),
        'event_delays':('_delays',delays['arrays'](),{}),
        'calendar_features':('_calendar',[pd.DatetimeIndex(['2000-01-01']),pd.DatetimeIndex(['2000-01-02'])],{}),
        'entity_attributes':('_entities',[np.array(['b','missing','a']),np.array(['a','b']),pd.DataFrame({'mode':['on','off']}),{'mode':['on']},'OTHER'],{}),
        'identifier_features':('_identifiers',identifier['args'](),identifier['policy']()),
        'assemble_tables':('_assemble',[pd.DataFrame({'key':[2,1,2]}),[(pd.DataFrame({'key':[1,2],'value':[3,4]}),['key'])]],{}),
        'prepare_features':('_prepare',[pd.DataFrame({'x':[np.nan,2.5],'cat':['on',None]}),['cat','x'],{'x':0},{'cat':'OTHER'}],{'integer_dtype':'int16'}),
        'estimate_fallback':('_fallback',fallback['args'](),fallback['policy']()),
        'clip_truncate':('_clip',[np.array([-10.,2.5,400.]),np.zeros(3),1,299],{}),
    }
    records=[]
    for name in names:
        implementation=REGISTRY[MODULE+'.'+name]['impl']
        target,args,kwargs=cases[name]
        inspect.signature(implementation).bind(*args,**kwargs)
        same(implementation(*args,**kwargs),getattr(module,target)(*args,**kwargs))
        records.append(dict(name=name,materialized_execution_passed=True,inputs=list(inspect.signature(implementation).parameters)))
    try:module.witness_materialized()
    except ValueError:pass
    else:raise ValueError('Unsupported static propagation was accepted')
    provider=Path(module.__file__).resolve()
    return dict(passed=True,approved=False,catalog_mutations=0,provider_preimports=False,
        standard_loader_discovery=True,providers=records,provider_count=len(records),
        unsupported_static_propagation_rejected=True,
        provider_source_sha256=hashlib.sha256(provider.read_bytes()).hexdigest(),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=['Local registration and materialized parity only; database staging, semantic ports, version-bound review and publication remain pending.'])


if __name__=='__main__':
    report=validate()
    (ROOT/'docs/reviews/available_feature_providers.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ['passed','provider_count','standard_loader_discovery','approved']}))
