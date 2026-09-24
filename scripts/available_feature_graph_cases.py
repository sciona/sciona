"""Synthetic case constructors shared by provider graph qualification."""
from pathlib import Path
import runpy
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]


def cases():
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
    return cases
