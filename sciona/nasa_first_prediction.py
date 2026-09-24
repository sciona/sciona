"""Corrected raw-input prediction and fallback composition for the source workflow."""
import numpy as np

from sciona.asof_estimate_fallback import asof_estimate_fallback
from sciona.model_feature_policy import prepare_model_features
from sciona.nasa_first_feature_adapter import nasa_first_feature_tables, _times, MINUTE
from sciona.prediction_fallback_chain import prediction_fallback_chain
from sciona.prediction_postprocessing import clip_truncate_prediction


def nasa_first_prediction(queries, models, population, *, raw_options, feature_columns,
        numeric_fills, categorical_fills):
    """Return an owned query table with predictions and the selected route.

    Native model prediction threads are bounded to one. The primary path shares
    raw feature computation and fixed input policy with training. The baseline
    uses latest nonmissing estimates observed by each query. Ordinary failures
    select baseline then constant 30; process-control exceptions propagate.
    The three source population overrides are software routing policy, not data.
    """
    def primary():
        raw=nasa_first_feature_tables(queries.copy(deep=True),**raw_options)
        prepared=prepare_model_features(raw,feature_columns,numeric_fills,categorical_fills)
        baseline=prepared['etd_time_till_est_dep'].to_numpy(dtype=np.float64)
        local=[]
        for slot in range(3):
            estimator=models[slot]
            values=np.asarray(estimator.predict(prepared[estimator.feature_names_],thread_count=1),dtype=np.float64)
            local.append(clip_truncate_prediction(values,baseline if slot in (0,2) else np.zeros(len(queries)),1,299))
        global_features=prepared.copy()
        global_features['feat_cat_airport']=population.lower()
        estimator=models['global_model']
        global_values=np.asarray(estimator.predict(global_features[estimator.feature_names_],thread_count=1),dtype=np.float64)
        if global_values.shape!=(len(queries),) or not np.isfinite(global_values).all():
            raise ValueError('Finite aligned global prediction required')
        combined=local[1] if population.upper() in ['KPHX','KMIA','KJFK'] else (local[0]+local[1]+local[2]+global_values)/4
        return clip_truncate_prediction(combined,np.zeros(len(queries)),4,260)

    def baseline():
        history=raw_options['etd']
        return asof_estimate_fallback(history['gufi'].to_numpy(),_times(history['timestamp']),
            _times(history['departure_runway_estimated_time'],missing=True),
            history['departure_runway_estimated_time'].notna().to_numpy(),
            queries['gufi'].to_numpy(),_times(queries['timestamp']),
            ticks_per_unit=MINUTE,offset=30,lower=1,upper=299,missing=15)

    values,route=prediction_fallback_chain(primary,baseline,len(queries),30)
    output=queries.copy(deep=True)
    output['minutes_until_pushback']=values
    return output,route
