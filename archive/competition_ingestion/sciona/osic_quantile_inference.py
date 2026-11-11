"""Raw-query quantile replay with explicit frozen transductive preparation."""
import json

import numpy as np
import pandas as pd

from sciona.osic_feature_state import transform_queries
from sciona.osic_quantile_training import replay_quantile_branch


def predict_from_state(preparation, models, queries, *, feature_coordinate, response_coordinate):
    """Replay without fitting; preserve caller keys alongside the frozen query order.

    Queries contain exactly key/entity/time. Preparation is bound to the complete
    transductive query population and order. Baselines and observation availability
    are private caller-owned state, not inferred here. The response coordinate must
    match both fitted models and preparation. Application units and availability
    remain caller responsibilities. Returns source point and raw uncertainty columns.
    """
    if (not isinstance(queries,pd.DataFrame) or queries.empty
            or set(queries.columns)!={'key','entity','time'}
            or any(type(k) is not str or not k for k in queries['key'])
            or queries['key'].duplicated().any()):
        raise ValueError('Nonempty raw queries with unique explicit keys required')
    prepared=transform_queries(preparation,queries[['entity','time']],
        feature_coordinate=feature_coordinate,response_coordinate=response_coordinate)
    columns=[json.dumps(c,separators=(',',':')) for c in prepared['feature_columns']]
    values=replay_quantile_branch(models,prepared['features'].astype(np.float32),columns,
        feature_coordinate,response_coordinate,prepared['observed_mask'],prepared['observed_values'])
    return dict(keys=queries['key'].tolist(),values=values)
