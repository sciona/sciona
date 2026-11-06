"""Explicit raw numeric boundary and finite result encoding for the VSB recipe."""
import json
import numpy as np
from .vsb_execution import prepare


def unpack(payload):
    """Use the unchanged root validation; return independent numeric inputs.

    Consecutive columns form ordered three-signal measurements. Opaque identity
    validation remains a root concern; numeric downstream operations receive no
    identities. This stage does not extract features or train models.
    """
    canonical=json.loads(prepare(payload).configuration)
    features=dict(retention=.99,window=25,width=9,trim=8,threshold=-.01,count=1000,
                  offset=4,small_radius=5,large_radius=25)
    training=dict(repetitions=25,split_seed=123948,tree_seed=23974,rounds=10000,patience=100)
    return (np.asarray(canonical['training']['signals']),np.asarray(canonical['training']['labels']),
            np.asarray(canonical['query']['signals']),features,training)


def encode_result(trained):
    """Retain the reference output fields and query order; reject nonfinite JSON."""
    out=dict(probabilities=trained['probabilities'].tolist(),signal_decisions=trained['signal_decisions'].tolist(),
        threshold=trained['threshold'],models=trained['models'],scope=trained['scope'])
    json.dumps(out,allow_nan=False)
    return out
