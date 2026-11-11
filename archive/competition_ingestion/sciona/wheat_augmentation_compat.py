"""Scoped removed-API aliases for isolated historical augmentation workers."""
import collections
import collections.abc
from contextlib import contextmanager

import numpy as np


@contextmanager
def historical_augmentation_api():
    """Use only in a dedicated, serial augmentation process.

    imgaug 0.2.6 reads only the float entry of np.sctypes at import and uses
    collections.Iterable during execution. Both namespaces are restored even
    when a transform raises; no package files are changed.
    """
    missing=object()
    previous_types=np.__dict__.get('sctypes',missing)
    previous_iterable=collections.__dict__.get('Iterable',missing)
    if previous_types is missing:
        floats=[np.sctypeDict[f'float{bits}'] for bits in (16,32,64,80,96,128,256,512)
                if f'float{bits}' in np.sctypeDict]
        np.sctypes={'float':floats}
    if previous_iterable is missing:
        collections.Iterable=collections.abc.Iterable
    try:
        yield
    finally:
        if previous_types is missing:
            del np.sctypes
        else:
            np.sctypes=previous_types
        if previous_iterable is missing:
            del collections.Iterable
        else:
            collections.Iterable=previous_iterable
