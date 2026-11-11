"""Independent positional TGS pseudo-label selection and source fold slicing.

Pinned references: bes/ensemble.py and phalanx/make_pseudo.py. No source
identifiers are required; all outputs index the caller's aligned population.
"""
import numpy as np


def select_pseudo_labels(confidence, area, nonconstant):
    confidence, area, nonconstant = map(np.asarray, (confidence, area, nonconstant))
    if confidence.ndim != 1 or not len(confidence) or area.shape != confidence.shape or nonconstant.shape != confidence.shape:
        raise ValueError('nonempty aligned population vectors required')
    if not np.isfinite(confidence).all() or (confidence < 0).any() or (confidence > 1).any():
        raise ValueError('finite confidence in [0,1] required')
    if area.dtype.kind not in 'iu' or (area < 0).any() or (area > 101**2).any() or nonconstant.dtype != np.bool_:
        raise ValueError('integer mask area and boolean image eligibility required')
    # Source shuffles all metadata separately from the Keras confident subset.
    order = np.random.RandomState(123).permutation(len(area))
    eligible = np.flatnonzero((confidence >= .9) & nonconstant)
    keras = eligible[np.random.RandomState(123).permutation(len(eligible))]
    buckets = [[] for _ in range(6)]
    for index in order:
        if confidence[index] < .97:
            continue
        pixels = area[index]
        if pixels == 0:
            if nonconstant[index]:
                buckets[0].append(int(index))
        elif 20 <= pixels <= 500:
            if confidence[index] >= .99:
                buckets[1].append(int(index))
        elif 500 < pixels <= 101**2 / 4:
            buckets[2].append(int(index))
        elif 101**2 / 4 < pixels <= 101**2 / 2:
            buckets[3].append(int(index))
        elif 101**2 / 2 < pixels <= 101**2 * 3 / 4:
            buckets[4].append(int(index))
        elif 101**2 * 3 / 4 < pixels < 101**2 * .9:
            buckets[5].append(int(index))
    buckets[0] = buckets[0][::2]
    boundaries = [(0,600,1200,1800,2400), (0,110,220,330,440),
                  (0,300,600,900,1200), (0,240,480,740,980),
                  (0,200,400,600,800), (0,130,260,390,520)]
    folds = []
    for fold in range(5):
        selected = []
        for bucket, cuts in zip(buckets, boundaries):
            selected.extend(bucket[cuts[fold]:cuts[fold+1] if fold < 4 else None])
        folds.append(np.asarray(selected, dtype=np.int64))
    return dict(keras_indices=keras, torch_folds=folds, metadata_order=order)
