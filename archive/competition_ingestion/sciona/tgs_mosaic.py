"""Independent reconstruction of the winning TGS vertical-mask propagation.

Method: bes/ensemble.py, winner revision
2f81d4dd8d50a01579e5f7650259dde92c5c3b8d. Mosaic construction remains a
separate required stage. Inputs use positional indices, never source identities.
"""
import numpy as np


def propagate_masks(training_masks, nonconstant_training, query_masks, mosaics):
    """Apply source propagation to caller-supplied, ordered mosaic grids.

    Grid indices address training masks followed by query masks; -1 is empty.
    The first qualifying training tile in each column propagates downward.
    Fully vertical masks additionally propagate to the immediately preceding
    tile only when their row index is at least three, preserving the source's
    unusual inclusive slice. Later grids/columns overwrite earlier matches.
    Caller ordering is explicit because source directory enumeration is not.
    """
    train = np.asarray(training_masks)
    query = np.asarray(query_masks)
    eligible = np.asarray(nonconstant_training)
    for array in (train, query):
        if array.ndim != 3 or array.shape[0] == 0 or array.shape[1:] != (101, 101):
            raise ValueError('masks must have nonempty population and 101 by 101 spatial shape')
        if not np.isin(array, [0, 1]).all():
            raise ValueError('masks must be binary')
    if eligible.dtype != np.bool_ or eligible.shape != (len(train),):
        raise ValueError('nonconstant_training must be an aligned boolean vector')
    grids = []
    for raw in mosaics:
        grid = np.asarray(raw)
        if grid.ndim != 2 or not grid.size or grid.dtype.kind not in 'iu':
            raise ValueError('mosaics must be nonempty integer matrices')
        if (grid < -1).any() or (grid >= len(train) + len(query)).any():
            raise ValueError('mosaic index outside supplied populations')
        grids.append(grid)
    result = query.astype(bool, copy=True)
    replacements = {}
    for grid in grids:
        for column in grid.T:
            for row, value in enumerate(column):
                index = int(value)
                if index < 0 or index >= len(train) or not eligible[index]:
                    continue
                mask = train[index]
                last = mask[-1].astype(bool)
                if not last.any() or last.all() or not np.all(mask[-50:] == last):
                    continue
                vertical = np.all(mask == last)
                propagated = np.broadcast_to(last, (101, 101)).copy()
                for target in column[row + 1:]:
                    if int(target) >= len(train):
                        replacements[int(target) - len(train)] = propagated
                if vertical and row >= 3:
                    target = int(column[row - 1])
                    if target >= len(train):
                        replacements[target - len(train)] = propagated
                break
    for index, mask in replacements.items():
        result[index] = mask
    return dict(masks=result, replaced_query_indices=sorted(replacements))
