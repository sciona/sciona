"""In-memory reconstruction of the winner's pseudo-label population split.

Derived from the MIT-licensed helper by DungNB (Nguyen Ba Dung),2020.
See docs/licenses/global-wheat-MIT.txt for the retained license notice.
Runtime paths and rows remain caller-owned private data. This function writes
no files. It preserves source row ordering before a seeded NumPy permutation.
"""
import os

import numpy as np
import pandas as pd


COORDINATES = ['xmin', 'ymin', 'xmax', 'ymax']
COLUMNS = ['image_path', *COORDINATES, 'isbox']


def build_pseudo_population(original, query_ids, predictions, *, train_directory,
                            query_directory, seed, validation_fold=1):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('explicit uint32 shuffle seed required')
    if type(validation_fold) is not int or validation_fold < 0:
        raise ValueError('nonnegative validation fold required')
    required = {'image_id', 'fold', 'isbox', *COORDINATES}
    if not isinstance(original, pd.DataFrame) or not required.issubset(original.columns) or not original.index.is_unique:
        raise ValueError('original population with unique row index required')
    ids = list(query_ids)
    if any(not isinstance(x, str) or not x for x in ids + original['image_id'].tolist()):
        raise ValueError('nonempty string image identities required')
    if not original['isbox'].map(lambda x: isinstance(x, (bool, np.bool_))).all():
        raise ValueError('boolean box-presence flags required')
    rows = []
    for identity in np.unique(ids):
        if identity not in predictions:
            raise ValueError('query prediction missing')
        boxes, scores = (np.asarray(x) for x in predictions[identity])
        if (boxes.ndim != 2 or boxes.shape[1:] != (4,) or scores.shape != (len(boxes),)
                or boxes.dtype.kind not in 'fiu' or scores.dtype.kind not in 'fiu'
                or not np.isfinite(boxes).all() or not np.isfinite(scores).all()
                or (boxes < 0).any() or (boxes[:, 2:] < boxes[:, :2]).any()
                or (scores < 0).any() or (scores > 1).any()):
            raise ValueError('finite aligned post-fusion boxes and scores required')
        path = os.path.join(query_directory, identity + '.jpg')
        if not len(boxes):
            rows.append(dict(image_path=path, **dict.fromkeys(COORDINATES), isbox=False))
        else:
            for box in boxes:
                rows.append(dict(image_path=path, **dict(zip(COORDINATES, box)), isbox=True))
    pseudo = pd.DataFrame(rows, columns=COLUMNS)
    frame = original.copy(deep=True)
    frame['image_path'] = [os.path.join(train_directory, identity + '.jpg') for identity in frame['image_id']]
    validation = frame.loc[frame['fold'] == validation_fold]
    training = frame.loc[~frame.index.isin(validation.index)]
    validation = validation.loc[validation['isbox'] == True]
    training = training[COLUMNS].reset_index(drop=True)
    validation = validation[COLUMNS].reset_index(drop=True)
    training = pd.concat([training, pseudo], ignore_index=True).sample(
        frac=1, random_state=np.random.RandomState(seed)).reset_index(drop=True)
    return training, validation
