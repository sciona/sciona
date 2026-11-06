"""Private local JSONL feature access with bounded record reads.

Only offsets, labels and group identifiers remain resident during indexing.
Feature records are read on demand; callers own paths and never serialize the
store into catalog evidence. No worker pools or prefetch threads are created.
"""
import json
import math
import os
from pathlib import Path
from collections.abc import Sequence


def _signature(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _record(raw, labelled):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate record key')
            result[key] = value
        return result
    try:
        record = json.loads(raw, object_pairs_hook=unique_pairs)
    except (ValueError, UnicodeError) as error:
        raise ValueError('Invalid feature record') from None
    fields = {'frames', 'sparse', 'group'} | ({'labels'} if labelled else set())
    if type(record) is not dict or set(record) != fields:
        raise ValueError('Unexpected feature record fields')
    frames = record['frames']
    if type(frames) is not list or not frames or type(frames[0]) is not list or not frames[0]:
        raise ValueError('Nonempty frame matrix required')
    width = len(frames[0])
    def finite(value):
        try:
            return type(value) in (int, float) and math.isfinite(value)
        except OverflowError:
            return False
    if any(type(row) is not list or len(row) != width or not all(finite(v) for v in row) for row in frames):
        raise ValueError('Aligned finite frame vectors required')
    sparse = record['sparse']
    if type(sparse) is not dict or any(type(k) is not str or not k or not finite(v) or v < 0 for k,v in sparse.items()):
        raise ValueError('Finite nonnegative sparse mapping required')
    if type(record['group']) is not str or not record['group']:
        raise ValueError('Nonempty group required')
    if labelled:
        labels = record['labels']
        if type(labels) is not list or not labels or any(type(v) is not int or v not in (0,1) for v in labels):
            raise ValueError('Nonempty binary label vector required')
    return record, width


class FeatureShards(Sequence):
    """Indexed immutable local records, with at most one bounded record per read.

    Index storage is O(records + labels + group text); it is not a fixed total
    memory cap. A parsed record may occupy more memory than its encoded bytes.
    Source files must remain unchanged for the entire lifetime of the store.
    """
    def __init__(self, paths, *, labelled, max_record_bytes=8 * 1024 * 1024):
        if type(labelled) is not bool or type(max_record_bytes) is not int or max_record_bytes < 1:
            raise ValueError('Explicit label mode and positive record limit required')
        if not isinstance(paths, (list, tuple)) or not paths:
            raise ValueError('Nonempty explicit shard list required')
        self._paths = tuple(Path(p) for p in paths)
        self._signatures = []
        self._index = []
        groups, labels = [], []
        self.labelled = labelled
        self.max_record_bytes = max_record_bytes
        self.width = None
        self.label_count = None
        identities = set()
        for shard, path in enumerate(self._paths):
            with path.open('rb') as stream:
                signature = _signature(os.fstat(stream.fileno()))
                if signature[:2] in identities:
                    raise ValueError('Repeated shard file')
                identities.add(signature[:2])
                while True:
                    offset = stream.tell()
                    raw = stream.readline(max_record_bytes + 1)
                    if not raw:
                        break
                    if len(raw) > max_record_bytes:
                        raise ValueError('Feature record exceeds configured byte limit')
                    record, width = _record(raw, labelled)
                    if self.width is None:
                        self.width = width
                    if width != self.width:
                        raise ValueError('Feature widths differ across records')
                    if labelled:
                        count = len(record['labels'])
                        if self.label_count is None:
                            self.label_count = count
                        if count != self.label_count:
                            raise ValueError('Label widths differ across records')
                        labels.append(tuple(record['labels']))
                    groups.append(record['group'])
                    self._index.append((shard, offset, len(raw)))
                if _signature(os.fstat(stream.fileno())) != signature:
                    raise ValueError('Shard changed during indexing')
                self._signatures.append(signature)
        if not self._index:
            raise ValueError('Empty feature population')
        self.groups = tuple(groups)
        self.labels = tuple(labels) if labelled else None

    def __len__(self):
        return len(self._index)

    def __getitem__(self, index):
        if isinstance(index, slice):
            raise TypeError('Explicit bounded batches required; store slicing is disabled')
        if type(index) is not int:
            raise TypeError('Integer record index required')
        shard, offset, length = self._index[index]
        with self._paths[shard].open('rb') as stream:
            if _signature(os.fstat(stream.fileno())) != self._signatures[shard]:
                raise ValueError('Shard changed after indexing')
            stream.seek(offset)
            raw = stream.read(length)
            if len(raw) != length or _signature(os.fstat(stream.fileno())) != self._signatures[shard]:
                raise ValueError('Shard changed during read')
        return _record(raw, self.labelled)[0]

    def batches(self, batch_size, indices=None):
        """Read caller ordering in bounded batches, including the final partial batch."""
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('Positive batch size required')
        batch = []
        for index in range(len(self)) if indices is None else indices:
            batch.append(self[index])
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def validate_populations(training, calibration, query):
    """Keep group separation and labelled/query boundaries explicit."""
    stores = (training, calibration, query)
    if not all(isinstance(s, FeatureShards) for s in stores):
        raise ValueError('Feature shard stores required')
    if not training.labelled or not calibration.labelled or query.labelled:
        raise ValueError('Training/calibration labels and unlabelled query required')
    if len({s.width for s in stores}) != 1 or training.label_count != calibration.label_count:
        raise ValueError('Population feature or label widths differ')
    groups = [set(s.groups) for s in stores]
    if any(groups[a] & groups[b] for a,b in ((0,1),(0,2),(1,2))):
        raise ValueError('Population groups overlap')
