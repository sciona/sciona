"""Data-only reader for the pinned, pre-zip ResNet152 archive.

No archive extraction or imported pickle globals. Historical class names map
to inert local tokens; only contiguous float32 tensors are accepted.
"""
from collections import OrderedDict
import io
import pickle
import struct
import tarfile

import torch


class _Storage:
    pass


class _Tensor:
    pass


class _Parameter:
    def __setstate__(self, state):
        if (not isinstance(state, tuple) or len(state) != 5
                or not isinstance(state[0], torch.Tensor)
                or state[1:] != (None, None, True, False)):
            raise ValueError('unsupported historical parameter state')
        self.tensor = state[0]


class _Reader(pickle.Unpickler):
    def __init__(self, stream, tensors=None):
        super().__init__(stream)
        self.tensors = tensors or {}

    def find_class(self, module, name):
        allowed = {('collections', 'OrderedDict'): OrderedDict,
                   ('torch', 'FloatStorage'): _Storage,
                   ('torch', 'FloatTensor'): _Tensor,
                   ('torch.nn.parameter', 'Parameter'): _Parameter}
        if (module, name) not in allowed:
            raise ValueError('unsupported pickle global')
        return allowed[module, name]

    def persistent_load(self, key):
        if not isinstance(key, str) or not key.isdecimal() or int(key) not in self.tensors:
            raise ValueError('unsupported persistent tensor reference')
        return self.tensors[int(key)]


def read_legacy_resnet152(data):
    """Caller authenticates full archive digest before invoking this reader."""
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:') as archive:
        members = archive.getmembers()
        if (len(members) != 4 or {m.name for m in members} != {'sys_info', 'pickle', 'tensors', 'storages'}
                or any(not m.isfile() or m.size > 250_000_000 for m in members)):
            raise ValueError('unsupported legacy archive layout')
        def stream(name):
            return io.BytesIO(archive.extractfile(name).read())
        info = _Reader(stream('sys_info')).load()
        if info != {'type_sizes': {'long': 4, 'short': 2, 'int': 4}, 'protocol_version': 1000, 'little_endian': True}:
            raise ValueError('unsupported legacy platform')
        storages = {}
        f = stream('storages')
        if _Reader(f).load() != 777:
            raise ValueError('unexpected storage count')
        for _ in range(777):
            key, location, kind = _Reader(f).load()
            if type(key) is not int or key in storages or location != 'cpu' or kind is not _Storage:
                raise ValueError('unsupported storage record')
            count, = struct.unpack('<q', f.read(8))
            if not 0 < count <= 25_000_000:
                raise ValueError('invalid storage size')
            raw = bytearray(f.read(count * 4))
            if len(raw) != count * 4:
                raise ValueError('truncated storage')
            storages[key] = torch.frombuffer(raw, dtype=torch.float32).clone()
        if _Reader(f).load() != [] or f.read(1):
            raise ValueError('storage views or trailing bytes unsupported')
        tensors = {}
        f = stream('tensors')
        if _Reader(f).load() != 777:
            raise ValueError('unexpected tensor count')
        used = set()
        for _ in range(777):
            key, storage_id, kind = _Reader(f).load()
            if key in tensors or storage_id not in storages or kind is not _Tensor:
                raise ValueError('unsupported tensor record')
            ndim, = struct.unpack('<i', f.read(4))
            # Historical writer emitted an eight-byte slot for a C int;
            # its upper four bytes are padding, not a value or zero promise.
            if not 1 <= ndim <= 4 or len(f.read(4)) != 4:
                raise ValueError('invalid tensor dimensions')
            shape = struct.unpack(f'<{ndim}q', f.read(8 * ndim))
            stride = struct.unpack(f'<{ndim}q', f.read(8 * ndim))
            offset, = struct.unpack('<q', f.read(8))
            expected, count = [], 1
            for size in reversed(shape):
                if size < 1:
                    raise ValueError('invalid tensor shape')
                expected.insert(0, count)
                count *= size
            if offset != 0 or tuple(expected) != stride or count != storages[storage_id].numel():
                raise ValueError('only complete contiguous tensors supported')
            tensors[key] = storages[storage_id].reshape(shape)
            used.add(storage_id)
        if f.read(1) or used != set(storages):
            raise ValueError('unconsumed tensor content')
        f = stream('pickle')
        result = _Reader(f, tensors).load()
        if f.read(1) or not isinstance(result, OrderedDict) or len(result) != 777:
            raise ValueError('invalid reference dictionary')
        result = {key: value.tensor if isinstance(value, _Parameter) else value for key, value in result.items()}
        if any(not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in result.items()):
            raise ValueError('tensor dictionary required')
        return result
