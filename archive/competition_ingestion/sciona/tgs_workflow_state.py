"""Private runtime persistence for TGS fit receipts and pseudo-label arrays.

Never place this store in source control. Arrays use non-pickle NPY files;
an atomic JSON manifest binds them to caller-qualified input/code context.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np


class WorkflowStateStore:
    def __init__(self, directory, *, context_sha256):
        if not isinstance(context_sha256, str) or not re.fullmatch('[0-9a-f]{64}', context_sha256):
            raise ValueError('qualified runtime context digest required')
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.context = context_sha256

    def _write(self, path, data):
        descriptor, temporary = tempfile.mkstemp(prefix='.state-', dir=self.directory)
        try:
            with os.fdopen(descriptor, 'wb') as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def save(self, state):
        def encode(value):
            if isinstance(value, np.ndarray):
                if value.dtype.kind not in 'biuf' or not np.isfinite(value).all():
                    raise ValueError('finite numeric runtime arrays required')
                stream = io.BytesIO()
                np.save(stream, value, allow_pickle=False)
                data = stream.getvalue()
                digest = hashlib.sha256(data).hexdigest()
                self._write(self.directory / (digest + '.npy'), data)
                return dict(kind='array', sha256=digest, bytes=len(data))
            if isinstance(value, dict):
                if any(type(key) not in (str, int) for key in value):
                    raise ValueError('string or integer mapping keys required')
                return dict(kind='dict', items=[[key, encode(child)] for key, child in value.items()])
            if isinstance(value, (list, tuple)):
                return dict(kind='tuple' if isinstance(value, tuple) else 'list', items=[encode(v) for v in value])
            if isinstance(value, np.generic):
                value = value.item()
            if value is None or type(value) in (str, int, float, bool):
                return dict(kind='scalar', value=value)
            raise ValueError('unsupported runtime state value')
        payload = json.dumps(encode(state), allow_nan=False, separators=(',', ':'))
        manifest = dict(version=1, context_sha256=self.context,
                        payload_sha256=hashlib.sha256(payload.encode()).hexdigest(), payload=payload)
        self._write(self.directory / 'state.json', json.dumps(manifest, allow_nan=False).encode())

    def load(self):
        manifest = json.loads((self.directory / 'state.json').read_text())
        if manifest.get('version') != 1 or manifest.get('context_sha256') != self.context:
            raise ValueError('runtime context differs from saved state')
        payload = manifest['payload']
        if hashlib.sha256(payload.encode()).hexdigest() != manifest['payload_sha256']:
            raise ValueError('state manifest digest mismatch')
        def decode(value):
            kind = value['kind']
            if kind == 'array':
                digest = value['sha256']
                if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
                    raise ValueError('invalid array digest')
                data = (self.directory / (digest + '.npy')).read_bytes()
                if len(data) != value['bytes'] or hashlib.sha256(data).hexdigest() != digest:
                    raise ValueError('runtime array digest mismatch')
                array = np.load(io.BytesIO(data), allow_pickle=False)
                if array.dtype.kind not in 'biuf' or not np.isfinite(array).all():
                    raise ValueError('invalid runtime array')
                return array
            if kind == 'dict':
                return {key: decode(child) for key, child in value['items']}
            if kind in ('list', 'tuple'):
                items = [decode(child) for child in value['items']]
                return tuple(items) if kind == 'tuple' else items
            if kind == 'scalar':
                return value['value']
            raise ValueError('unknown state encoding')
        return decode(json.loads(payload))
