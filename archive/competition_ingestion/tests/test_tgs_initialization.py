"""Synthetic rejection fixtures; public model references are opt-in checks."""
import io
import pickle

import pytest
import torch

from sciona.tgs_initialization import load_resnet34_reference, load_resnext50_reference
from sciona.tgs_legacy_weights import _Reader


@pytest.mark.parametrize('loader', [load_resnet34_reference, load_resnext50_reference])
def test_bad_digest_rejected_before_model_access(tmp_path, loader):
    reference = tmp_path / 'synthetic'
    reference.write_bytes(b'synthetic invalid weight archive')
    with pytest.raises(ValueError, match='digest mismatch'):
        loader(object(), reference, '0' * 64)


def test_legacy_reader_rejects_arbitrary_globals():
    # GLOBAL itself is rejected before the following callable can run.
    payload = b'\x80\x02cbuiltins\neval\nX\x03\x00\x00\x001+1\x85R.'
    with pytest.raises(ValueError, match='unsupported pickle global'):
        _Reader(io.BytesIO(payload)).load()


def test_legacy_reader_only_resolves_known_numeric_tensor_ids():
    class ReferencePickler(pickle.Pickler):
        def persistent_id(self, value):
            return value if isinstance(value, str) else None
    for reference in ['12', 'missing']:
        output = io.BytesIO()
        ReferencePickler(output, protocol=2).dump(reference)
        if reference == '12':
            expected = torch.ones(3)
            assert _Reader(io.BytesIO(output.getvalue()), {12: expected}).load() is expected
        else:
            with pytest.raises(ValueError, match='persistent tensor reference'):
                _Reader(io.BytesIO(output.getvalue()), {12: torch.ones(3)}).load()
