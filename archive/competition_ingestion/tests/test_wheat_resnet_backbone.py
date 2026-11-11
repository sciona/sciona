"""Reject unauthenticated weights and executable legacy pickle globals."""
import io
import pickle

import pytest

from sciona import wheat_resnet_backbone
from sciona.wheat_legacy_weights import _Reader


def test_digest_rejected_before_archive_reader(tmp_path, monkeypatch):
    path = tmp_path / 'synthetic.pth'
    path.write_bytes(b'synthetic invalid checkpoint')
    def forbidden_reader(raw):
        pytest.fail('unauthenticated archive reached reader')
    monkeypatch.setattr(wheat_resnet_backbone, 'read_legacy_resnet152', forbidden_reader)
    with pytest.raises(ValueError, match='checkpoint differs'):
        wheat_resnet_backbone.initialized_resnet152(path)


def test_legacy_reader_rejects_unlisted_global():
    raw = pickle.dumps(eval, protocol=2)
    with pytest.raises(ValueError, match='unsupported pickle global'):
        _Reader(io.BytesIO(raw)).load()
