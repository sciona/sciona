"""Frozen intake must reject source, identity and interface substitutions."""
import json
from pathlib import Path
from unittest.mock import patch
import pytest

from scripts.stage_connectomics_filter_providers import PLAN, reviewed_plan


@pytest.mark.parametrize('change', [
    lambda r: r.update(graph_digest='changed'),
    lambda r: r.update(review_sha256='changed'),
    lambda r: r['providers'][0].update(atom_id='changed'),
    lambda r: r['providers'][0].update(version_id='changed'),
    lambda r: r['providers'][0].update(content_hash='changed'),
    lambda r: r['providers'][0].update(provider_sha256='changed'),
    lambda r: r['providers'][0]['interfaces'][0].update(name='changed'),
    lambda r: r['providers'][0]['interfaces'][0].update(type_desc='changed'),
    lambda r: r['providers'][0]['interfaces'][0].update(required=False),
    lambda r: r['providers'][0]['interfaces'].pop(),
])
def test_rejects_changed_frozen_intake(change):
    original = Path.read_text
    def read(path, *args, **kwargs):
        value = original(path, *args, **kwargs)
        if path == PLAN:
            result = json.loads(value)
            change(result)
            return json.dumps(result)
        return value
    with patch.object(Path, 'read_text', read), pytest.raises(ValueError, match='Frozen provider intake changed'):
        reviewed_plan()
