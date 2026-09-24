"""Supplemental runtime evidence must not bypass publication prerequisites."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.review_conditional_correction_publication import review


@pytest.mark.parametrize('filename,mutation,message', [
    ('conditional_correction_runtime_profile.json', lambda r: r.update(passed=False), 'Runtime qualification missing'),
    ('conditional_correction_runtime_profile.json', lambda r: r.update(validator_sha256='0'*64), 'Qualification code drift'),
    ('conditional_correction_runtime_profile.json', lambda r: r.update(manifest_sha256={}), 'Manifest evidence drift'),
    ('conditional_correction_runtime_profile.json', lambda r: r['dependency_closure'].update(compatible=False), 'Runtime dependency drift'),
    ('conditional_correction_runtime_profile.json', lambda r: r.update(libpq_version=0), 'Database client profile differs'),
    ('conditional_correction_database_gates.json', lambda r: r.update(rejected_faults=[]), 'Database negative gates missing'),
    ('conditional_correction_database_gates.json', lambda r: r.update(rollback_verified=False), 'Database negative gates missing'),
])
def test_incomplete_qualification_rejected(filename, mutation, message):
    original = Path.read_text
    def read(path, *args, **kwargs):
        text = original(path, *args, **kwargs)
        if path.name == filename:
            report = json.loads(text)
            mutation(report)
            return json.dumps(report)
        return text
    with patch.object(Path, 'read_text', read), pytest.raises(ValueError, match=message):
        review()


def test_qualification_retains_deployment_and_source_scope_limits():
    result = review()
    assert result['eligible_for_approval_transaction']
    assert result['proposed_tier'] == 3 and not result['approved']
    assert any('HTTP deployment and clean installation' in text for text in result['limitations'])
    assert any('complete NASA workflow' in text for text in result['limitations'])
