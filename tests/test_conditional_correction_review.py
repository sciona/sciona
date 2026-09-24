"""Tampered promotion evidence must not pass semantic qualification."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.review_conditional_correction import audit


@pytest.mark.parametrize('mutation,expected', [
    (lambda r: r.update(provider_sha256='0'*64), 'Provider evidence drift'),
    (lambda r: r.update(serialized_graph_sha256='0'*64), 'Graph evidence drift'),
    (lambda r: r.update(source_float64_cases=0), 'Source comparison evidence missing'),
    (lambda r: r.update(source_sha256={}), 'Source identity differs'),
    (lambda r: r.update(scenarios=[]), 'Cross-domain execution missing'),
    (lambda r: r.update(passed=False), 'Execution evidence missing'),
    (lambda r: r.update(implementation_sha256={}), 'Incomplete implementation evidence'),
])
def test_altered_execution_evidence_rejected(mutation, expected):
    original = Path.read_text
    def read(path, *args, **kwargs):
        text = original(path, *args, **kwargs)
        if path.name == 'conditional_correction_reuse.json':
            report = json.loads(text)
            mutation(report)
            return json.dumps(report)
        return text
    with patch.object(Path, 'read_text', read), pytest.raises(ValueError, match=expected):
        audit()


def test_current_review_preserves_incomplete_publication_status():
    report = audit()
    assert report['semantic_verdict'] == 'acceptable_with_limits'
    assert report['witness_composition_verified']
    assert report['publication_ready'] is False
    assert report['approved'] is False
    assert report['publication_blockers']
    assert report['catalog_mutations'] == 0
