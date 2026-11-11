"""Reject incomplete or changed source and execution evidence before catalog work."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from scripts.review_adaptive_history import review


@pytest.mark.parametrize('filename,mutation,message', [
    ('adaptive_history_source_comparison.json',lambda r:r.update(source_cases=0),'Source comparisons missing'),
    ('adaptive_history_source_comparison.json',lambda r:r.update(source_sha256='0'*64),'Source pin differs'),
    ('adaptive_history_source_comparison.json',lambda r:r.update(provider_sha256='0'*64),'Provider evidence drift'),
    ('adaptive_history_graph_execution.json',lambda r:r.update(implementation_sha256={}),'Missing implementation evidence'),
    ('adaptive_history_graph_execution.json',lambda r:r.update(scenarios=[]),'Required graph scenarios missing'),
    ('adaptive_history_graph_execution.json',lambda r:r.update(dimensional_witness_passed=False),'Graph execution missing'),
    ('adaptive_history_graph_execution.json',lambda r:r.update(graph_sha256='0'*64),'Graph digest differs'),
])
def test_changed_evidence_rejected(filename,mutation,message):
    original=Path.read_text
    def read(path,*args,**kwargs):
        text=original(path,*args,**kwargs)
        if path.name==filename:
            data=json.loads(text);mutation(data);return json.dumps(data)
        return text
    with patch.object(Path,'read_text',read),pytest.raises(ValueError,match=message):
        review()
