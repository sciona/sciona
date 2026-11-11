"""Reject stale or incomplete evidence before any Connectomics catalog action."""
import json
from pathlib import Path
from unittest.mock import patch
import pytest
from scripts.review_decomposed_connectomics import review


@pytest.mark.parametrize('filename,change', [
    ('connectomics_decomposed_graph_execution.json', lambda r: r.update(status='failed')),
    ('connectomics_decomposed_graph_execution.json', lambda r: r.update(serialized_graph_sha256='changed')),
    ('connectomics_decomposed_graph_execution.json', lambda r: r.update(all_nodes_uncached=False)),
    ('connectomics_decomposed_graph_execution.json', lambda r: r['cases'].pop()),
    ('connectomics_decomposed_graph_execution.json', lambda r: r['cases'][0].update(exact_source_match=False)),
    ('connectomics_decomposed_graph_execution.json', lambda r: r.update(code_sha256={})),
    ('connectomics_decomposed_graph_execution.json', lambda r: r.update(provider_sha256='changed')),
    ('connectomics_decomposed_graph_execution.json', lambda r: r.update(circular_provider_sha256='changed')),
    ('connectomics_generic_circular_filter_source_parity.json', lambda r: r.update(component_cases=0)),
    ('connectomics_generic_circular_filter_source_parity.json', lambda r: r.update(test_sha256='changed')),
    ('connectomics_generic_circular_filter_source_parity.json', lambda r: r['sweeps'][1].update(mode='simple')),
    ('connectomics_generic_circular_filter_source_parity.json', lambda r: r['sweeps'][1]['filter_calls'].update(f4=0)),
])
def test_review_rejects_bad_evidence(filename, change):
    original = Path.read_text
    def read(path, *args, **kwargs):
        text = original(path, *args, **kwargs)
        if path.name == filename:
            report = json.loads(text)
            change(report)
            return json.dumps(report)
        return text
    with patch.object(Path, 'read_text', read), pytest.raises(ValueError):
        review()


def test_current_review_covers_all_source_stages_and_generic_binding():
    result = review()
    assert len(result['source_stage_mapping']) == 9
    generic, = [p for p in result['providers'] if p['classification'] == 'generic_operation']
    assert generic['bound_nodes'] == ['f1', 'f2', 'f3', 'f4']
    assert result['approved'] is False and result['catalog_mutations'] == 0
