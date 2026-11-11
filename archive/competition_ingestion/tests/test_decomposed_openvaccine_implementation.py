"""Reject incomplete or changed full-execution qualification evidence."""
import copy
import json
from unittest.mock import patch

import pytest

from scripts import review_decomposed_openvaccine_implementation as reviewed


def evidence():
    return json.loads((reviewed.ROOT/reviewed.EVIDENCE).read_text())


def test_current_full_execution_identity():
    graph = reviewed.validate_execution(evidence())
    assert len(graph.nodes) == 6 and len(graph.edges) == 7


@pytest.mark.parametrize('failure', ['header', 'header_type', 'extra_field', 'missing_field',
    'incomplete_models', 'no_reference', 'no_byte_parity', 'check_type', 'threads', 'thread_type',
    'missing_code', 'changed_code', 'runtime', 'software', 'graph', 'missing_provider', 'changed_provider'])
def test_execution_gates(failure):
    full = copy.deepcopy(evidence())
    if failure == 'header':
        full['synthetic_only'] = False
    elif failure == 'header_type':
        full['catalog_mutations'] = False
    elif failure == 'extra_field':
        full['unexpected'] = True
    elif failure == 'missing_field':
        full.pop('runtime')
    elif failure == 'incomplete_models':
        full['checks']['models'] = 19
    elif failure == 'no_reference':
        full['checks']['full_reference_retrained'] = False
    elif failure == 'no_byte_parity':
        full['checks']['prediction_and_teacher_bytes_equal'] = False
    elif failure == 'check_type':
        full['checks']['refinement_rounds'] = True
    elif failure == 'threads':
        full['resource_policy']['native_threads'] = 8
    elif failure == 'thread_type':
        full['resource_policy']['native_threads'] = True
    elif failure == 'missing_code':
        full['code_sha256'].pop('sciona/openvaccine_lifecycle.py')
    elif failure == 'changed_code':
        full['code_sha256']['sciona/openvaccine_lifecycle.py'] = '0'*64
    elif failure == 'runtime':
        full['runtime']['numpy'] = '0.0.0'
    elif failure == 'software':
        full['provisioned_software']['binary_sha256'] = '0'*64
    elif failure == 'graph':
        full['serialized_graph_sha256'] = '0'*64
    elif failure == 'missing_provider':
        full['bound_provider_sha256'].pop(next(iter(full['bound_provider_sha256'])))
    else:
        full['bound_provider_sha256'][next(iter(full['bound_provider_sha256']))] = '0'*64
    with pytest.raises(ValueError):
        reviewed.validate_execution(full)


def test_committed_evidence_gate():
    with patch.object(reviewed.subprocess, 'check_output', return_value=b'{}'), pytest.raises(ValueError):
        reviewed.committed(reviewed.EVIDENCE)


def test_hash_inventory_rejects_external_paths():
    with pytest.raises(ValueError):
        reviewed.check_hashes({'../outside': '0'*64})
