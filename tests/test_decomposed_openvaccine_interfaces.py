"""Negative interface/registration gates; full execution is separate evidence."""
import copy
from unittest.mock import patch

import pytest

from scripts import review_decomposed_openvaccine_interfaces as reviewed


def test_current_interfaces_and_reuse_boundary():
    result = reviewed.review()
    assert result['passed'] and not result['approved'] and result['catalog_mutations'] == 0
    assert result['root_inputs'] == ['payload'] and result['root_outputs'] == ['result']
    generic = [p for p in result['providers'] if p['classification'] == 'generic_operation']
    assert [p['runtime_fqdn'] for p in generic] == [reviewed.GENERIC]
    assert len([p for p in result['providers'] if p['semantics'] is not None]) == 4


@pytest.mark.parametrize('failure', ['missing_contract', 'extra_field', 'missing_parameters',
    'wrong_parameter', 'empty_bounds', 'empty_dependencies'])
def test_contract_gates(failure):
    contracts = copy.deepcopy(reviewed.CONTRACTS)
    if failure == 'missing_contract':
        contracts.pop('features')
    elif failure == 'extra_field':
        contracts['features']['invented'] = True
    elif failure == 'missing_parameters':
        contracts['features'].pop('parameters')
    elif failure == 'wrong_parameter':
        contracts['features']['parameters'] = {'raw': 'wrong input'}
    elif failure == 'empty_bounds':
        contracts['features']['validity_bounds'] = ''
    else:
        contracts['features']['dependencies'] = []
    with patch.object(reviewed, 'CONTRACTS', contracts), pytest.raises(ValueError):
        reviewed.review()


@pytest.mark.parametrize('failure', ['missing', 'duplicate'])
def test_registration_gates(failure):
    parse = reviewed._parse_registered_atoms
    def altered(**kwargs):
        specs = list(parse(**kwargs))
        selected = next((s for s in specs if s.import_module+'.'+s.source_symbol == reviewed.PREFIX+'features'), None)
        if selected is not None:
            if failure == 'missing':
                specs.remove(selected)
            else:
                specs.append(selected)
        return specs
    with patch.object(reviewed, '_parse_registered_atoms', side_effect=altered), pytest.raises(ValueError):
        reviewed.review()


def test_symbolic_execution_is_rejected():
    import sciona.atoms.ml.decomposed_openvaccine_execution as provider
    with patch.object(provider, 'concrete_only', return_value=object()), pytest.raises(ValueError):
        reviewed.review()


@pytest.mark.parametrize('failure', ['node_count', 'edge_count', 'root_input'])
def test_graph_boundary_gates(failure):
    graph = reviewed.build_decomposed_openvaccine_graph()
    if failure == 'node_count':
        graph.nodes.pop()
    elif failure == 'edge_count':
        graph.edges.pop()
    else:
        graph.nodes[0].inputs[0].name = 'different'
    with patch.object(reviewed, 'build_decomposed_openvaccine_graph', return_value=graph), pytest.raises(ValueError):
        reviewed.review()
