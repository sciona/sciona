import numpy as np
import pandas as pd
import pytest

from sciona.atoms.ml.domain_adapters.airport_features import training_arrays, prediction_arrays, attach_output, history_features
from sciona.nasa_domain_graph import build_nasa_domain_graph
from sciona.nasa_workflow_inputs import ETD_FEATURES, HISTORY_FEATURES


def frame():
    result = pd.DataFrame(dict(gufi=['AAA1','AAA2'], timestamp=pd.to_datetime(['2020-01-01','2020-01-02']),
                               airport=['KAAA','KAAA'], minutes_until_pushback=[10.,20.]))
    for name in ['unix_time']+ETD_FEATURES+['Other','AAA']+HISTORY_FEATURES:
        result[name] = [1.,2.]
    return result


@pytest.mark.parametrize('names', [[], ['AAA','AAA'], ['missing'], ('AAA',), ['bad<name'], [7]])
def test_prediction_schema_is_explicit_and_unique(names):
    data = frame()
    with pytest.raises(ValueError):
        prediction_arrays(data, data, names)


@pytest.mark.parametrize('values', [['1','2'], [1+2j,2+1j], [np.inf, 2.]])
def test_prediction_values_reject_strings_complex_and_infinity(values):
    data = frame()
    data['AAA'] = values
    with pytest.raises(ValueError):
        prediction_arrays(data, data, ['AAA'])


def test_missing_query_features_preserve_identity_and_schema():
    data = frame()
    _, _, _, names = training_arrays(data, ('AAA',))
    query = data.iloc[::-1].reset_index(drop=True)
    data.loc[0, 'AAA'] = np.nan
    values, identities = prediction_arrays(data, query, names)
    assert np.isnan(values[1, names.index('AAA')])
    assert values[0, names.index('AAA')] == 2.
    pd.testing.assert_frame_equal(identities, query[['gufi','timestamp','airport']])
    result = attach_output(identities, np.array([3,4], dtype=np.int32))
    assert list(result.minutes_until_pushback) == [3,4]
    assert 'minutes_until_pushback' not in identities


@pytest.mark.parametrize('vocabulary', [['AAA'], ('AAA','AAA'), ('AA',)])
def test_training_vocabulary_must_be_fixed_and_well_formed(vocabulary):
    with pytest.raises(ValueError):
        training_arrays(frame(), vocabulary)


@pytest.mark.parametrize('values', [['10','20'], [10+1j,20+2j], [np.nan,20.], [np.inf,20.]])
def test_training_target_boundary(values):
    data = frame()
    data['minutes_until_pushback'] = values
    with pytest.raises(ValueError):
        training_arrays(data, ('AAA',))


def test_mixed_populations_and_duplicate_output_identities_rejected():
    data = frame()
    mixed = data.copy()
    mixed.loc[0,'airport'] = 'KBBB'
    with pytest.raises(ValueError):
        training_arrays(mixed, ('AAA',))
    with pytest.raises(ValueError):
        history_features(data, pd.DataFrame(), pd.DataFrame(), 'KBBB')
    duplicated = pd.concat([data.iloc[:1]]*2)
    with pytest.raises(ValueError):
        attach_output(duplicated, np.array([1,2], dtype=np.int32))


def test_domain_graph_keeps_all_numerical_nodes_and_explicit_contracts():
    graph = build_nasa_domain_graph()
    domain = [node for node in graph.nodes if node.node_id.startswith('domain_')]
    assert len(graph.nodes) == 40 and len(domain) == 15
    assert all(port.constraints and 'boundary; caller' not in port.constraints
               for node in domain for port in node.inputs+node.outputs)
    numeric_roots = {(edge.target_id,edge.input_name) for edge in graph.edges if edge.source_id.startswith('domain_')}
    assert {('training','features'),('training','targets'),('training','groups'),('training','feature_names'),
            ('query','prediction_features')} <= numeric_roots
