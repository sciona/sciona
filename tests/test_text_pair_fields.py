"""Synthetic separate-field fidelity, full-record split keys and lifecycle."""
import numpy as np
import pytest
from sciona.text_pair_fields import normalize_populations, normalize_records, FieldPairFeatures, execute
from tests.test_text_pair_complete_recipe import CONTROLS, policy
from tests.test_text_pair_training import data


def record(title, description):
    return dict(title=title, description=description)


def records(pairs):
    return [[record(a, a), record(b, b)] for a, b in pairs]


def feature_controls():
    return {k: v for k, v in CONTROLS.items() if k not in ('trees', 'max_depth', 'min_leaf')}


def test_description_jaccard_and_title_edit_are_independent():
    training, _, _, _, _ = data()
    normalized = normalize_records(records(training), policy())
    features = FieldPairFeatures().fit(normalized, **feature_controls())
    first = normalize_records([[record('cat', 'red bird'), record('cat', 'blue lake')]], policy())
    second = normalize_records([[record('cat', 'red bird'), record('river', 'blue lake')]], policy())
    a, b = features.transform(first)[0], features.transform(second)[0]
    assert a[1] == b[1] == 0 and a[4] == 1 and b[4] < 1
    np.testing.assert_array_equal(a[[0,1,2,3,5,6,7,12]], b[[0,1,2,3,5,6,7,12]])


def test_same_descriptions_different_titles_are_distinct_training_rows():
    train = records(data()[0][0:3])
    train += [[record('purple cloud', 'red bird'), record('wide river', 'bird red')]]
    rows = normalize_records(train, policy())
    features = FieldPairFeatures().fit(rows, **feature_controls())
    assert features.transform_training().shape == (4, 13)
    np.testing.assert_array_equal(features.transform_training()[:, 11], np.zeros(4))
    assert len(features.rows) == 4


def test_full_field_recipe_reproducible_and_symmetric():
    train, y, cal, cy, query = data()
    inputs = records(train), y, records(cal), cy, records(query)
    first = execute(*inputs, policy=policy(), **CONTROLS)
    assert first == execute(*inputs, policy=policy(), **CONTROLS)
    swapped = [pair[::-1] for pair in inputs[-1]]
    assert first == execute(*inputs[:-1], swapped, policy=policy(), **CONTROLS)
    assert first['feature_count'] == 13 and first['query_rows'] == 2


def test_overlap_uses_both_fields_after_normalization():
    train = [[record('colour', 'bird'), record('cat', 'lake')]]
    cal = [[record('cat', 'lake'), record('red', 'bird')]]
    query = [[record('tree', 'river'), record('cloud', 'green')]]
    with pytest.raises(ValueError, match='Overlapping'):
        normalize_populations(train, cal, query, policy())
    cal[0][1]['description'] = 'blue'
    assert len(normalize_populations(train, cal, query, policy())) == 3


@pytest.mark.parametrize('rows', [[], [[{'title':'cat'}, record('bird','tree')]],
    [[record('', 'bird'), record('cat', 'tree')]], [[record('cat','bird')]]])
def test_malformed_record_fields_reject(rows):
    with pytest.raises(ValueError):
        normalize_records(rows, policy())
