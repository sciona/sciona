"""Synthetic contracts for explicitly corrected feature adapters."""
import numpy as np
import pandas as pd
import pytest

from sciona.nasa_feature_adapters import (estimated_departure_features,
    fit_airline_vocabulary, airline_features, arrival_history_features)

T = pd.Timestamp('2000-01-01')


def test_etd_includes_singleton_and_final_entity_with_strict_bounds():
    queries = pd.DataFrame({'gufi': ['AAA.X.ZZZ', 'BBB.X.ZZZ'], 'timestamp': [T, T]})
    estimates = pd.DataFrame([
        dict(gufi=g, timestamp=T-pd.Timedelta(minutes=age),
             departure_runway_estimated_time=T+pd.Timedelta(minutes=remaining))
        for g in queries.gufi for age, remaining in [(5, 20), (10, 10), (0, 99), (1, 0)]])
    result = estimated_departure_features(queries, estimates)
    assert list(result.gufi) == list(queries.gufi)
    np.testing.assert_array_equal(result.found_etd_vals, [2, 2])
    np.testing.assert_array_equal(result.minutes_until_departure_from_timepoint, [20, 20])
    np.testing.assert_array_equal(result.mean_departure_from_timepoint, [15, 15])
    assert len(estimated_departure_features(queries.iloc[:1], estimates)) == 1


def test_etd_missing_history_has_explicit_empty_columns():
    q = pd.DataFrame({'gufi': ['AAA.X.ZZZ'], 'timestamp': [T]})
    e = q.assign(departure_runway_estimated_time=T)
    result = estimated_departure_features(q, e)
    assert result.empty and 'found_etd_vals' in result
    with pytest.raises(ValueError, match='Unique entity'):
        estimated_departure_features(pd.concat([q, q]), e)


def test_category_fit_deduplicates_and_breaks_ties_lexically():
    labels = pd.DataFrame({'gufi': ['BBB.X.ZZZ', 'AAA.X.ZZZ', 'BBB.X.ZZZ']})
    assert fit_airline_vocabulary(labels, 1) == ('AAA',)


def test_category_other_always_present_and_inference_uses_fixed_vocabulary():
    labels = pd.DataFrame({'gufi': ['AAA.X.ZZZ', 'BBB.X.ZZZ'], 'airport': ['KZZZ']*2})
    vocabulary = fit_airline_vocabulary(labels)
    before = vocabulary
    features = airline_features(labels, vocabulary)
    assert features.Other.sum() == 0
    unseen = pd.DataFrame({'gufi': ['CCC.X.ZZZ'], 'airport': ['KZZZ']})
    encoded = airline_features(unseen, vocabulary)
    assert encoded.Other.iloc[0] == 1 and encoded[['AAA', 'BBB']].sum().sum() == 0
    assert vocabulary == before and list(encoded) == list(features)


def history(future=False):
    stands, arrivals = [], []
    for i, (age, duration) in enumerate([(10, 10), (70, 20), (130, 30)]):
        actual = T-pd.Timedelta(minutes=age)
        observed = T+pd.Timedelta(minutes=1) if future else actual
        stands.append(dict(gufi=f'N{i:02}.X.ZZZ', timestamp=observed, arrival_stand_actual_time=actual))
        arrivals.append(dict(gufi=f'N{i:02}.X.ZZZ', timestamp=observed,
                             arrival_runway_actual_time=actual-pd.Timedelta(minutes=duration)))
    return pd.DataFrame(stands), pd.DataFrame(arrivals)


def test_history_reuses_one_shot_window_and_excludes_future_observations():
    s, a = history()
    result = arrival_history_features([T], s, a, 'ZZZ')
    np.testing.assert_array_equal(result.iloc[:, 1:].to_numpy(), [[2, 15, 5]])
    sf, af = history(True)
    assert arrival_history_features([T], sf, a, 'ZZZ').empty
    assert arrival_history_features([T], s, af, 'ZZZ').empty
    assert arrival_history_features([T], sf, af, 'ZZZ').empty


def test_history_observations_at_query_are_excluded():
    s, a = history()
    s['timestamp'] = T
    assert arrival_history_features([T], s, a, 'ZZZ').empty


def test_history_latest_available_revisions_do_not_multiply_rows():
    s, a = history()
    revision = a.iloc[:1].copy()
    revision['timestamp'] = T-pd.Timedelta(minutes=1)
    revision['arrival_runway_actual_time'] -= pd.Timedelta(minutes=10)
    revised = pd.concat([a, revision], ignore_index=True)
    result = arrival_history_features([T], s, revised, 'ZZZ')
    np.testing.assert_array_equal(result.iloc[:, 1:].to_numpy(), [[2, 20, 0]])
    revision['timestamp'] = T+pd.Timedelta(minutes=1)
    result = arrival_history_features([T], s, pd.concat([a, revision]), 'ZZZ')
    np.testing.assert_array_equal(result.iloc[:, 1:].to_numpy(), [[2, 15, 5]])


def test_history_rejects_ambiguous_revision_and_negative_duration():
    s, a = history()
    with pytest.raises(ValueError, match='Ambiguous'):
        arrival_history_features([T], pd.concat([s, s.iloc[:1]]), a, 'ZZZ')
    a.loc[0, 'arrival_runway_actual_time'] = s.loc[0, 'arrival_stand_actual_time']+pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match='Negative'):
        arrival_history_features([T], s, a, 'ZZZ')


@pytest.mark.parametrize('bad', ['missing', 'timezone', 'text'])
def test_timestamp_contract_rejects_ambiguous_inputs(bad):
    s, a = history()
    if bad == 'missing':
        s.loc[0, 'timestamp'] = pd.NaT
    elif bad == 'timezone':
        s['timestamp'] = s.timestamp.dt.tz_localize('UTC')
    else:
        s['timestamp'] = s.timestamp.astype(str)
    with pytest.raises(ValueError):
        arrival_history_features([T], s, a, 'ZZZ')


def test_history_destination_filter():
    s, a = history()
    assert arrival_history_features([T], s, a, 'YYY').empty
