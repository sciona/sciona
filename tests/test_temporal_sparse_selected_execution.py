from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import pytest
from scripts.temporal_sparse_synthetic import payload
from sciona import temporal_sparse_selected_execution as execution
from sciona.temporal_sparse_selection import fit_selected
from sciona.temporal_sparse_source import events


def recipe(tmp_path):
    result = payload(tmp_path)
    result['selection'] = dict(max_epochs=4, patience=2, min_delta=0.)
    return result


def forbidden(*args, **kwargs):
    raise AssertionError('Training must not start')


def test_full_lifecycle_matches_selected_model_and_is_private(tmp_path):
    config = recipe(tmp_path)
    prepared = execution.prepare(config)
    result = execution.execute(prepared)
    assert result == execution.execute(prepared)
    def read(role):
        with open(config['sources'][role], 'rb') as stream:
            yield from events(stream, labeled=role != 'query', max_line_bytes=4096)
    selected = fit_selected(lambda: read('training'), lambda: read('validation'),
        encoder_controls=config['encoder_controls'], controls=config['controls'], **config['selection'])
    expected = [float(v) for batch, _ in selected.model.predict_batches(read('query')) for v in batch]
    assert result['predictions'] == expected
    assert result['selection_validation']['scores'] == list(selected.validation_scores)
    assert result['selection_validation']['rmsle'] == selected.validation_scores[selected.selected_epoch - 1]
    assert result['selected_updates_per_model'] == 5 * result['selected_epoch']
    assert result['training_rows'] == 30 and result['query_rows'] == 2
    assert result['selection_validation']['used_for_selection'] is True
    serialized = json.dumps(result, allow_nan=False) + repr(prepared)
    assert str(tmp_path) not in serialized
    assert all(s.sha256 not in serialized for s in prepared.source.sources)


@pytest.mark.parametrize('role', ['training', 'validation', 'query'])
def test_mutation_before_execution_rejected_before_fit(tmp_path, monkeypatch, role):
    config = recipe(tmp_path)
    prepared = execution.prepare(config)
    with open(config['sources'][role], 'a') as stream:
        stream.write('\n')
    monkeypatch.setattr(execution, 'fit_selected', forbidden)
    with pytest.raises(ValueError, match='changed'):
        execution.execute(prepared)


def test_live_sources_cannot_change_snapshot_replays(tmp_path, monkeypatch):
    config = recipe(tmp_path)
    prepared = execution.prepare(config)
    expected = execution.execute(prepared)
    original = execution.fit_selected
    def mutate_then_fit(*args, **kwargs):
        for path in config['sources'].values():
            Path(path).write_text('invalid replacement\n')
        return original(*args, **kwargs)
    monkeypatch.setattr(execution, 'fit_selected', mutate_then_fit)
    assert execution.execute(prepared) == expected


def test_real_patience_stop_restores_first_epoch_predictions(tmp_path):
    config = recipe(tmp_path)
    config['selection'] = dict(max_epochs=8, patience=2, min_delta=1e6)
    stopped = execution.execute(execution.prepare(config))
    config['selection']['max_epochs'] = 1
    first = execution.execute(execution.prepare(config))
    assert stopped['stopped_early'] and stopped['evaluated_epochs'] == 3
    assert stopped['selected_epoch'] == 1
    assert stopped['predictions'] == first['predictions']
    assert stopped['selected_updates_per_model'] == first['selected_updates_per_model'] == 5


def test_snapshots_close_on_training_failure(tmp_path, monkeypatch):
    original = execution.snapshot
    streams = []
    @contextmanager
    def tracked(source):
        with original(source) as stream:
            streams.append(stream)
            yield stream
    monkeypatch.setattr(execution, 'snapshot', tracked)
    monkeypatch.setattr(execution, 'fit_selected', forbidden)
    with pytest.raises(AssertionError):
        execution.execute(execution.prepare(recipe(tmp_path)))
    assert len(streams) == 3 and all(s.closed for s in streams)


@pytest.mark.parametrize('change', ['summary', 'limit', 'selection'])
def test_forged_preparation_rejected_before_fit(tmp_path, monkeypatch, change):
    prepared = execution.prepare(recipe(tmp_path))
    if change == 'selection':
        prepared = replace(prepared, selection='{}')
    else:
        source = prepared.source.sources[2]
        source = replace(source, **({'rows': 1} if change == 'summary' else {'max_bytes': -1}))
        prepared = replace(prepared, source=replace(prepared.source, sources=prepared.source.sources[:2] + (source,)))
    monkeypatch.setattr(execution, 'fit_selected', forbidden)
    with pytest.raises(ValueError):
        execution.execute(prepared)


@pytest.mark.parametrize('selection', [dict(max_epochs=True, patience=1, min_delta=0.),
    dict(max_epochs=2, patience=0, min_delta=0.), dict(max_epochs=2, patience=1, min_delta=float('inf'))])
def test_bad_selection_rejected_before_read(tmp_path, monkeypatch, selection):
    config = recipe(tmp_path)
    config['selection'] = selection
    monkeypatch.setattr(execution.base, 'prepare', forbidden)
    with pytest.raises(ValueError):
        execution.prepare(config)
