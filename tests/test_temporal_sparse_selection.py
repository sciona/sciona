from copy import deepcopy
import numpy as np
import pytest
from sciona.temporal_sparse_selection import fit_selected
from sciona.temporal_sparse_training import StreamEnsemble, fit
from tests.test_temporal_sparse_training import sample


def test_single_epoch_matches_existing_training():
    rows, encoder, controls = sample()
    selected = fit_selected(lambda: iter(rows[:30]), lambda: iter(rows[30:]),
        encoder_controls=encoder, controls=controls, max_epochs=1, patience=2, min_delta=0.)
    expected = fit(rows[:30], encoder_controls=encoder, controls=controls)
    assert selected.selected_epoch == 1 and not selected.stopped_early
    for a, b in zip(selected.model.models, expected.models):
        np.testing.assert_array_equal(a.coef_, b.coef_)
    assert selected.validation_scores == (expected.evaluate(rows[30:])['rmsle'],)


def test_restores_best_checkpoint_and_resets_causal_history(monkeypatch):
    rows, encoder, controls = sample()
    scores = iter([3., 2., 2., 4.])
    checkpoints = []
    real_evaluate = StreamEnsemble.evaluate

    def score(self, events):
        real_evaluate(self, events)
        checkpoints.append(deepcopy(self))
        return dict(rows=10, rmsle=next(scores))

    monkeypatch.setattr(StreamEnsemble, 'evaluate', score)
    result = fit_selected(lambda: iter(rows[:30]), lambda: iter(rows[30:]),
        encoder_controls=encoder, controls=controls, max_epochs=10, patience=2, min_delta=0.)
    assert result.stopped_early and result.selected_epoch == 2
    assert result.validation_scores == (3., 2., 2., 4.)
    assert result.model.updates_per_model == 10
    for a, b in zip(result.model.models, checkpoints[1].models):
        np.testing.assert_array_equal(a.coef_, b.coef_)
    # Replaying without resetting the encoder would fail time ordering or retain
    # future history; each epoch must end at the same causal history state.
    assert all(c.encoder.history == checkpoints[0].encoder.history for c in checkpoints)
    assert all(c.encoder.pending == checkpoints[0].encoder.pending for c in checkpoints)
    assert result.model is not checkpoints[1]


def test_no_dense_conversion_and_repeatability(monkeypatch):
    from scipy.sparse import csr_matrix
    def forbidden(*args, **kwargs):
        raise AssertionError('Dense population conversion forbidden')
    monkeypatch.setattr(csr_matrix, 'toarray', forbidden)
    rows, encoder, controls = sample()
    def run():
        return fit_selected(lambda: iter(rows[:30]), lambda: iter(rows[30:]),
            encoder_controls=encoder, controls=controls, max_epochs=3, patience=2, min_delta=0.)
    a, b = run(), run()
    assert a.validation_scores == b.validation_scores
    for x, y in zip(a.model.models, b.model.models):
        np.testing.assert_array_equal(x.coef_, y.coef_)


def test_replay_row_count_change_rejected():
    rows, encoder, controls = sample()
    populations = iter([rows[:30], rows[:29]])
    with pytest.raises(ValueError, match='population changed'):
        fit_selected(lambda: iter(next(populations)), lambda: iter(rows[30:]),
            encoder_controls=encoder, controls=controls, max_epochs=2, patience=2, min_delta=0.)
