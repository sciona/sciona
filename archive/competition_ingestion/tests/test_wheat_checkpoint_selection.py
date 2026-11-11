import pytest

from sciona.wheat_checkpoint_selection import PseudoCheckpointSelection


def test_first_round_keeps_first_strict_minimum_not_last_epoch():
    selection = PseudoCheckpointSelection(round_number=1)
    decisions = [selection.observe(i, v) for i, v in enumerate([5., 3., 3., 4., 2., 2., 3., 4., 5., 6.])]
    assert decisions == [True, True, False, False, True, False, False, False, False, False]
    assert selection.result()['selected_epoch'] == 4


def test_second_round_retains_inherited_checkpoint_without_improvement():
    selection = PseudoCheckpointSelection(round_number=2, inherited_best_loss=1.)
    assert not any(selection.observe(i, loss) for i, loss in enumerate([1., 2., 3., 1., 1., 2.]))
    assert selection.result() == dict(round_number=2, completed_epochs=6, selected_epoch=None,
                                      selected_validation_loss=1., selected_origin='inherited_round1')


def test_second_round_replaces_inherited_checkpoint_only_on_strict_improvement():
    selection = PseudoCheckpointSelection(round_number=2, inherited_best_loss=2.)
    for i, value in enumerate([3., 2., 1., 1., 4., 2.]):
        selection.observe(i, value)
    assert selection.result()['selected_epoch'] == 2
    assert selection.result()['selected_origin'] == 'current_round'


@pytest.mark.parametrize('epoch,loss', [(1, 1.), (0, float('nan')), (0, float('inf')), (0, -1.)])
def test_invalid_observation_does_not_advance_selection(epoch, loss):
    selection = PseudoCheckpointSelection(round_number=1)
    with pytest.raises(ValueError):
        selection.observe(epoch, loss)
    assert selection.completed_epochs == 0 and selection.selected_epoch is None
    with pytest.raises(ValueError, match='incomplete'):
        selection.result()
