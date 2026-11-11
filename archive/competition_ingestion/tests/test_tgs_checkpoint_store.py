import pytest
import torch

from sciona.tgs_checkpoint_store import CheckpointStore, persist_phase_result


def test_handoff_isolated_from_training_and_loaded_mutation(tmp_path):
    store = CheckpointStore(tmp_path)
    state = {'weight': torch.arange(12, dtype=torch.float32).reshape(3, 4),
             'counter': torch.tensor(7)}
    receipt = store.put(state)
    state['weight'].zero_()
    loaded = store.get(receipt)
    torch.testing.assert_close(loaded['weight'], torch.arange(12, dtype=torch.float32).reshape(3, 4))
    loaded['weight'].zero_()
    assert store.get(receipt)['weight'].sum() == 66
    assert store.get(receipt)['counter'].dtype == torch.int64


def test_corruption_rejected_before_deserialization(tmp_path):
    store = CheckpointStore(tmp_path)
    receipt = store.put({'weight': torch.ones(3)})
    path = tmp_path / (receipt['sha256'] + '.pt')
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match='digest or length'):
        store.get(receipt)


def test_only_tensor_states_and_valid_receipts(tmp_path):
    store = CheckpointStore(tmp_path)
    for state in ({}, {'weight': torch.tensor(float('nan'))}, {'weight': 'invalid'}):
        with pytest.raises(ValueError):
            store.put(state)
    with pytest.raises(ValueError, match='receipt'):
        store.get({'sha256': '../elsewhere', 'bytes': 1, 'tensor_count': 1})
    assert not list(tmp_path.iterdir())


def test_phase_preserves_best_and_cycle_end_as_distinct_states(tmp_path):
    store = CheckpointStore(tmp_path)
    a, b = {'weight': torch.tensor(1.)}, {'weight': torch.tensor(2.)}
    result = persist_phase_result(store, dict(best_state=a, periodic_states={40: b}, actual_epochs=40), branch='keras')
    assert 'best_state' not in result and 'periodic_states' not in result
    assert store.get(result['best_checkpoint'])['weight'] == 1
    assert store.get(result['periodic_checkpoints'][40])['weight'] == 2
    result = persist_phase_result(store, dict(cycle_states={50: a, 100: b}, actual_epochs=100), branch='pytorch')
    assert 'cycle_states' not in result
    assert store.get(result['cycle_checkpoints'][100])['weight'] == 2
