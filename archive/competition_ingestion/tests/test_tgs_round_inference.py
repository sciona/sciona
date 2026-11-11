import numpy as np
import pytest
import torch

from sciona.tgs_checkpoint_selection import ensemble_selection
from sciona.tgs_checkpoint_store import CheckpointStore
from sciona.tgs_round_inference import round_receipts, predict_round


def fits(stage, receipt):
    selection = ensemble_selection(stage)
    result = {item['fit']: {'best_checkpoint': receipt} for group in selection['keras'] for item in group}
    for item in selection['pytorch']:
        result[item['fit']] = {'cycle_checkpoints': {str(n): receipt for n in range(50, 301 if stage == 1 else 201, 50)}}
    return result


def test_missing_unselected_cycle_cannot_shift_snapshot_identity():
    completed = fits(1, {})
    del completed['torch.p0.f0']['cycle_checkpoints']['50']
    with pytest.raises(ValueError, match='missing or duplicated'):
        round_receipts(1, completed)


def test_json_cycle_keys_and_all_round_counts():
    for stage, count in [(1, 25), (2, 20), (3, 18)]:
        selected = round_receipts(stage, fits(stage, {}))
        assert sum(map(len, selected['keras'])) == 20
        assert sum(map(len, selected['pytorch'])) == count


class KerasOracle(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.))
    def forward(self, images):
        return torch.ones_like(images[:, :1]) * self.bias.sigmoid()


class TorchOracle(KerasOracle):
    variant = 5
    def forward(self, images):
        return images[:, :1] + self.bias


def test_complete_round_reads_persisted_weights_and_matches_analytic_blend(tmp_path):
    store = CheckpointStore(tmp_path)
    receipt = store.put({'bias': torch.tensor(0.)})
    values = np.broadcast_to(np.linspace(0, 1, 101, dtype=np.float32), (1, 101, 101)).copy()
    rgb = np.repeat((values * 255)[..., None], 3, axis=-1)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = predict_round(3, fits(3, receipt), store, KerasOracle(), TorchOracle(),
                               rgb, values, np.array([False]), keras_batch_size=1, torch_batch_size=1)
    finally:
        torch.set_num_threads(previous)
    probability = torch.from_numpy(values).sigmoid().numpy().astype(np.float64)
    normalized = (probability - probability.min()) / (probability.max() - probability.min())
    expected = (np.float32(127 / 255) + normalized) / 2
    np.testing.assert_allclose(result['scores'], expected, atol=1e-7)
    assert result['checkpoint_counts'] == {'keras': 20, 'pytorch': 18}
