"""Lifecycle checks with synthetic models; full B7 qualification is separate."""
import hashlib

import pytest
import torch
from torch import nn

from sciona.grapheme_b7_fit import fit_classifier


class Classifier(nn.Module):
    branch = 'ood'

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 1295)

    def forward(self, images):
        return self.fc(images.mean((2, 3)))


def setup(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(71)
    model = Classifier()
    images = torch.randn(1, 3, 137, 236)
    options = dict(branch='ood', epochs=3, steps_per_epoch=2,
                   batches_for_epoch=lambda epoch: [(images, torch.tensor([0]))] * 2,
                   validate=lambda model, epoch: [0., 1., .5][epoch], warmup_steps=2,
                   first_schedule_index=0, checkpoint_policy='max_validation',
                   output_directory=tmp_path/'fit')
    return model, options


@pytest.mark.parametrize('policy,selected_epoch', [('last', 2), ('max_validation', 1)])
def test_exact_lifecycle_schedule_validation_and_checkpoint(tmp_path, policy, selected_epoch):
    model, options = setup(tmp_path)
    snapshots = []
    def validate(model, epoch):
        assert not model.training and not torch.is_grad_enabled()
        snapshots.append({key: value.clone() for key, value in model.state_dict().items()})
        return [0., 1., .5][epoch]
    options.update(checkpoint_policy=policy, validate=validate)
    result = fit_classifier(model, **options)
    assert result['completed_steps'] == 6 and result['completed_epochs'] == 3
    assert [h['last_learning_rate'] for h in result['history']] == [.0005, .00075, .00025]
    assert result['selected_checkpoint']['epoch'] == selected_epoch
    path = tmp_path/'fit'/'selected.pt'
    state = torch.load(path, weights_only=True)
    for key, expected in snapshots[selected_epoch].items():
        torch.testing.assert_close(state[key], expected, rtol=0, atol=0)
    assert result['selected_checkpoint']['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert not result['approved'] and not result['resumable_checkpoint']


@pytest.mark.parametrize('count', [1, 3])
def test_wrong_epoch_cardinality_cannot_return_success_or_save_checkpoint(tmp_path, count):
    model, options = setup(tmp_path)
    original = options['batches_for_epoch']
    options['batches_for_epoch'] = lambda epoch: original(epoch)[:1] * count
    with pytest.raises(ValueError, match='declared batch count'):
        fit_classifier(model, **options)
    assert not (tmp_path/'fit'/'selected.pt').exists()


@pytest.mark.parametrize('score', [float('nan'), float('inf'), True])
def test_invalid_validation_cannot_save_checkpoint(tmp_path, score):
    model, options = setup(tmp_path)
    options['validate'] = lambda model, epoch: score
    with pytest.raises(ValueError, match='validation score'):
        fit_classifier(model, **options)
    assert not (tmp_path/'fit'/'selected.pt').exists()


def test_invalid_schedule_rejected_before_files_or_model_changes(tmp_path):
    model, options = setup(tmp_path)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    options['warmup_steps'] = 6
    with pytest.raises(ValueError):
        fit_classifier(model, **options)
    assert not (tmp_path/'fit').exists()
    for key, expected in before.items():
        torch.testing.assert_close(model.state_dict()[key], expected, rtol=0, atol=0)
