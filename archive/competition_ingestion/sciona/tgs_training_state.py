"""Explicit optimizer, schedule and snapshot contracts for TGS training.

Modern CPU realization. Keras RMSprop formula checked against Keras 2.2.4
optimizers.py; this reference does not establish the winner's installed version.
"""
import copy
import math
import torch


def optimizer_for(model, branch, learning_rate):
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('positive finite learning rate required')
    if branch == 'keras':
        return torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=.9,
                                   eps=1e-7, weight_decay=0, momentum=0, centered=False, foreach=False)
    if branch == 'pytorch':
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=.9,
                               weight_decay=1e-4, nesterov=False, foreach=False)
    raise ValueError('unknown source branch')


def snapshot_learning_rate(epoch, total_epochs, snapshots, maximum):
    """Keras source epoch-start cosine; no added minimum learning rate."""
    for value in (epoch, total_epochs, snapshots):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError('integer schedule controls required')
    if not 0 <= epoch < total_epochs or not 1 <= snapshots <= total_epochs:
        raise ValueError('invalid epoch/snapshot bounds')
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError('positive maximum learning rate required')
    period = total_epochs // snapshots
    return maximum * (1 + math.cos(math.pi * (epoch % period) / period)) / 2


def capture_training_state(model, optimizer):
    """Detached CPU copies; later updates cannot mutate a selected snapshot."""
    return dict(model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                optimizer=copy.deepcopy(optimizer.state_dict()),
                torch_rng=torch.get_rng_state().clone())


def restore_training_state(model, optimizer, state):
    """Restore an internally captured state, including momentum and CPU RNG.

    This is not an untrusted checkpoint loader. Architecture and parameter
    ordering must be the same as at capture time.
    """
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
    torch.set_rng_state(state['torch_rng'].clone())
