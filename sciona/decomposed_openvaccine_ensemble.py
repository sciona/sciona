"""Recipe-specific ensemble preparation around an external numeric reduction.

Preserve native forward/reverse averaging, unclipped Welford teacher moments
and the reconstructed twenty-member inventory. The caller binds the generic
ordered sum separately. This materializes twenty clipped predictions, trading
memory for an explicit graph boundary; it is not the historical winning blend.
"""
import gc
import re

import numpy as np

from sciona.openvaccine_checkpoints import load_checkpoint
from sciona.openvaccine_ensemble import EXPECTED_MEMBERS
from sciona.openvaccine_models import validate_inputs
from sciona.openvaccine_teacher import MemberCheckpoint


def collect_members(members, nodes, adjacency, *, std_ddof):
    nodes, adjacency = validate_inputs(nodes, adjacency)
    if type(std_ddof) is not int or std_ddof not in (0, 1):
        raise ValueError('Explicit integer standard deviation ddof required')
    seen = set()
    mean = np.zeros((nodes.shape[0], nodes.shape[1]-2, 5), dtype=np.float64)
    moment = np.zeros_like(mean)
    clipped = []
    reversed_nodes = nodes[:, ::-1, :].copy()
    reversed_adjacency = adjacency[:, ::-1, ::-1, :].copy()
    for family, slot, runtime in members:
        key = (family, slot)
        if (type(slot) is not int or key not in EXPECTED_MEMBERS
                or key in seen or runtime.family != family):
            raise ValueError('Invalid, duplicate or mismatched ensemble member')
        seen.add(key)
        forward = runtime.predict(nodes, adjacency)
        reverse = runtime.predict(reversed_nodes, reversed_adjacency)[:, ::-1, :]
        prediction = (forward + reverse)/2
        if prediction.shape != mean.shape or not np.isfinite(prediction).all():
            raise ValueError('Invalid reverse-averaged member prediction')
        value = prediction.astype(np.float64)
        delta = value-mean
        mean += delta/len(seen)
        moment += delta*(value-mean)
        clipped.append(np.clip(value, -.5, 6.))
    if seen != EXPECTED_MEMBERS:
        raise ValueError('Complete four-family five-state ensemble required')
    return dict(clipped_members=np.stack(clipped),
                teacher_mean=mean.astype(np.float32),
                teacher_std=np.sqrt(np.maximum(moment, 0)/(20-std_ddof)).astype(np.float32))


def collect_checkpoints(source_dir, checkpoints, nodes, adjacency, *, std_ddof):
    checkpoints = tuple(checkpoints)
    keys = []
    for record in checkpoints:
        if (not isinstance(record, MemberCheckpoint) or type(record.slot) is not int
                or not isinstance(record.manifest_sha256, str)
                or re.fullmatch('[0-9a-f]{64}', record.manifest_sha256) is None):
            raise ValueError('Invalid teacher checkpoint identity')
        keys.append((record.family, record.slot))
    if len(keys) != 20 or set(keys) != EXPECTED_MEMBERS:
        raise ValueError('Exactly twenty distinct reconstructed teacher members required')

    def members():
        for record in checkpoints:
            runtime = load_checkpoint(source_dir, record.directory,
                expected_manifest_sha256=record.manifest_sha256, expected_family=record.family)
            yield record.family, record.slot, runtime
            del runtime
            gc.collect()

    return collect_members(members(), nodes, adjacency, std_ddof=std_ddof)


def checked_members(collected):
    if type(collected) is not dict or not {'clipped_members', 'teacher_mean', 'teacher_std'} <= set(collected):
        raise ValueError('Same-invocation collected ensemble required')
    values = np.asarray(collected['clipped_members'])
    if (values.ndim != 4 or values.shape[0] != 20 or values.shape[1] < 1
            or values.shape[2] < 2 or values.shape[3] != 5 or values.dtype != np.float64
            or not np.isfinite(values).all() or (values < -.5).any() or (values > 6.).any()):
        raise ValueError('Twenty finite clipped member arrays required')
    for name in ('teacher_mean', 'teacher_std'):
        array = np.asarray(collected[name])
        if (array.shape != values.shape[1:] or array.dtype != np.float32
                or not np.isfinite(array).all()
                or (name == 'teacher_std' and (array < 0).any())):
            raise ValueError('Finite aligned unclipped teacher moments required')
    return values


def blend_inputs(collected):
    """Flatten ordered members and supply unit coefficients; do not sum here."""
    values = checked_members(collected)
    rows = values.reshape(20, -1).copy()
    # The reference starts from positive zero, while the generic reducer starts
    # from the first product. Preserve even an all-negative-zero ensemble.
    rows[0] = np.zeros_like(rows[0]) + rows[0]
    return rows, np.ones(20, dtype=np.float64)


def finish_prediction(collected, total):
    """Restore shape and divide the same-invocation external sum by twenty.

    Shape/finite checks cannot authenticate that the caller supplied the sum
    from this invocation. The execution graph must enforce that connection.
    """
    values = checked_members(collected)
    total = np.asarray(total)
    if (total.shape != (values[0].size,) or total.dtype != np.float64
            or not np.isfinite(total).all() or (total < -10.).any() or (total > 120.).any()):
        raise ValueError('Finite aligned sum of twenty clipped members required')
    return (total.reshape(values.shape[1:])/20).astype(np.float32)
