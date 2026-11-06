"""Complete unchanged training recipe with final cut indexing exposed as dataflow.

All preprocessing, model families, budgets, augmentation, checkpoint replay
and temporary-file cleanup follow the qualified reference. The four cut
points and equality policy are recipe inputs to a separate generic provider.
"""
import gc
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import torch

from sciona.aptos_pipeline import _predict, _batches, _digest, _restore
from sciona.aptos_ensemble import MODEL_KEYS, CUT_POINTS, _keys, blend, classify
from sciona.aptos_initialization import build_from_checkpoint
from sciona.aptos_models import FAMILIES
from sciona.aptos_population import prepare_population, first_stage_targets, second_stage_targets
from sciona.aptos_training import train_epochs


def train_scores(*, base, average, grouped, pseudo_keys, query_keys, images,
                  pretrained, first_stage_epochs, seeds, batch_size, learning_rate,
                  lower_deviation, upper_deviation, tie_policy, work_root):
    """Train both stages, verify checkpoint replay and return scores before cut indexing.

    External images/identities are private runtime inputs and never logged.
    Temporary trained checkpoints are deleted on success or failure. Queries
    may overlap the pseudo-label role, but never the supplied-label roles.
    This explicitly supports the source's transductive pseudo-labeling scope.
    """
    population = prepare_population(base=base, average=average, grouped=grouped, pseudo_keys=pseudo_keys)
    queries = _keys(query_keys)
    labeled = population.base_keys + population.average_keys + population.group_keys
    if set(queries) & set(labeled):
        raise ValueError('Queries must not overlap supplied-label roles')
    if not isinstance(images, dict) or set(images) != set(population.all_keys) | set(queries):
        raise ValueError('Exactly the complete population and query images required')
    for value in images.values():
        if (not isinstance(value, np.ndarray) or value.dtype != np.uint8 or value.ndim != 3
                or value.shape[2] != 3 or min(value.shape[:2]) < 1):
            raise ValueError('Nonempty RGB uint8 runtime images required')
    for mapping in (first_stage_epochs, seeds):
        if (not isinstance(mapping, dict) or set(mapping) != set(MODEL_KEYS)
                or any(not isinstance(k, tuple) or len(k) != 2 or type(k[1]) is not int for k in mapping)):
            raise ValueError('Exactly eight architecture/replica configurations required')
    if any(type(n) is not int or n < 5 or n % 5 for n in first_stage_epochs.values()):
        raise ValueError('First-stage budgets must be positive five-epoch increments')
    if any(type(n) is not int or not 0 <= n < 2**32 for n in seeds.values()) or len(set(seeds.values())) != 8:
        raise ValueError('Eight distinct nonnegative 32-bit seeds required')
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('Positive integer batch size required')
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)) or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('Finite positive learning rate required')
    if not isinstance(pretrained, dict) or set(pretrained) != set(FAMILIES):
        raise ValueError('All four explicit pretrained artifact paths and digests required')
    # Validate refinement bounds and tie policy before expensive training.
    added = population.average_keys + population.group_keys + population.pseudo_keys
    second_stage_targets(population, (added, np.zeros(len(added))),
                         lower_deviation=lower_deviation, upper_deviation=upper_deviation)
    classify(np.zeros(1), tie_policy=tie_policy)
    root = Path(work_root)
    if not root.is_dir():
        raise ValueError('Existing private temporary-work root required')
    stage1, stage2, checkpoints, fits = {}, {}, {}, []
    base_keys, base_values = first_stage_targets(population)
    with TemporaryDirectory(prefix='aptos-training-', dir=root) as directory:
        directory = Path(directory)
        for index, key in enumerate(MODEL_KEYS):
            family, replica = key
            torch.manual_seed(seeds[key])
            path, digest = pretrained[family]
            model = build_from_checkpoint(family, path, expected_sha256=digest, checkpoint_format='safetensors')
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            losses = train_epochs(model, optimizer, _batches(base_keys, base_values, images, family, seeds[key], batch_size),
                                  epochs=first_stage_epochs[key])
            steps = max((int(state['step'].item()) for state in optimizer.state.values()), default=0)
            predictions = _predict(model, family, added, images, batch_size)
            checkpoint = directory / f'stage1_{index}.pt'
            torch.save(model.state_dict(), checkpoint)
            digest = _digest(checkpoint)
            del optimizer, model
            gc.collect()
            restored = _restore(family, checkpoint, digest)
            np.testing.assert_array_equal(predictions, _predict(restored, family, added, images, batch_size))
            del restored
            gc.collect()
            stage1[key] = (added, predictions)
            checkpoints[key] = (checkpoint, digest)
            fits.append({'stage': 1, 'family': family, 'replica': replica,
                         'epochs': first_stage_epochs[key], 'optimizer_steps': steps, 'losses': losses})
        teacher = blend(added, stage1)
        keys, values = second_stage_targets(population, (added, teacher),
            lower_deviation=lower_deviation, upper_deviation=upper_deviation)
        for index, key in enumerate(MODEL_KEYS):
            family, replica = key
            seed = seeds[key] + 2**32
            torch.manual_seed(seed)
            model = _restore(family, *checkpoints[key])
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            losses = train_epochs(model, optimizer, _batches(keys, values, images, family, seed, batch_size), epochs=10)
            steps = max((int(state['step'].item()) for state in optimizer.state.values()), default=0)
            predictions = _predict(model, family, queries, images, batch_size)
            checkpoint = directory / f'stage2_{index}.pt'
            torch.save(model.state_dict(), checkpoint)
            digest = _digest(checkpoint)
            del optimizer, model
            gc.collect()
            restored = _restore(family, checkpoint, digest)
            np.testing.assert_array_equal(predictions, _predict(restored, family, queries, images, batch_size))
            del restored
            gc.collect()
            stage2[key] = (queries, predictions)
            fits.append({'stage': 2, 'family': family, 'replica': replica, 'epochs': 10,
                         'optimizer_steps': steps, 'losses': losses})
        scores = blend(queries, stage2)
    return {'scores': scores, 'tie_policy': tie_policy, 'fits': fits,
            'first_stage_predictions': stage1, 'second_stage_predictions': stage2,
            'second_stage_targets': values, 'checkpoint_replays_exact': True,
            'temporary_checkpoints_removed': not directory.exists()}


def cut_inputs(training_result):
    """Expose the recipe's scores, fixed cuts and explicit caller equality policy."""
    scores=training_result['scores']
    policy=training_result['tie_policy']
    if (not isinstance(scores,np.ndarray) or scores.ndim!=1 or not scores.size
            or scores.dtype!=np.dtype('float64') or not np.isfinite(scores).all()):
        raise ValueError('Nonempty finite float64 recipe scores required')
    if not isinstance(policy,str) or policy not in ('upper','lower'):
        raise ValueError('Explicit upper or lower tie policy required')
    return scores.copy(),np.asarray(CUT_POINTS,dtype=np.float64),policy


def result(training_result,ordinals):
    """Attach externally computed classes to the complete training result.

    Both inputs must originate from the same graph invocation. This stage
    checks representation and bounds, not training provenance or calibration.
    """
    scores,_,_=cut_inputs(training_result)
    if (not isinstance(ordinals,np.ndarray) or ordinals.shape!=scores.shape
            or ordinals.dtype.kind not in 'iu' or np.any(ordinals<0) or np.any(ordinals>len(CUT_POINTS))):
        raise ValueError('Aligned integer class indices within the recipe bounds required')
    expected={'scores','tie_policy','fits','first_stage_predictions','second_stage_predictions',
        'second_stage_targets','checkpoint_replays_exact','temporary_checkpoints_removed'}
    if set(training_result)!=expected:
        raise ValueError('Complete training result fields required')
    output={k:v for k,v in training_result.items() if k!='tie_policy'}
    output['ordinals']=ordinals.copy()
    return output
