"""Post-completion consistency receipt for the fixed synthetic B0 training probe.

This cannot recover a missing training-time checkpoint digest. It records the
observed checkpoint bytes and independently replays the selected epoch's
evaluation. It does not establish font recognition or catalog eligibility.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from efficientnet_pytorch import EfficientNet

from sciona.bengali_checkpoint_handoff import load_frozen_classifier
from sciona.bengali_font_classifier import BengaliFontClassifier
from sciona.bengali_preprocessing import prepare_handwriting
from sciona.bengali_sampling import SamplingBudget, paired_batches


LAUNCHER = 'scripts/run_bengali_classifier_training_probe.py'
LAUNCHER_SHA256 = 'e9c75a5a6c5d9f40dd9532ae24efd9dfd87000b0245f0a7ee678aa228852d148'
MODULES = ('classifier_fit', 'font_classifier', 'population', 'preprocessing',
           'pretraining_sampling', 'pretraining_augmentation', 'font_sampling',
           'font_augmentation', 'shear_sampling', 'sampling', 'joint_labels',
           'label_corrections')
BOUND_FILES = {LAUNCHER} | {'sciona/bengali_' + name + '.py' for name in MODULES}


def validate_completion(report, history, root):
    """Reject incomplete or altered evidence before loading any model weights."""
    for key in ('passed', 'synthetic_only', 'full_efficientnet_b0',
                'source_best_checkpoint_present', 'fresh_checkpoint_output_exact'):
        if report.get(key) is not True:
            raise ValueError('classifier completion evidence missing: ' + key)
    if report.get('approved') is not False or report.get('catalog_mutations') != 0:
        raise ValueError('expected an unapproved synthetic execution report')
    for key, expected in [('epochs', 60), ('batch_size', 32), ('optimizer_updates', 120)]:
        if type(report.get(key)) is not int or report[key] != expected:
            raise ValueError('classifier source budget differs: ' + key)
    bindings = dict(report.get('implementation_sha256', {}))
    # Python may expose __file__ as absolute even for a relative CLI launch.
    # Accept only this exact launcher alias, without weakening the inventory.
    absolute_launcher = str((root / LAUNCHER).resolve())
    if absolute_launcher in bindings:
        if LAUNCHER in bindings:
            raise ValueError('duplicate classifier launcher binding')
        bindings[LAUNCHER] = bindings.pop(absolute_launcher)
    if set(bindings) != BOUND_FILES or bindings.get(LAUNCHER) != LAUNCHER_SHA256:
        raise ValueError('classifier implementation binding inventory differs')
    for name, digest in bindings.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError('classifier implementation changed: ' + name)
    if not isinstance(history, list) or len(history) != 60:
        raise ValueError('incomplete classifier epoch history')
    best_epoch = None
    best_score = 0.
    for epoch, row in enumerate(history):
        if type(row.get('epoch')) is not int or row['epoch'] != epoch:
            raise ValueError('classifier epoch sequence differs')
        loss, accuracy = row.get('training_loss'), row.get('same_population_accuracy')
        if (type(loss) not in (int, float) or not math.isfinite(loss) or loss < 0
                or type(accuracy) not in (int, float) or not math.isfinite(accuracy)
                or not 0 <= accuracy <= 1 or accuracy * 64 != int(accuracy * 64)):
            raise ValueError('invalid classifier history metrics')
        if accuracy > best_score:
            best_epoch, best_score = epoch, accuracy
    if best_epoch is None:
        raise ValueError('source strict-improvement rule did not select a checkpoint')
    if report.get('same_population_accuracy') != history[-1]['same_population_accuracy']:
        raise ValueError('completion report and final history disagree')
    return best_epoch, best_score


def main(runtime, completion, output):
    root = Path(__file__).resolve().parents[1]
    report_bytes = completion.read_bytes()
    history_bytes = (runtime / 'history.json').read_bytes()
    report, history = json.loads(report_bytes), json.loads(history_bytes)
    epoch, expected_accuracy = validate_completion(report, history, root)
    # This recipe is synthetic and fixed by the pinned launcher above.
    rng = np.random.default_rng(817)
    labels = np.arange(64, dtype=np.int64) % 8
    images = np.empty((64, 137, 236), dtype=np.uint8)
    for index, label in enumerate(labels):
        images[index] = np.clip(24 + int(label) * 27 + rng.integers(0, 12, (137, 236)), 0, 255).astype(np.uint8)
    checkpoint = runtime / 'best.pt'
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    torch.set_num_threads(2)
    model = BengaliFontClassifier(EfficientNet.from_name('efficientnet-b0'))
    load_frozen_classifier(model, checkpoint, expected_sha256=digest)
    correct = count = 0
    with torch.no_grad():
        for batch in paired_batches(SamplingBudget(64, 64, 32, 60), hand_seed=831, font_seed=832):
            if batch['epoch'] != epoch:
                continue
            indices = batch['font_indices'].numpy()
            logits = model(prepare_handwriting(images[indices]))
            if logits.shape != (32, 14784) or not torch.isfinite(logits).all():
                raise ValueError('checkpoint evaluation output contract differs')
            correct += int((logits.argmax(1) == torch.from_numpy(labels[indices])).sum())
            count += len(indices)
    if count != 64 or correct / count != expected_accuracy:
        raise ValueError('best checkpoint does not reproduce selected epoch accuracy')
    # Reject files changed during replay; the returned digest binds future loads.
    if (completion.read_bytes() != report_bytes or (runtime / 'history.json').read_bytes() != history_bytes
            or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != digest):
        raise ValueError('classifier evidence changed during qualification')
    validate_completion(report, history, root)
    receipt = dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
        receipt_kind='post_completion_synthetic_consistency', checkpoint_sha256=digest,
        completion_sha256=hashlib.sha256(report_bytes).hexdigest(),
        history_sha256=hashlib.sha256(history_bytes).hexdigest(),
        selected_epoch=epoch, same_population_accuracy=expected_accuracy,
        selected_epoch_evaluation_replayed=True, training_time_checkpoint_digest_available=False,
        classifier_implementation_sha256=report['implementation_sha256'],
        verifier_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [Path(__file__).resolve(), root / 'sciona/bengali_checkpoint_handoff.py']},
        limits=['Post-completion observation, not a recovered training-time checkpoint attestation.',
                'Same-population synthetic intensity classification; no font recognition or held-out accuracy claim.',
                'Does not qualify complete Bengali pipeline, dependencies, licensing or publication.'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        handle.write(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if not k.endswith('_sha256')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime-directory', type=Path, required=True)
    parser.add_argument('--completion-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.runtime_directory, args.completion_report, args.output)
