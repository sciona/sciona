import hashlib
import json
from pathlib import Path

import pytest

from scripts.qualify_bengali_classifier_checkpoint import BOUND_FILES, LAUNCHER, validate_completion
from scripts.run_bengali_trained_guidance_probe import validate_handoff


@pytest.fixture
def evidence(tmp_path):
    root = Path(__file__).resolve().parents[1]
    bindings = {}
    for name in BOUND_FILES:
        payload = (root / name).read_bytes()
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        bindings[name] = hashlib.sha256(payload).hexdigest()
    report = dict(passed=True, synthetic_only=True, approved=False, catalog_mutations=0,
                  full_efficientnet_b0=True, source_best_checkpoint_present=True,
                  fresh_checkpoint_output_exact=True, epochs=60, batch_size=32,
                  optimizer_updates=120, implementation_sha256=bindings,
                  same_population_accuracy=.5)
    history = [dict(epoch=e, training_loss=1., same_population_accuracy=.5) for e in range(60)]
    history[3]['same_population_accuracy'] = .75
    history[7]['same_population_accuracy'] = .75
    return report, history, tmp_path


def test_selection_retains_first_strict_best_not_last_or_tied_epoch(evidence):
    assert validate_completion(*evidence) == (3, .75)


def test_python_absolute_launcher_binding_is_same_exact_file(evidence):
    report, history, root = evidence
    bindings = report['implementation_sha256']
    bindings[str((root / LAUNCHER).resolve())] = bindings.pop(LAUNCHER)
    assert validate_completion(report, history, root) == (3, .75)


def test_duplicate_launcher_alias_is_rejected(evidence):
    report, history, root = evidence
    bindings = report['implementation_sha256']
    bindings[str((root / LAUNCHER).resolve())] = bindings[LAUNCHER]
    with pytest.raises(ValueError, match='duplicate'):
        validate_completion(report, history, root)


@pytest.mark.parametrize('corruption', [
    'incomplete', 'epoch_order', 'no_best', 'nonfinite', 'invalid_accuracy',
    'report_disagrees', 'missing_binding', 'modified_code', 'wrong_budget',
    'not_completed', 'unverified_checkpoint',
])
def test_incomplete_or_changed_training_evidence_is_rejected(evidence, corruption):
    report, history, root = evidence
    if corruption == 'incomplete':
        history.pop()
    elif corruption == 'epoch_order':
        history[10]['epoch'] = 9
    elif corruption == 'no_best':
        for row in history:
            row['same_population_accuracy'] = 0.
        report['same_population_accuracy'] = 0.
    elif corruption == 'nonfinite':
        history[4]['training_loss'] = float('nan')
    elif corruption == 'invalid_accuracy':
        history[4]['same_population_accuracy'] = .123
    elif corruption == 'report_disagrees':
        report['same_population_accuracy'] = .75
    elif corruption == 'missing_binding':
        report['implementation_sha256'].pop('sciona/bengali_classifier_fit.py')
    elif corruption == 'modified_code':
        (root / 'sciona/bengali_classifier_fit.py').write_text('changed')
    elif corruption == 'wrong_budget':
        report['optimizer_updates'] = 119
    elif corruption == 'not_completed':
        report['passed'] = False
    elif corruption == 'unverified_checkpoint':
        report['fresh_checkpoint_output_exact'] = False
    with pytest.raises(ValueError):
        validate_completion(report, history, root)


@pytest.fixture
def handoff(evidence):
    report, history, root = evidence
    source_root = Path(__file__).resolve().parents[1]
    verifiers = {}
    for name in ['scripts/qualify_bengali_classifier_checkpoint.py', 'sciona/bengali_checkpoint_handoff.py']:
        payload = (source_root / name).read_bytes()
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        verifiers[name] = hashlib.sha256(payload).hexdigest()
    runtime = root / 'runtime'
    runtime.mkdir()
    # Receipt validation checks opaque content; tensor loading has separate tests.
    checkpoint = b'synthetic checkpoint placeholder'
    (runtime / 'best.pt').write_bytes(checkpoint)
    history_bytes = json.dumps(history).encode()
    (runtime / 'history.json').write_bytes(history_bytes)
    completion_bytes = json.dumps(report).encode()
    completion = root / 'completion.json'
    completion.write_bytes(completion_bytes)
    receipt = dict(receipt_kind='post_completion_synthetic_consistency', passed=True,
        approved=False, catalog_mutations=0, synthetic_only=True,
        selected_epoch_evaluation_replayed=True, training_time_checkpoint_digest_available=False,
        completion_sha256=hashlib.sha256(completion_bytes).hexdigest(),
        history_sha256=hashlib.sha256(history_bytes).hexdigest(),
        checkpoint_sha256=hashlib.sha256(checkpoint).hexdigest(),
        selected_epoch=3, same_population_accuracy=.75,
        classifier_implementation_sha256=report['implementation_sha256'], verifier_sha256=verifiers)
    receipt_path = root / 'receipt.json'
    receipt_path.write_text(json.dumps(receipt))
    digest = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    return receipt_path, digest, runtime, completion, root


def test_complete_handoff_returns_bound_checkpoint_receipt(handoff):
    receipt = validate_handoff(*handoff)
    assert receipt['selected_epoch'] == 3
    assert receipt['training_time_checkpoint_digest_available'] is False


@pytest.mark.parametrize('corruption', ['receipt', 'checkpoint', 'history', 'completion',
                                       'verifier', 'missing_verifier', 'selected_epoch'])
def test_handoff_rejects_changed_or_inconsistent_qualification(handoff, corruption):
    receipt_path, digest, runtime, completion, root = handoff
    if corruption == 'receipt':
        receipt_path.write_text(receipt_path.read_text() + ' ')
    elif corruption == 'checkpoint':
        (runtime / 'best.pt').write_bytes(b'changed synthetic checkpoint')
    elif corruption == 'history':
        (runtime / 'history.json').write_text('[]')
    elif corruption == 'completion':
        completion.write_text('{}')
    elif corruption == 'verifier':
        (root / 'scripts/qualify_bengali_classifier_checkpoint.py').write_text('changed')
    else:
        receipt = json.loads(receipt_path.read_text())
        if corruption == 'missing_verifier':
            receipt['verifier_sha256'].pop('sciona/bengali_checkpoint_handoff.py')
        else:
            receipt['selected_epoch'] = 7
        receipt_path.write_text(json.dumps(receipt))
        digest = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        validate_handoff(receipt_path, digest, runtime, completion, root)
