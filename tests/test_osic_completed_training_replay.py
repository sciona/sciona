"""Synthetic orchestration gates; these tests do not qualify model execution."""
import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from scripts import replay_osic_completed_training as recovery
from scripts.audit_osic_training_finalization import FILES


@pytest.fixture
def run(tmp_path):
    directory = tmp_path / 'run'
    (directory / 'checkpoints').mkdir(parents=True)
    launch = dict(synthetic_only=True, code_sha256={
        name: hashlib.sha256(recovery.frozen(name)).hexdigest() for name in FILES},
        epochs=30, batch_size=8, training_steps=32, validation_steps=16,
        training_seed=517, fixture_seed=805, native_threads=1)
    (directory / 'launch.json').write_text(json.dumps(launch))
    rows = [dict(epoch=i, optimizer_iterations=32*i, loss=1., val_loss=1.,
                 learning_rate=.003) for i in range(1, 31)]
    (directory / 'progress.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (directory / 'checkpoints/best.h5').write_bytes(b'synthetic orchestration checkpoint')
    (directory / 'replay.npz').write_bytes(b'synthetic orchestration replay inputs')
    return directory


def test_incomplete_schedule_rejects_before_loading(run):
    progress = run / 'progress.jsonl'
    progress.write_text('\n'.join(progress.read_text().splitlines()[:19])+'\n')
    with pytest.raises(ValueError, match='thirty-epoch'):
        recovery.prerequisites(run)


def test_partial_final_line_and_missing_replay_reject(run):
    progress = run / 'progress.jsonl'
    progress.write_text(progress.read_text()+'{')
    with pytest.raises(ValueError, match='thirty-epoch'):
        recovery.prerequisites(run)
    progress.write_text(progress.read_text()[:-1])
    (run / 'replay.npz').unlink()
    with pytest.raises(ValueError, match='original replay inputs'):
        recovery.prerequisites(run)


@pytest.mark.parametrize('failure', [None, 'input_mutation', 'bad_result', 'dependency_drift'])
def test_snapshot_replay_gates(run, tmp_path, monkeypatch, failure):
    dependencies = tmp_path / 'dependencies'
    dependencies.mkdir()
    wheel = dependencies / 'synthetic.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('synthetic_module.py', '# synthetic fixture\n')
    pins = {wheel.name: recovery.digest(wheel)}
    original_frozen = recovery.frozen
    def frozen(name):
        if name == 'scripts/validate_osic_image_model.py':
            return ('PINS = '+repr(pins)).encode()
        return original_frozen(name)
    monkeypatch.setattr(recovery, 'frozen', frozen)
    if failure == 'dependency_drift':
        wheel.write_bytes(b'changed synthetic wheel')
    calls = []
    def execute(command, *, env, cwd, check):
        calls.append(command)
        assert check and command[2] == '--replay'
        assert Path(command[1]).read_bytes() == original_frozen('scripts/run_osic_synthetic_image_training.py')
        assert env['OMP_NUM_THREADS'] == env['TF_NUM_INTRAOP_THREADS'] == '1'
        snapshot = Path(command[-1])
        assert snapshot != run
        for name in ('launch.json', 'progress.jsonl', 'checkpoints/best.h5', 'replay.npz'):
            assert (snapshot/name).read_bytes() == (run/name).read_bytes()
        result = dict(passed=True, queries=2, maximum_absolute_error=0.,
                      fresh_process=True, fit_disabled=True)
        if failure == 'bad_result':
            result['passed'] = False
        (snapshot / 'replay.json').write_text(json.dumps(result))
        if failure == 'input_mutation':
            (run / 'replay.npz').write_bytes(b'changed synthetic inputs')
    # check_output also uses subprocess.run. Cache actual Git bytes before
    # replacing the replay subprocess with a synthetic orchestration stub.
    frozen_files = {name: original_frozen(name) for name in FILES}
    monkeypatch.setattr(recovery.subprocess, 'run', execute)
    monkeypatch.setattr(recovery.subprocess, 'check_output',
                        lambda command, **kwargs: frozen_files[command[-1].split(':', 1)[1]])
    output = tmp_path / 'review'
    if failure:
        with pytest.raises(ValueError):
            recovery.replay(run, dependencies, output)
        assert not output.exists()
    else:
        report = recovery.replay(run, dependencies, output)
        assert report['approved'] is False and report['original_inputs_unchanged']
        assert (output/'private-input-identities.json').exists()
        assert not (run/'result.json').exists() and not (run/'replay.json').exists()
    assert len(calls) == (0 if failure == 'dependency_drift' else 1)
