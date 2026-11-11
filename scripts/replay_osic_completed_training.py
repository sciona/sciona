"""Replay a completed synthetic checkpoint without modifying its original run.

This is separate relocation-recovery evidence, never a replacement result.json
or approval of training, relocated providers, or the original competition CDG.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

from scripts.audit_osic_training_finalization import ROOT, REVISION, audit


def frozen(name):
    return subprocess.check_output(['git', 'show', REVISION + ':' + name], cwd=ROOT)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def prerequisites(directory):
    report = audit(directory)
    if not report['complete_schedule_observed'] or report['partial_progress_line']:
        raise ValueError('Complete thirty-epoch schedule required before replay')
    if not report['checkpoint_present'] or not report['replay_inputs_present']:
        raise ValueError('Completed checkpoint and original replay inputs required')
    return report


def replay(directory, dependencies, output):
    directory, dependencies, output = (p.resolve() for p in (directory, dependencies, output))
    if output.is_relative_to(directory) or directory.is_relative_to(output):
        raise ValueError('Output must be separate from the original run')
    if output.exists():
        raise ValueError('Output directory must not already exist')
    before = prerequisites(directory)
    names = ('launch.json', 'progress.jsonl', 'checkpoints/best.h5', 'replay.npz')
    identities = {name: digest(directory / name) for name in names}
    script = frozen('scripts/run_osic_synthetic_image_training.py')
    tree = ast.parse(frozen('scripts/validate_osic_image_model.py'))
    pins = [ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'PINS' for t in n.targets)]
    if len(pins) != 1:
        raise ValueError('Expected one frozen dependency pin mapping')
    pins = pins[0]
    for name, expected in pins.items():
        if digest(dependencies / name) != expected:
            raise ValueError('Dependency archive identity differs')
    with tempfile.TemporaryDirectory(prefix='sciona-osic-final-replay-') as temporary:
        root = Path(temporary)
        overlay = root / 'overlay'
        overlay.mkdir()
        for name in pins:
            if name.endswith('.whl'):
                with zipfile.ZipFile(dependencies / name) as wheel:
                    wheel.extractall(overlay)
        if any(digest(dependencies / name) != expected for name, expected in pins.items()):
            raise ValueError('Dependency archive changed during extraction')
        snapshot = root / 'snapshot'
        (snapshot / 'checkpoints').mkdir(parents=True)
        for name in names:
            shutil.copyfile(directory / name, snapshot / name)
            if digest(snapshot / name) != identities[name]:
                raise ValueError('Original run changed while copying replay inputs')
        runner = root / 'scripts/run_osic_synthetic_image_training.py'
        runner.parent.mkdir()
        runner.write_bytes(script)
        env = dict(os.environ, PYTHONPATH=str(overlay), TF_USE_LEGACY_KERAS='1',
                   TF_CPP_MIN_LOG_LEVEL='3', MPLCONFIGDIR=str(root / 'matplotlib'))
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS',
                    'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS'):
            env[key] = '1'
        subprocess.run([sys.executable, str(runner), '--replay',
                        '--runtime-directory', str(snapshot)], env=env, cwd=root, check=True)
        result = json.loads((snapshot / 'replay.json').read_text())
        expected = dict(passed=True, queries=2, maximum_absolute_error=0.,
                        fresh_process=True, fit_disabled=True)
        if result != expected:
            raise ValueError('Unexpected frozen replay result')
        if any(digest(directory / name) != identities[name] for name in names):
            raise ValueError('Original run changed during replay')
        if prerequisites(directory) != before:
            raise ValueError('Original run finalization state changed during replay')
    report = dict(format='osic-completed-training-replay.v1', approved=False,
                  catalog_mutations=0, synthetic_only=True, source_revision=REVISION,
                  replay_script_sha256=hashlib.sha256(script).hexdigest(),
                  completed_epochs=30, observed_optimizer_updates=960,
                  original_inputs_unchanged=True, replay=result,
                  limitations=[
                      'Separate frozen-script replay after source relocation; original result is not repaired or replaced.',
                      'Launch identities and progress do not authenticate execution provenance.',
                      'Prepared synthetic same-population training is not empirical or source stochastic parity evidence.',
                      'Does not qualify relocated providers, numeric-state portability, raw preparation, or the original CDG.'])
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    (output / 'replay-review.json').write_text(json.dumps(report, indent=2) + '\n')
    # Keep model/input identities private, separate from the publishable aggregate.
    (output / 'private-input-identities.json').write_text(json.dumps(identities, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-directory', type=Path, required=True)
    parser.add_argument('--dependencies', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.runtime_directory, args.dependencies, args.output_directory), indent=2))
