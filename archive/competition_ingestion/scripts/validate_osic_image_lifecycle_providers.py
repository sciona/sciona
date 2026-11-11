"""Exercise separate lifecycle providers against frozen synthetic state evidence."""
import argparse
from contextlib import ExitStack
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from unittest.mock import patch

from scripts import validate_osic_image_state as original
from sciona.atoms.ml.domain_adapters import osic_image_lifecycle as provider


def redirect_run(command, *args, **kwargs):
    command = list(command)
    target = str(Path(original.__file__).resolve())
    if target in command and any(mode in command for mode in ('--worker', '--replay')):
        command[command.index(target)] = str(Path(__file__).resolve())
    return RUN(command, *args, **kwargs)


RUN = subprocess.run


def execute(directory, replay=False):
    from sciona import osic_image_state as state
    export = state.export_prediction_state
    restore = state.restore_prediction_state
    calls = dict(build_image_model=0, export_image_prediction_state=0,
                 restore_image_prediction_state=0)

    def build():
        calls['build_image_model'] += 1
        return provider.build_image_model()

    def export_via_provider(model, directory, **coordinates):
        calls['export_image_prediction_state'] += 1
        with patch.object(state, 'export_prediction_state', export):
            return provider.export_image_prediction_state(model, directory, **coordinates)

    def restore_via_provider(directory, **coordinates):
        calls['restore_image_prediction_state'] += 1
        with patch.object(state, 'restore_prediction_state', restore):
            return provider.restore_image_prediction_state(directory, **coordinates)

    with ExitStack() as stack:
        stack.enter_context(patch.object(state, 'build_model', build))
        stack.enter_context(patch.object(state, 'export_prediction_state', export_via_provider))
        stack.enter_context(patch.object(state, 'restore_prediction_state', restore_via_provider))
        stack.enter_context(patch.object(original.subprocess, 'run', redirect_run))
        result = original.replay(directory) if replay else original.worker(directory)
    expected = (dict(build_image_model=1, export_image_prediction_state=0,
                     restore_image_prediction_state=1) if replay else
                dict(build_image_model=3, export_image_prediction_state=1,
                     restore_image_prediction_state=10))
    if calls != expected:
        raise ValueError('Lifecycle provider execution coverage differs')
    result['lifecycle_provider_calls'] = calls
    return result


def validate(dependencies, source_directory):
    baseline = json.loads((original.ROOT/'docs/reviews/osic_image_state.json').read_text())
    for name, digest in baseline['software_sha256'].items():
        if original.sha(original.ROOT/name) != digest:
            raise ValueError('Frozen state qualification implementation changed')
    source_directory = Path(source_directory)
    sources = [source_directory/name for name in ('source.h5', 'snapshot.json')]
    before = [original.sha(path) for path in sources]
    with tempfile.TemporaryDirectory(prefix='sciona-image-lifecycle-', dir='/private/tmp') as temporary:
        directory = Path(temporary)
        for path in sources:
            shutil.copyfile(path, directory/path.name)
        with patch.object(original.subprocess, 'run', redirect_run):
            report = original.validate(dependencies, directory)
    if before != [original.sha(path) for path in sources]:
        raise ValueError('Private source checkpoint changed during qualification')
    export_calls = report['execution'].pop('lifecycle_provider_calls')
    replay_calls = report['execution']['fresh_process_replay'].pop('lifecycle_provider_calls')
    if report != baseline:
        raise ValueError('Provider lifecycle differs from frozen synthetic qualification')
    try:
        provider.witness_image_lifecycle()
    except ValueError:
        pass
    else:
        raise ValueError('Abstract lifecycle must reject')
    return dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
        state_qualification=report, provider_calls=dict(export=export_calls, fresh_replay=replay_calls),
        private_source_unchanged=True, abstract_execution_rejected=True,
        provider_sha256=original.sha(inspect.getfile(provider)),
        validator_sha256=original.sha(__file__), limitations=[
            'Fixed source B5 architecture and private numeric inference state; not a general model serialization format.',
            'Intermediate synthetic checkpoint only; no full-training or original-CDG approval.',
            'Coordinates and provenance are caller declarations; numeric digests do not establish authenticity.',
            'Model construction consumes RNG state. Export is inference-only and may leave partial files on I/O failure.',
            'Export and fresh replay run sequentially with one compute thread and one dataset worker.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-directory', required=True, type=Path)
    parser.add_argument('--dependencies', type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--worker', action='store_true')
    modes.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    if args.worker or args.replay:
        print(json.dumps(execute(args.runtime_directory, replay=args.replay)))
    else:
        if not args.dependencies:
            parser.error('--dependencies required')
        result = validate(args.dependencies, args.runtime_directory)
        (original.ROOT/'docs/reviews/osic_image_lifecycle_providers.json').write_text(
            json.dumps(result, indent=2)+'\n')
        print(json.dumps(dict(passed=True, provider_calls=result['provider_calls'],
                             private_source_unchanged=True)))
