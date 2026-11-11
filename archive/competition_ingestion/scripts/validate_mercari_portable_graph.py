"""Connect production feature JSON to previously qualified synthetic model states.

Only aggregate evidence is committed. Runtime bundles and learned states stay private.
The parent refits deterministic feature recipes; neural models are never retrained.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import warnings

from scripts.validate_mercari_full_feature_replay import validate, sha, ROOT
from scripts.replay_mercari_portable_graph import verify
from sciona.atoms.ml import sparse_neural_prediction as neural
from sciona.atoms.ml import positive_log_blending as blending


def run(source, reference, digest, runtime):
    runtime.mkdir(parents=True, exist_ok=True)
    if any(runtime.iterdir()):
        raise ValueError('Fresh private synthetic runtime required')
    previous = verify(reference, digest)
    if previous['format'] != 'mercari-registered-synthetic.v1':
        raise ValueError('Qualified registered synthetic bundle required')
    prior = json.loads((ROOT/'docs/reviews/mercari_registered_model_graph.json').read_text())
    feature_prior = json.loads((ROOT/'docs/reviews/mercari_full_feature_replay.json').read_text())
    paths = set(prior['software_sha256']) | set(feature_prior['software_sha256']) | {
        'scripts/replay_mercari_portable_graph.py', 'scripts/validate_mercari_portable_graph.py'}
    before = {p: sha(ROOT/p) for p in sorted(paths)}
    for evidence in (prior, feature_prior):
        if not evidence['passed'] or any(before[p] != value for p, value in evidence['software_sha256'].items()):
            raise ValueError('Previously qualified implementation changed')
    providers = {m.__name__: sha(Path(m.__file__)) for m in (neural, blending)}
    if any(prior['provider_sha256'][name] != value for name, value in providers.items()):
        raise ValueError('Qualified model or blend provider changed')
    feature_result = validate(source, runtime)
    for name in ['blend.json', 'expected.npy'] + [f"model-{m['seed']}.json" for m in previous['models']]:
        shutil.copyfile(reference/name, runtime/name)
    manifest = dict(format='mercari-portable-synthetic.v1', models=previous['models'],
                    files={p.name: sha(p) for p in sorted(runtime.iterdir()) if p.is_file()})
    (runtime/'manifest.json').write_text(json.dumps(manifest, sort_keys=True))
    child = subprocess.run([sys.executable, '-m', 'scripts.replay_mercari_portable_graph', str(runtime),
        '--manifest-sha256', sha(runtime/'manifest.json')], cwd=ROOT, env=dict(os.environ),
        capture_output=True, text=True, timeout=240, check=True)
    replay = json.loads(child.stdout.strip().splitlines()[-1])
    expected = dict(passed=True, restored_feature_pipelines=3, restored_models=12,
                    restored_blends=1, exact_final_predictions=True, training_executed=False)
    if replay != expected:
        raise ValueError('Complete portable graph replay required')
    verify(reference, digest)
    if before != {p: sha(ROOT/p) for p in before} or providers != {m.__name__: sha(Path(m.__file__)) for m in (neural, blending)}:
        raise ValueError('Implementation changed during graph validation')
    return dict(format='mercari-portable-graph-validation.v1', passed=True, approved=False,
        catalog_mutations=0, synthetic_only=True, neural_retraining=False,
        feature_reference_cases=feature_result['cases'], fresh_process_replay=replay,
        software_sha256=before, provider_sha256={**providers, 'clipping': feature_result['provider_sha256']},
        source_commit=feature_result['source_commit'], source_sha256=feature_result['source_sha256'],
        resource_policy=dict(concurrent_compute_workers=1, native_threads=1, parents_wait_for_children=True),
        limitations=['Only declared twelve-model TensorFlow recipe; winning backend and additional models remain unresolved.',
            'Synthetic current-runtime parity does not establish historical equivalence or real-data accuracy.',
            'Application preprocessing remains explicit and domain-specific; reusable operations require deliberate adapters.',
            'Unidecode dependency publication review remains open; no original CDG approval is implied.',
            'Parent source comparisons use diagnostic loaders. Fresh prediction replay uses JSON states and production modules only.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-directory', type=Path, required=True)
    parser.add_argument('--reference-directory', type=Path, required=True)
    parser.add_argument('--reference-manifest-sha256', required=True)
    parser.add_argument('--runtime-directory', type=Path, required=True)
    args = parser.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = run(args.source_directory, args.reference_directory, args.reference_manifest_sha256, args.runtime_directory)
    (ROOT/'docs/reviews/mercari_portable_graph.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['fresh_process_replay']))
