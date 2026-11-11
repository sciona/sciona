"""Compare whole-feature JSON restoration with source and fresh-process replay.

All inputs are synthetic. The reference uses diagnostic compatibility loaders; the candidate imports
production modules only. This does not approve the original competition graph.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import warnings
import tempfile
import subprocess
import sys
import os
from scipy.sparse import save_npz

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from probe_mercari_feature_runtime import load_feature_namespace
from sciona.atoms.ml import fitted_sparse_clipping as provider

ROOT = Path(__file__).resolve().parents[1]


from sciona.sparse_clip_transformer import SparseClipTransformer as RegisteredClip
from sciona import mercari_feature_recipes
from sciona.mercari_feature_state import export_mercari_features, restore_mercari_features


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(directory, runtime):
    paths = ['scripts/validate_mercari_full_feature_replay.py',
             'scripts/replay_mercari_feature_states.py', 'sciona/mercari_feature_state.py',
             'sciona/feature_transform_state.py', 'sciona/text_vectorizer_state.py', 'sciona/dict_vectorizer_state.py',
             'sciona/mercari_feature_recipes.py', 'sciona/sparse_clip_transformer.py',
             'sciona/mercari_pj_preprocessing.py', 'sciona/mercari_table_preprocessing.py',
             'sciona/mercari_text_cleaning.py', 'sciona/mercari_text_policy.py',
             'sciona/character_run_tokenizer.py', 'sciona/table_feature_operations.py',
             'sciona/serial_feature_union.py',
             'scripts/probe_mercari_feature_runtime.py', 'scripts/inventory_mercari_features.py',
             'scripts/run_mercari_tensorflow_synthetic_graph.py', 'sciona/fitted_sparse_clip.py']
    before = {p: sha(ROOT/p) for p in paths}
    provider_hash = sha(provider.__file__)
    execution = json.loads((ROOT/'docs/reviews/competition_mercari_tensorflow_synthetic_graph.json').read_text())
    population_path = 'scripts/run_mercari_tensorflow_synthetic_graph.py'
    if before[population_path] != execution['software_sha256'][population_path]:
        raise ValueError('Qualified synthetic population generator changed')
    # Extract only the pure synthetic generator, avoiding TensorFlow startup.
    tree = ast.parse((ROOT/population_path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'population')
    ns = dict(np=np, pd=pd)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<synthetic-generator>', 'exec'), ns)
    training, targets = ns['population'](8192, 857)
    calibration, _ = ns['population'](256, 859)
    query, _ = ns['population'](32, 863)
    original, topology = load_feature_namespace(directory, modernize=True)
    recipes = ast.parse((directory/'mercari/datasets_tf.py').read_text())
    definitions = [n for n in recipes.body if isinstance(n, ast.FunctionDef)]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), '<pinned-reference-recipes>', 'exec'), original)
    for role, frame in [('calibration', calibration), ('query', query)]:
        (runtime/f'{role}.json').write_text(frame.to_json(orient='records'))
    cases = []
    for group in (1, 3, 2):
        name = f'prepare_vectorizer_{group}_tf'
        reference = original[name](n_jobs=1)
        candidate = getattr(mercari_feature_recipes, name)(feature_coordinate='synthetic-comparable-features.v1', n_jobs=1)
        pairs = [('training', reference.fit_transform(training.copy(deep=True), targets),
                  candidate.fit_transform(training.copy(deep=True), targets))]
        state = export_mercari_features(candidate)
        (runtime/f'group-{group}.json').write_text(json.dumps(state, allow_nan=False))
        candidate = restore_mercari_features(json.loads(json.dumps(state, allow_nan=False)))
        for role, frame in [('calibration', calibration), ('query', query)]:
            pairs.append((role, reference.transform(frame.copy(deep=True)),
                          candidate.transform(frame.copy(deep=True))))
        if sum(isinstance(stage, RegisteredClip) for _, stage in candidate.steps) != 1:
            raise ValueError('Expected exactly one registered clipping adapter')
        for role, expected, actual in pairs:
            expected = RegisteredClip.canonical(expected)
            assert actual.shape == expected.shape and actual.dtype == expected.dtype
            for part in ('indptr', 'indices', 'data'):
                np.testing.assert_array_equal(getattr(actual, part), getattr(expected, part))
            assert np.isfinite(actual.data).all()
            cases.append(dict(group=group, synthetic_role=role, exact_canonical_csr_equal=True))
            if role != 'training':
                save_npz(runtime/f'group-{group}-{role}.npz', expected)
    child = subprocess.run([sys.executable, '-m', 'scripts.replay_mercari_feature_states', str(runtime)],
                           cwd=ROOT, env=dict(os.environ), capture_output=True, text=True, check=True)
    replay = json.loads(child.stdout)
    if replay != dict(passed=True, restored_pipelines=3, exact_heldout_matrices=6, training_executed=False):
        raise ValueError('Exact complete fresh-process replay required')
    if before != {p: sha(ROOT/p) for p in paths} or provider_hash != sha(provider.__file__):
        raise ValueError('Implementation changed during validation')
    return dict(format='mercari-full-feature-replay.v1', passed=True, approved=False,
        catalog_mutations=0, synthetic_only=True, cases=cases, registered_clip_stages=3, full_pipeline_json_replay=replay,
        role_counts=dict(training=8192, calibration=256, query=32),
        source_commit=topology['source_commit'], source_sha256=topology['source_sha256'],
        software_sha256=before, provider_sha256=provider_hash,
        resource_policy=dict(concurrent_pipelines=1, training_worker_processes=0, replay_child_processes=1, parent_waits_during_child=True, native_threads=1),
        limitations=['Exact current-runtime synthetic feature matrices only; no historical runtime or real-data accuracy claim.',
                    'Explicit adapter sorts CSR indices without summing duplicates. Source matrices are compared after the same storage normalization; downstream floating-point reduction order is not certified by this feature-only check.',
                    'Slot identities bind this fitted pipeline ordering; they do not establish vocabulary equivalence across independent pipelines.',
                    'Candidate assembly uses production modules only; the source reference still uses diagnostic compatibility shims. No historical-runtime equivalence is implied.',
                    'Application text/category policy is explicit and lossy; other domains must supply deliberate adapters. Unidecode dependency publication review remains open.',
                    'Source matrix logging is replaced by explicit identity operations. Their logging effects and empty-input errors are not reproduced.',
                    'No model retraining or whole-CDG approval; dependency and catalog publication review remain pending.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-directory', type=Path, required=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        args = parser.parse_args()
        with tempfile.TemporaryDirectory(prefix='synthetic-full-feature-replay-') as runtime:
            result = validate(args.source_directory, Path(runtime))
    (ROOT/'docs/reviews/mercari_full_feature_replay.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, exact_feature_matrix_cases=len(result['cases']), approved=False)))
