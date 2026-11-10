"""Verify JSON-restored text encoders inside all three source TF feature recipes.

All inputs are synthetic. Source loaders remain diagnostic; this does not
publish feature recipes or certify the original competition graph.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from probe_mercari_feature_runtime import load_feature_namespace
from sciona.atoms.ml import fitted_sparse_clipping as provider

ROOT = Path(__file__).resolve().parents[1]


from validate_sparse_clip_feature_integration import RegisteredClip
from sciona.mercari_text_policy import source_tokenizer
from sciona.text_vectorizer_state import export_text_vectorizer, restore_text_vectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def restore_encoders(stage, records):
    if type(stage) in (CountVectorizer, TfidfVectorizer):
        state = json.loads(json.dumps(export_text_vectorizer(stage), allow_nan=False))
        restored = restore_text_vectorizer(state)
        records.append(dict(mode=state['mode'], dtype=state['parameters']['dtype'],
                            tokenizer_policy=state['parameters']['tokenizer'] is not None))
        return restored
    if hasattr(stage, 'steps'):
        stage.steps = [(name, restore_encoders(child, records)) for name, child in stage.steps]
    if hasattr(stage, 'branches'):
        stage.branches = [(name, restore_encoders(child, records)) for name, child in stage.branches]
    return stage


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(directory):
    paths = ['scripts/validate_text_vectorizer_feature_replay.py',
             'scripts/validate_sparse_clip_feature_integration.py',
             'sciona/text_vectorizer_state.py', 'sciona/character_run_tokenizer.py',
             'sciona/mercari_text_policy.py',
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
    adapted, second_topology = load_feature_namespace(directory, modernize=True)
    if topology != second_topology:
        raise ValueError('Source changed between loads')
    adapted['SanitizeSparseMatrix'] = RegisteredClip
    adapted['FastTokenizer'] = source_tokenizer
    recipes = ast.parse((directory/'mercari/datasets_tf.py').read_text())
    definitions = [n for n in recipes.body if isinstance(n, ast.FunctionDef)]
    for namespace in (original, adapted):
        exec(compile(ast.Module(body=definitions, type_ignores=[]), '<pinned-feature-recipes>', 'exec'), namespace)
    cases = []
    restored_records = []
    for group in (1, 3, 2):
        name = f'prepare_vectorizer_{group}_tf'
        reference, candidate = original[name](n_jobs=1), adapted[name](n_jobs=1)
        pairs = [('training', reference.fit_transform(training.copy(deep=True), targets),
                  candidate.fit_transform(training.copy(deep=True), targets))]
        records = []
        candidate = restore_encoders(candidate, records)
        if len(records) != {1: 5, 2: 3, 3: 3}[group]:
            raise ValueError('Exact text encoder count required')
        restored_records.append(dict(group=group, encoders=records))
        for role, frame in [('training_replay', training), ('calibration', calibration), ('query', query)]:
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
    if before != {p: sha(ROOT/p) for p in paths} or provider_hash != sha(provider.__file__):
        raise ValueError('Implementation changed during validation')
    return dict(format='text-vectorizer-feature-replay.v1', passed=True, approved=False,
        catalog_mutations=0, synthetic_only=True, cases=cases, registered_clip_stages=3, restored_text_encoders=restored_records,
        role_counts=dict(training=8192, calibration=256, query=32),
        source_commit=topology['source_commit'], source_sha256=topology['source_sha256'],
        software_sha256=before, provider_sha256=provider_hash,
        resource_policy=dict(concurrent_pipelines=1, worker_processes=0, native_threads=1),
        limitations=['Exact current-runtime synthetic feature matrices only; no historical runtime or real-data accuracy claim.',
                    'Explicit adapter sorts CSR indices without summing duplicates. Source matrices are compared after the same storage normalization; downstream floating-point reduction order is not certified by this feature-only check.',
                    'Slot identities bind this fitted pipeline ordering; they do not establish vocabulary equivalence across independent pipelines.',
                    'Only count and TF-IDF encoders are restored from non-executable JSON state. Table preprocessing, categorical encoders and branch composition remain diagnostic; no whole-pipeline portable state is claimed.',
                    'State includes private fitted vocabulary and IDF values; only aggregate synthetic checks are published. Exact installed sklearn version is required.',
                    'No model retraining or whole-CDG approval; dependency and catalog publication review remain pending.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-directory', type=Path, required=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = validate(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/text_vectorizer_feature_replay.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, exact_feature_matrix_cases=len(result['cases']), approved=False)))
