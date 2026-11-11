"""Compare registered tokenization and clipping in all three TF feature recipes.

All inputs are synthetic. Source loaders remain diagnostic; this does not
publish feature recipes or certify the original competition graph.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import warnings
import random

import numpy as np
import pandas as pd

from probe_mercari_feature_runtime import load_feature_namespace
from sciona.atoms.ml import fitted_sparse_clipping as provider

from validate_sparse_clip_feature_integration import RegisteredClip
from sciona.atoms.ml import character_run_tokenization as tokenizer_provider
from sciona.mercari_text_policy import WORD_CHARACTERS

ROOT = Path(__file__).resolve().parents[1]


class RegisteredTokenizer:
    calls = 0

    def __call__(self, text):
        type(self).calls += 1
        return tokenizer_provider.tokenize_character_runs(text, WORD_CHARACTERS, ' \t\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(directory):
    paths = ['scripts/validate_registered_tokenizer_features.py',
             'scripts/validate_sparse_clip_feature_integration.py',
             'sciona/character_run_tokenizer.py', 'sciona/mercari_text_policy.py',
             'scripts/probe_mercari_feature_runtime.py', 'scripts/inventory_mercari_features.py',
             'scripts/run_mercari_tensorflow_synthetic_graph.py', 'sciona/fitted_sparse_clip.py']
    before = {p: sha(ROOT/p) for p in paths}
    provider_hash = {p.__name__: sha(p.__file__) for p in (provider, tokenizer_provider)}
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
    reference_tokenizer = original['FastTokenizer']()
    if (set(WORD_CHARACTERS) != reference_tokenizer._default_word_chars_set
            or set(' \t\n') != reference_tokenizer._default_white_space_set):
        raise ValueError('Explicit character policy differs from source')
    rng = random.Random(833)
    alphabet = WORD_CHARACTERS + ' \t\n\r!?🙂漢字\u0301\u00a0'
    texts = ['', 'A--B 12&3!!', '\t \n\r', '🙂🙂漢字', 'a\u0301', 'a' * 10000]
    texts += [''.join(rng.choices(alphabet, k=rng.randrange(1, 500))) for _ in range(256)]
    for text in texts:
        tokens = RegisteredTokenizer()(text)
        if tokens != reference_tokenizer(text) or ''.join(tokens) != text:
            raise ValueError('Registered tokenizer differs from source')
    adapted['SanitizeSparseMatrix'] = RegisteredClip
    adapted['FastTokenizer'] = RegisteredTokenizer
    RegisteredTokenizer.calls = 0
    recipes = ast.parse((directory/'mercari/datasets_tf.py').read_text())
    definitions = [n for n in recipes.body if isinstance(n, ast.FunctionDef)]
    for namespace in (original, adapted):
        exec(compile(ast.Module(body=definitions, type_ignores=[]), '<pinned-feature-recipes>', 'exec'), namespace)
    cases = []
    group_calls = {}
    for group in (1, 3, 2):
        prior_calls = RegisteredTokenizer.calls
        name = f'prepare_vectorizer_{group}_tf'
        reference, candidate = original[name](n_jobs=1), adapted[name](n_jobs=1)
        pairs = [('training', reference.fit_transform(training.copy(deep=True), targets),
                  candidate.fit_transform(training.copy(deep=True), targets))]
        for role, frame in [('calibration', calibration), ('query', query)]:
            pairs.append((role, reference.transform(frame.copy(deep=True)),
                          candidate.transform(frame.copy(deep=True))))
        group_calls[str(group)] = RegisteredTokenizer.calls - prior_calls
        if (group in (1, 3)) != (group_calls[str(group)] > 0):
            raise ValueError('Unexpected tokenizer use in feature recipe')
        if sum(isinstance(stage, RegisteredClip) for _, stage in candidate.steps) != 1:
            raise ValueError('Expected exactly one registered clipping adapter')
        for role, expected, actual in pairs:
            expected = RegisteredClip.canonical(expected)
            assert actual.shape == expected.shape and actual.dtype == expected.dtype
            for part in ('indptr', 'indices', 'data'):
                np.testing.assert_array_equal(getattr(actual, part), getattr(expected, part))
            assert np.isfinite(actual.data).all()
            cases.append(dict(group=group, synthetic_role=role, exact_canonical_csr_equal=True))
    if RegisteredTokenizer.calls < 1:
        raise ValueError('Registered tokenizer was not exercised')
    if before != {p: sha(ROOT/p) for p in paths} or provider_hash != {p.__name__: sha(p.__file__) for p in (provider, tokenizer_provider)}:
        raise ValueError('Implementation changed during validation')
    return dict(format='registered-tokenizer-feature-integration.v1', passed=True, approved=False,
        catalog_mutations=0, synthetic_only=True, cases=cases, registered_clip_stages=3, registered_tokenizer_calls=RegisteredTokenizer.calls,
        role_counts=dict(training=8192, calibration=256, query=32),
        exact_tokenizer_reference_cases=len(texts), tokenizer_calls_by_group=group_calls,
        source_commit=topology['source_commit'], source_sha256=topology['source_sha256'],
        software_sha256=before, provider_sha256=provider_hash,
        resource_policy=dict(concurrent_pipelines=1, worker_processes=0, native_threads=1),
        limitations=['Exact current-runtime synthetic feature matrices only; no historical runtime or real-data accuracy claim.',
                    'Explicit adapter sorts CSR indices without summing duplicates. Source matrices are compared after the same storage normalization; downstream floating-point reduction order is not certified by this feature-only check.',
                    'Slot identities bind this fitted pipeline ordering; they do not establish vocabulary equivalence across independent pipelines.',
                    'Source character policy is an explicit application adapter; generic tokenizer accepts caller-defined alphabets. Text cleaning, fitted encoders and other feature operations remain diagnostic source code with documented compatibility shims.',
                    'No model retraining or whole-CDG approval; dependency and catalog publication review remain pending.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-directory', type=Path, required=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = validate(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/registered_tokenizer_feature_integration.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, exact_feature_matrix_cases=len(result['cases']), approved=False)))
