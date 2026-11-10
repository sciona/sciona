"""Compare generic fitted sparse clipping with pinned source on synthetic inputs."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin

from sciona.fitted_sparse_clip import fit_sparse_clip

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = '3dcc0f896e965bf0acece4cbed61fc58b7d82d10be3bb8a71eb768f747466c44'


def validate(source):
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError('Pinned source required')
    tree = ast.parse(source.read_text())
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name in ('SparseMatrixOptimize', 'SanitizeSparseMatrix')]
    if len(classes) != 2:
        raise ValueError('Both source preprocessing operations required')
    namespace = dict(np=np, sp=sparse, BaseEstimator=BaseEstimator, TransformerMixin=TransformerMixin)
    exec(compile(ast.Module(body=classes, type_ignores=[]), '<pinned-sparse-reference>', 'exec'), namespace)
    rng = np.random.default_rng(829)
    cases = 0
    for width in (1, 3, 17):
        for scale in (0.25, 1., 20.):
            training = rng.uniform(-1, 3, (23, width)) * scale
            training[0] = np.nan
            training[1] = 0
            query = rng.uniform(-4, 8, (11, width)) * scale
            query[0] = np.nan
            query[1] = np.inf
            query[2] = -np.inf
            for storage in (lambda x: x, sparse.csr_matrix):
                tr, qr = storage(training.copy()), storage(query.copy())
                optimize = namespace['SparseMatrixOptimize']()
                reference = namespace['SanitizeSparseMatrix']().fit(optimize.transform(tr), None)
                expected = reference.transform(optimize.transform(qr))
                state = fit_sparse_clip(tr)
                actual = state.transform(qr)
                assert state.upper == reference.datamax
                assert actual.dtype == expected.dtype
                np.testing.assert_array_equal(actual.indptr, expected.indptr)
                np.testing.assert_array_equal(actual.indices, expected.indices)
                np.testing.assert_array_equal(actual.data, expected.data)
                cases += 1
    paths = ['sciona/fitted_sparse_clip.py', 'tests/test_fitted_sparse_clip.py',
             'scripts/validate_mercari_sparse_clip.py', 'docs/licenses/mercari-solution-LICENSE.txt']
    return dict(format='mercari-sparse-clip-reference.v1', passed=True, approved=False,
        catalog_mutations=0, synthetic_only=True, exact_reference_cases=cases,
        source_commit='374301ad1c32cbc93dcc40313d5d7bb9c5503746', source_sha256=SOURCE_SHA,
        software_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
        limits=['Only float32 conversion and fitted sparse clipping are qualified; no complete feature graph or original CDG approval.',
                'Canonical numeric matrix storage and consistent feature ordering required; equal width does not prove equal semantics.',
                'No stored training values, all-NaN training, infinite training entries, negative fitted maxima and duplicate CSR entries are explicitly rejected boundaries.',
                'Inference NaNs become zero and infinities clip to frozen training bounds, matching source behavior on the supported domain.',
                'Immutable fitted state and copied inputs intentionally replace source in-place mutation; application adapters retain responsibility for training/held-out separation and feature units.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    result = validate(parser.parse_args().source)
    (ROOT/'docs/reviews/competition_mercari_sparse_clip_reference.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, exact_reference_cases=result['exact_reference_cases'], approved=False)))
