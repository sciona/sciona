"""Compare reusable quantile encoding with pinned MIT source on synthetic values."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sciona.quantile_target_encoding import quantile_boundaries, quantile_midpoints, encode_quantile_targets

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = 'bfcf3c64486544eba4b2a7f95b07a98b6fe9aeaad6f72ae11ad7e7575c1645cd'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(source):
    if sha(source) != SOURCE_SHA:
        raise ValueError('Pinned source-only function module required')
    namespace = {}
    exec(compile(source.read_text(), '<pinned-quantile-reference>', 'exec'), namespace)
    rng = np.random.default_rng(813)
    populations = [np.linspace(-1., 1., 101), rng.uniform(-1, 1, 73),
                   np.repeat([-.5, 0., .5], [20, 30, 40]), np.full(17, .25)]
    cases = 0
    for training in populations:
        for count in (1, 2, 7, 64):
            expected_edges = namespace['get_percentiles'](training, count)
            edges = quantile_boundaries(training, count)
            np.testing.assert_array_equal(edges, expected_edges)
            np.testing.assert_array_equal(quantile_midpoints(edges), namespace['get_mean_percentiles'](edges))
            query = np.concatenate([training[:9], edges])
            for soft in (False, True):
                actual = encode_quantile_targets(query, edges, soft=soft)
                expected = namespace['binarize'](query, expected_edges, soft=soft)
                np.testing.assert_array_equal(actual, expected)
                cases += 1
    with np.errstate(under='ignore', invalid='ignore', divide='ignore'):
        reference = namespace['binarize'](np.array([1e6]), np.array([0., .5, 1.]))
    if np.isfinite(reference).all():
        raise ValueError('Expected source underflow boundary was not observed')
    try:
        encode_quantile_targets([1e6], [0., .5, 1.])
    except ValueError:
        pass
    else:
        raise ValueError('Nonfinite source boundary must be rejected')
    paths = ['sciona/quantile_target_encoding.py', 'tests/test_quantile_target_encoding.py',
             'scripts/validate_mercari_quantile_reference.py', 'docs/licenses/mercari-solution-LICENSE.txt']
    return dict(format='mercari-quantile-reference.v1', passed=True, approved=False,
        catalog_mutations=0, synthetic_only=True, source_commit='374301ad1c32cbc93dcc40313d5d7bb9c5503746',
        source_sha256=SOURCE_SHA, exact_encoding_cases=cases, exact_boundary_and_midpoint_cases=16,
        nonfinite_source_boundary_rejected=True,
        implementation_sha256={p: sha(ROOT/p) for p in paths},
        limits=['Float64 finite numeric domain; normalization underflow is an explicit rejected boundary.',
                'Response transform, precision units, training population and probability calibration remain caller obligations.',
                'Generic numeric tests do not establish application accuracy, complete Mercari training, or catalog approval.',
                'No historical TensorFlow/MXNet training was executed by this reference check.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    result = validate(parser.parse_args().source)
    (ROOT/'docs/reviews/competition_mercari_quantile_reference.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
