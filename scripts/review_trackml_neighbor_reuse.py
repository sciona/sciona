"""Compare existing reusable neighbor operations with the retained source backend."""
import inspect
import json
import numpy as np
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits
from scripts.audit_decomposed_trackml_readiness import ROOT, sha
from sciona.atoms.ml.sklearn.neighbors.atoms import (
    nearest_neighbors_fit, nearest_neighbors_kneighbors,
    nearest_neighbors_radius_neighbors,
)


def review():
    pins = json.loads((ROOT/'docs/reviews/competition_trackml_source_pins.json').read_text())
    source = ROOT.parent/'sciona-atoms-physics/src/sciona/atoms/particle_tracking/source_tracking/solution_trackml_solution_neighbors.py.source'
    if sha(source) != pins['files']['trackml_solution/neighbors.py']:
        raise ValueError('Pinned neighbor source changed')
    paths = dict(source=source, retained=ROOT/'sciona/trackml_neighbors.py',
        provider=inspect.getfile(inspect.unwrap(nearest_neighbors_fit)), reviewer=__file__)
    before = {k: sha(p) for k, p in paths.items()}
    counts = dict(cases=0, k_index_differences=0, k_distance_differences=0,
                  radius_index_order_differences=0, radius_membership_differences=0,
                  radius_distance_differences=0)
    methods = set()
    with threadpool_limits(limits=1):
        for seed in range(8):
            for dimension in (2, 3):
                for samples in (16, 64):
                    rng = np.random.RandomState(915+seed)
                    training = rng.normal(size=(samples, dimension))
                    queries = rng.normal(size=(7, dimension))
                    source_state = NearestNeighbors(n_jobs=1).fit(training)
                    methods.add(source_state._fit_method)
                    state = nearest_neighbors_fit(training)
                    expected = source_state.kneighbors(queries, n_neighbors=3)
                    actual = nearest_neighbors_kneighbors(queries, state, n_neighbors=3)
                    counts['cases'] += 1
                    counts['k_index_differences'] += int(not np.array_equal(expected[1], actual[1]))
                    counts['k_distance_differences'] += int(not np.array_equal(expected[0], actual[0]))
                    expected = source_state.radius_neighbors(queries, radius=1.5)
                    actual = nearest_neighbors_radius_neighbors(queries, state, radius=1.5)
                    counts['radius_index_order_differences'] += int(any(
                        not np.array_equal(a, b) for a, b in zip(expected[1], actual[1], strict=True)))
                    counts['radius_membership_differences'] += int(any(
                        set(a) != set(b) for a, b in zip(expected[1], actual[1], strict=True)))
                    counts['radius_distance_differences'] += int(any(
                        not np.array_equal(a, b) for a, b in zip(expected[0], actual[0], strict=True)))
        # Equidistant points expose tie selection/order, independently of rounding.
        training = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.], [2., 0.], [-2., 0.]])
        queries = np.zeros((1, 2))
        expected = NearestNeighbors(n_jobs=1).fit(training).kneighbors(queries, n_neighbors=3)
        actual = nearest_neighbors_kneighbors(queries, nearest_neighbors_fit(training), n_neighbors=3)
        tie_order_differs = not np.array_equal(expected[1], actual[1])
    if before != {k: sha(p) for k, p in paths.items()}:
        raise ValueError('Reviewed software changed')
    exact = not any(v for k, v in counts.items() if k != 'cases') and not tie_order_differs
    return dict(format='trackml-neighbor-reuse.v1', passed=True, approved=False,
        catalog_mutations=0, synthetic_only=True, comparison=counts,
        tied_k_neighbor_order_differs=tie_order_differs, source_fit_methods=sorted(methods),
        exact_replacement_supported=exact, implementation_sha256=before,
        decision=('Existing operations match these cases; lifecycle qualification remains required.' if exact else
                  'Do not substitute the existing dense-search atoms into the source-faithful lifecycle. Expose reusable fitted search and query operations preserving the retained backend behavior; keep layer partitioning, coordinate transforms and identifier mapping in the recipe.'),
        limitations=[
            'Synthetic numeric inputs only. Differences establish non-equivalence, not that either implementation is mathematically invalid.',
            'Unsorted radius results and k-neighbor ties are backend-dependent. No arbitrary-platform stable ordering is claimed.',
            'Exact floating-point and ordering differences require full downstream lifecycle comparison before substitution.',
            'This review neither publishes providers nor approves the original CDG.',
        ], resource_policy=dict(serial=True, native_threads=1, search_jobs=1))


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/trackml_neighbor_reuse.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
