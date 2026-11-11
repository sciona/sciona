"""Review reusable ordered resource selection against pinned public source."""
import ast
import inspect
import json

import numpy as np

from sciona.atoms.particle_tracking.track_matching.atoms import greedy_track_commit
from sciona.trackml_candidates import zeroUsedHits
from scripts.audit_decomposed_trackml_readiness import ROOT, sha


def source_function():
    pins = json.loads((ROOT/'docs/reviews/competition_trackml_source_pins.json').read_text())
    path = ROOT.parent/'sciona-atoms-physics/src/sciona/atoms/particle_tracking/source_tracking/solution_trackml_solution_candidates.py.source'
    if sha(path) != pins['files']['trackml_solution/candidates.py']:
        raise ValueError('Pinned candidate source changed')
    functions = [n for n in ast.parse(path.read_text()).body
                 if isinstance(n, ast.FunctionDef) and n.name == 'zeroUsedHits']
    if len(functions) != 1:
        raise ValueError('Unique source function required')
    namespace = dict(np=np)
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<pinned-commit>', 'exec'), namespace)
    return namespace['zeroUsedHits'], path


def compare(source, matrix, used, minimum, loss, fraction, limit):
    source_matrix, source_used = matrix.copy(), used.copy()
    retained_matrix, retained_used = matrix.copy(), used.copy()
    kwargs = dict(min_nhits=minimum, max_nloss=loss, max_loss_fraction=fraction, max_nrows=limit)
    source(source_matrix, source_used, **kwargs, reserve_skipped=False)
    zeroUsedHits(retained_matrix, retained_used, **kwargs, reserve_skipped=False)
    actual = greedy_track_commit(matrix, used, **dict(kwargs,
        max_nloss=matrix.shape[1] if loss is None else loss,
        max_nrows=matrix.shape[0] if limit is None else limit))
    for a, b, c in zip(actual, (source_matrix, source_used), (retained_matrix, retained_used), strict=True):
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(b, c)
    return actual


def review():
    source, path = source_function()
    paths = dict(source=path, retained=ROOT/'sciona/trackml_candidates.py', reviewer=__file__,
                 provider=inspect.getfile(inspect.unwrap(greedy_track_commit)))
    before = {k: sha(p) for k, p in paths.items()}
    count = 0
    for seed in range(12):
        rng = np.random.RandomState(571+seed)
        for rows, width in [(1, 1), (16, 4), (32, 12)]:
            matrix = rng.randint(0, 40, (rows, width)).astype(np.int64)
            matrix[:, 0] = rng.randint(1, 40, rows)  # source forbids all-zero rows
            used = rng.uniform(size=40) < .2
            for controls in [(1, None, 1., None), (3, 2, .5, 5), (1, 0, 0., 0), (2, 4, 1., 1)]:
                saved_matrix, saved_used = matrix.copy(), used.copy()
                compare(source, matrix, used, *controls)
                np.testing.assert_array_equal(matrix, saved_matrix)
                np.testing.assert_array_equal(used, saved_used)
                count += 1
    # Repetition within a row counts as multiple positions in both implementations.
    repeated, _ = compare(source, np.array([[1, 1, 0], [1, 2, 3]], dtype=np.int64),
                          np.zeros(4, dtype=bool), 2, None, 1., None)
    np.testing.assert_array_equal(repeated, [[1, 1, 0], [0, 2, 3]])
    # Reserving skipped rows is a genuine source option absent from the atom.
    matrix = np.array([[1, 2, 0], [1, 3, 4]], dtype=np.int64)
    used = np.zeros(5, dtype=bool)
    reserved, reserved_used = matrix.copy(), used.copy()
    source(reserved, reserved_used, min_nhits=3, reserve_skipped=True)
    ordinary, _ = compare(source, matrix, used, 3, None, 1., None)
    if np.array_equal(reserved, ordinary):
        raise ValueError('Expected reservation counterexample missing')
    # The provider accepts an empty row, whereas source asserts before selection.
    try:
        source(np.zeros((1, 3), dtype=np.int64), used.copy())
    except AssertionError:
        pass
    else:
        raise ValueError('Source empty-row boundary changed')
    empty, _ = greedy_track_commit(np.zeros((1, 3), dtype=np.int64), used)
    if empty.any():
        raise ValueError('Provider empty-row behavior changed')
    if before != {k: sha(p) for k, p in paths.items()}:
        raise ValueError('Reviewed software changed')
    return dict(format='trackml-commit-reuse.v1', passed=True, approved=False,
        catalog_mutations=0, synthetic_only=True, exact_source_batches=count,
        repeated_position_case=True, reservation_counterexample=True, empty_row_difference=True,
        implementation_sha256=before,
        decision='Reuse existing pure ordered resource-selection arithmetic for reserve_skipped=False, with explicit defaults and caller-managed mutation. Preserve reservation behavior separately if required by the calling recipe.',
        limitations=[
            'Nonnegative integer resource IDs index the supplied one-dimensional boolean used mask. Zero is a sentinel. IDs must be in bounds; source-qualified rows contain at least one nonzero entry.',
            'Selection is ordered and greedy, not globally optimal. Position counts include duplicate IDs within a row; this is not unique-resource counting.',
            'Provider defaults max_nloss=100 and max_nrows=100000 differ from source None, which means matrix width and row count. Translate None explicitly.',
            'Source mutates both arrays and returns None; provider copies both inputs and returns the updated pair. Integration must preserve caller-visible behavior.',
            'Source reserve_skipped=True consumes resources even for rejected rows; the existing atom has no equivalent option. The retained tracking lifecycle explicitly uses False.',
            'Source asserts on all-zero rows while provider skips them. Do not claim unconditional source equivalence.',
            'Candidate storage, ordering, identifier assignment, submission formatting and fill policy remain caller operations. Numeric resource IDs require no particle-specific interpretation.',
            'Synthetic parity is not historical competition accuracy or qualification of the original CDG. Lifecycle integration and catalog qualification remain pending.',
        ])


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/trackml_commit_reuse.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
