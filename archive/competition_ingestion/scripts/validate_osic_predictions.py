"""Execute pinned prediction-postprocessing cells with synthetic tables only."""
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from sciona import osic_predictions as candidate
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from scripts.import_flavours_provider_drafts import connect
from scripts.plan_group_distribution_intake import check_dependency
from scripts.review_osic_original_scope import ROOT, INFERENCE, REVISION

PINS = {
    42: '194f22da409fbd34ba80f03aad2a090190d7e9ec90c6452d0cdc3410f39e2a24',
    43: '746c471d43a96e453222c1b6086b311d71b6bdc330ceb1dadd104c2080889c36',
    46: '3051e6d1b59bd681262d0c84488b5aeecbee1a56be5fad9b17eabcb7c65747d1',
    50: '9346203f5c9e13671fa5d8f57b962ad022c71b6cf4c930ac1db0e2cd68b6a92b',
    51: 'f1f52206d7df3cd90a9e81679bf91a39d7187eba6dca915b7cd1ac8f457c732a',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(directory):
    paths = [Path(__file__), ROOT/'sciona/osic_predictions.py', ROOT/'tests/test_osic_predictions.py',
             Path(inspect.getfile(candidate.ordered_weighted_sum))]
    before = {str(p.relative_to(ROOT.parent)): sha(p) for p in paths}
    notebook = json.loads((Path(directory)/INFERENCE).read_text())
    trees = {}
    for cell, digest in PINS.items():
        tree = ast.parse(''.join(notebook['cells'][cell]['source']))
        if hashlib.sha256(ast.unparse(tree).encode()).hexdigest() != digest:
            raise ValueError('Pinned source postprocessing changed')
        trees[cell] = tree
    # Obtain source software field labels at runtime; no source records or outputs.
    final = trees[51].body
    key = final[0].value.func.value.slice.elts[0].value
    point = final[1].targets[0].slice.value
    uncertainty = final[2].targets[0].slice.value
    loop = trees[46].body[1]
    key_expression = loop.body[0].targets[0].slice.elts[0].comparators[0]
    subject_field = key_expression.left.left.value.attr
    time_field = key_expression.right.args[0].value.attr
    separator = key_expression.left.right.value
    known_point_field = loop.body[0].value.value.attr
    keys = [f'synthetic-{i}{separator}{i+2}' for i in range(6)]
    known = pd.DataFrame({subject_field: ['synthetic-1', 'synthetic-4'],
                         time_field: [3, 6], known_point_field: [17., -8.]})
    rng = np.random.default_rng(844)
    cases = []
    for mode in ('crossed_signed_outputs', 'ordered_positive_outputs', 'float32_subnormal_outputs'):
        folds = rng.normal(size=(2, 6, 3)).astype(np.float32 if mode == 'float32_subnormal_outputs' else np.float64)
        if mode == 'ordered_positive_outputs':
            folds = np.sort(np.abs(folds), axis=2)
        else:
            folds[:, 0] = [5., 3., 1.]
        if mode == 'float32_subnormal_outputs':
            folds[:, 2] = np.nextafter(np.float32(0), np.float32(1))
        pe = np.zeros((6, 3))
        for f in folds:
            pe += f/len(folds)
        image = rng.normal(size=(6, 2))
        env = dict(np=np, pe=pe,
            pd=SimpleNamespace(read_csv=lambda _: known.copy()),
            sub=pd.DataFrame({key: keys, point: np.zeros(6), uncertainty: np.zeros(6)}))
        for cell in (42, 43, 46):
            exec(compile(trees[cell], '<pinned-synthetic-postprocess>', 'exec'), env)
        expected_quantile = env['subm'][[point, uncertainty]].to_numpy()
        mask = np.array([False, True, False, False, True, False])
        values = np.array([0., 17., 0., 0., -8., 0.])
        actual_quantile = candidate.quantile_predictions(folds, mask, values)
        np.testing.assert_array_equal(actual_quantile, expected_quantile)
        order = [4, 1, 5, 0, 3, 2]
        env['img_sub'] = pd.DataFrame({key: [keys[i] for i in order],
            point: image[order, 0], uncertainty: image[order, 1]})
        env['reg_sub'] = env['subm'][[key, point, uncertainty]].copy()
        for cell in (50, 51):
            exec(compile(trees[cell], '<pinned-synthetic-blend>', 'exec'), env)
        actual_keys, actual = candidate.blend_predictions([keys[i] for i in order], image[order], keys, actual_quantile)
        if actual_keys != env['df'][key].tolist():
            raise ValueError('Source key ordering differs')
        np.testing.assert_array_equal(actual, env['df'][[point, uncertainty]].to_numpy())
        cases.append(dict(case=mode, queries=6, known_observations=2,
                          quantile_source_exact=True, final_blend_source_exact=True, key_order_exact=True))
    fqdn = 'sciona.atoms.numerical.statistics.ordered_sum.ordered_weighted_sum'
    with connect() as db:
        db.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        rows = db.execute('''SELECT a.artifact_id::text,a.fqdn,v.version_id::text,v.content_hash,v.trust_tier
            FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE a.fqdn=%s AND v.is_latest''', (fqdn,)).fetchall()
        if len(rows) != 1:
            raise ValueError('Unique current numerical provider required')
        check_dependency(db, rows[0])
    repo = ROOT.parent/'sciona-atoms'
    specs = [s for s in _parse_registered_atoms(repo=ProviderRepo(repo.name, repo),
        artifact_root=repo/'src/sciona/atoms') if s.fqdn == fqdn]
    if (len(specs) != 1 or str(specs[0].version_id) != rows[0]['version_id']
            or specs[0].content_hash != rows[0]['content_hash']):
        raise ValueError('Runtime blend differs from approved catalog version')
    if before != {str(p.relative_to(ROOT.parent)): sha(p) for p in paths}:
        raise ValueError('Implementation changed')
    return dict(format='osic-prediction-validation.v1', passed=True, approved=False,
        catalog_mutations=0, synthetic_only=True, source_revision=REVISION,
        normalized_source_cell_sha256=PINS, cases=cases, reused_dependencies=rows,
        software_sha256=before, runtime=dict(numpy=np.__version__, pandas=pd.__version__),
        limitations=['Postprocessing only; source model fitting and image trajectory production are not executed here.',
            'The original clipped-spread atom is not equivalent to the raw spread in the source and is not used.',
            'Known-observation availability and branch identity joins are caller responsibilities.',
            'Finite complete outputs and unique aligned keys only; source missing-prediction behavior remains outside this contract.',
            'No original CDG approval, calibrated interval coverage or historical accuracy claim.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-directory', required=True)
    result = validate(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/osic_prediction_validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, cases=len(result['cases']), reused_dependencies=len(result['reused_dependencies']), approved=False)))
