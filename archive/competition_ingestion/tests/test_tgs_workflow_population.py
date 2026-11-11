"""Keep full-workflow synthetic inputs identical to the completed population pilot."""
import ast
from pathlib import Path

import numpy as np

from scripts.run_tgs_synthetic_workflow import synthetic_populations


def test_full_population_matches_completed_pilot_and_branch_alignment():
    path = Path(__file__).resolve().parents[1]/'scripts/run_salt_population_pilot.py'
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    statements, active = [], False
    for node in function.body:
        names = [n.id for n in node.targets if isinstance(n, ast.Name)] if isinstance(node, ast.Assign) else []
        if 'seed' in names:
            active = True
        if active:
            statements.append(node)
        if 'population' in names:
            break
    namespace = dict(np=np)
    exec(compile(ast.Module(body=statements, type_ignores=[]), '<synthetic-pilot>', 'exec'), namespace)
    actual = synthetic_populations(namespace['seed'])
    for key, value in namespace['population'].items():
        np.testing.assert_array_equal(actual['keras'][key], value)
    for name in ('labeled_images', 'query_images'):
        expected = actual['keras'][name][..., 0].astype(np.float32)/255
        np.testing.assert_array_equal(actual['pytorch'][name], expected)
    masks = actual['keras']['labeled_masks']
    folds = actual['keras']['folds']
    for fold in range(5):
        assert (folds == fold).sum() == 32
        assert (folds != fold).sum() == 128
        assert set(masks[folds == fold].any(axis=(1, 2))) == {False, True}


def test_population_seed_changes_pixels_without_changing_fold_contract():
    first, second = synthetic_populations(7913), synthetic_populations(7914)
    assert not np.array_equal(first['keras']['labeled_images'], second['keras']['labeled_images'])
    np.testing.assert_array_equal(first['keras']['folds'], second['keras']['folds'])
