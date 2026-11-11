"""Compare joint feature preparation to pinned cells using synthetic tables."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

from sciona.osic_quantile_features import FIELDS, assemble_populations, prepare_features, joint_population_features
from scripts.review_osic_original_scope import ROOT, INFERENCE, REVISION

PINS = {
    25: '3030cf0c7c9f0d8cfa1495cafcff63b46af4cdd7d5b0ace67d1ee8bcd5af1745',
    26: '6697bebc904c1362b8ae7133bb8c08d945b6e1308bfbfb4dea9bc1f26022031c',
    27: 'eb6ead161d5798a1e890524697e1f7c6b0bcba25a6789392d7cd5cac0c013c9b',
    28: '52ad20e6247bd5e706aa48e236968b7153527833dfe235309e796d96437dc983',
    29: 'f31938bb6637fe9e46e692e60a36824fae56e0211969664b8ed5738a394958fc',
    30: '819477c019a40feaf5432c98f81e3221d2cd9c5581862a7eeded8925841e9050',
}


def synthetic_tables():
    training = pd.DataFrame([
        ['a', 0., 10., 20., 'a0', 'b0'], ['a', 2., 14., 20., 'a0', 'b0'],
        ['b', 1., 30., 40., 'a1', 'b1'], ['b', 3., 32., 40., 'a1', 'b1'],
        ['removed', 9., 100., 50., 'a2', 'b2'], ['removed', 9., 101., 50., 'a2', 'b2'],
    ], columns=FIELDS)
    baseline = pd.DataFrame([
        ['a', 0., 999., 20., 'a0', 'b0'], ['c', 4., 50., 60., 'a2', 'b2'],
    ], columns=FIELDS)
    queries = pd.DataFrame([['c', -2.], ['a', 7.], ['c', 10.]], columns=['entity', 'time'])
    return training, baseline, queries


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(directory):
    paths = [Path(__file__), ROOT/'sciona/osic_quantile_features.py']
    before = {str(p.relative_to(ROOT)):sha(p) for p in paths}
    notebook = json.loads((Path(directory)/INFERENCE).read_text())
    trees = {}
    for cell, digest in PINS.items():
        tree = ast.parse(''.join(notebook['cells'][cell]['source']))
        if hashlib.sha256(ast.unparse(tree).encode()).hexdigest() != digest:
            raise ValueError('Source feature cell changed')
        trees[cell] = tree
    # Discover software field names from source syntax, never notebook outputs.
    mapping = {
        'entity': trees[25].body[2].value.func.value.value.args[0].value,
        'time': trees[25].body[0].value.slice.value,
        'target': trees[26].body[1].value.func.value.slice.elts[1].value,
        'numeric': trees[29].body[0].value.left.left.slice.value,
        'category_a': trees[28].body[0].value.elts[0].value,
        'category_b': trees[28].body[0].value.elts[1].value,
        'population': trees[25].body[1].targets[0].slice.elts[0].left.attr,
    }
    roles = {role: trees[30].body[i].value.slice.comparators[0].value
             for i, role in enumerate(('training', 'baseline', 'query'))}
    output_variables = {role: trees[30].body[i].targets[0].id
                        for i, role in enumerate(('training', 'baseline', 'query'))}
    training, baseline, queries = synthetic_tables()
    originals = [v.copy(deep=True) for v in (training, baseline, queries)]
    combined = assemble_populations(training, baseline, queries)
    if len(combined[combined.population == 'training']) != 4 or 'removed' in set(combined.entity):
        raise ValueError('All duplicated training rows must be removed')
    if combined.loc[combined.population == 'query', 'entity'].tolist() != ['c', 'a', 'c']:
        raise ValueError('Query merge changed caller order')
    cases = []
    for label, frame in [('original_order', combined), ('reordered_training', pd.concat([
            combined[combined.population == 'training'].iloc[::-1],
            combined[combined.population != 'training']], ignore_index=True))]:
        candidate = joint_population_features(frame, allow_transductive=True)
        source_frame = frame.rename(columns=mapping).copy()
        source_frame[mapping['population']] = source_frame[mapping['population']].map(roles)
        namespace = dict(data=source_frame, np=np, pd=pd)
        for cell in PINS:
            exec(compile(trees[cell], '<pinned-synthetic-features>', 'exec'), namespace)
        for role in output_variables:
            expected = namespace[output_variables[role]]
            np.testing.assert_array_equal(candidate['populations'][role]['features'], expected[namespace['FE']].to_numpy())
            np.testing.assert_array_equal(candidate['populations'][role]['targets'], expected[mapping['target']].to_numpy())
            if candidate['populations'][role]['entities'] != expected[mapping['entity']].tolist():
                raise ValueError('Source entity ordering differs')
        cases.append(dict(case=label, exact_source_matrices=True, exact_targets=True,
                          ordered_rows_preserved=True, feature_count=len(candidate['feature_columns'])))
    first = prepare_features(training, baseline, queries, allow_transductive=True)
    extended = queries.copy(); extended.loc[2, 'time'] = 100.
    changed = prepare_features(training, baseline, extended, allow_transductive=True)
    if np.array_equal(first['populations']['training']['features'], changed['populations']['training']['features']):
        raise ValueError('Expected transductive query-range dependence absent')
    if first['ranges']['baseline_target'] != [10., 50.] or first['ranges']['time_offset'] != [-6., 7.]:
        raise ValueError('Observed-only baseline time or first matching target differs')
    rejected = []
    def reject(label, operation):
        try:
            operation()
        except ValueError:
            rejected.append(label)
        else:
            raise ValueError('Invalid feature contract accepted: '+label)
    reject('implicit_transduction', lambda:prepare_features(training, baseline, queries, allow_transductive=False))
    reject('duplicate_baseline', lambda:assemble_populations(training, pd.concat([baseline, baseline]), queries))
    reject('missing_query_baseline', lambda:assemble_populations(training, baseline.iloc[:1], queries))
    missing = combined.copy(); missing.loc[0, 'numeric'] = np.nan
    reject('missing_numeric', lambda:joint_population_features(missing, allow_transductive=True))
    constant = combined.copy(); constant['numeric'] = 1.
    reject('constant_scaling_range', lambda:joint_population_features(constant, allow_transductive=True))
    collision = combined.copy(); collision['category_b'] = collision['category_a']
    reject('category_label_collision', lambda:joint_population_features(collision, allow_transductive=True))
    for current, original in zip((training, baseline, queries), originals):
        pd.testing.assert_frame_equal(current, original)
    if before != {str(p.relative_to(ROOT)):sha(p) for p in paths}:
        raise ValueError('Implementation changed')
    return dict(format='osic-quantile-features-validation.v1', passed=True, approved=False,
        catalog_mutations=0, synthetic_only=True, source_revision=REVISION,
        normalized_source_cell_sha256=PINS, cases=cases, rejected_contract_cases=rejected,
        duplicate_training_rows_removed=2, query_range_changes_training_features=True,
        observed_only_baseline_time=True, first_matching_baseline_target=True, inputs_unchanged=True,
        software_sha256=before, runtime=dict(numpy=np.__version__, pandas=pd.__version__),
        limitations=['Source cells for baseline/category/scaling computation execute on synthetic preassembled tables.',
            'Outer deduplication and metadata join are tested by invariants, not by executing source file-loading cells.',
            'Complete mapped fields, one baseline per query entity, finite nondegenerate ranges and noncolliding category names are required.',
            'Query population affects training features; this recipe is unsuitable when queries are unavailable during preparation.',
            'Domain mapping, observation-availability proof, image branch, full graph and catalog promotion remain pending.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-directory', required=True)
    result = validate(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/osic_quantile_features_validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, source_cases=len(result['cases']), rejected_cases=len(result['rejected_contract_cases']), approved=False)))
