"""Probe pinned source feature recipes serially using entirely synthetic rows.

Diagnostic only: explicit compatibility shims are not historical equivalence.
No source dataset is opened, and no rows or feature vocabularies are reported.
"""
import argparse
import array
import ast
from collections import defaultdict
from functools import partial
import hashlib
from importlib.metadata import version
import json
import logging
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler

from inventory_mercari_features import inventory

ROOT = Path(__file__).resolve().parents[1]


class SerialPool:
    def __init__(self, *args):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def map(self, function, values, **kwargs):
        return list(map(function, values))


class SerialUnion:
    """Mirror source branch selection and process-isolated inputs serially."""
    def __init__(self, branches, n_jobs=1):
        self.branches = branches

    def _run(self, X, y, fit):
        results = []
        for _, branch in self.branches:
            selected = X[branch.steps[0][1].columns].copy(deep=True)
            results.append(branch.fit_transform(selected, y) if fit else branch.transform(selected))
        return sparse.hstack(results).tocsr()

    def fit_transform(self, X, y=None, **kwargs):
        return self._run(X, y, True)

    def transform(self, X):
        return self._run(X, None, False)


class ModernApiNames(ast.NodeTransformer):
    def visit_Attribute(self, node):
        self.generic_visit(node)
        if node.attr == 'ix':
            node.attr = 'loc'
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'replace' and any(k.arg == 'flags' for k in node.keywords):
            node.keywords.append(ast.keyword(arg='regex', value=ast.Constant(True)))
        if isinstance(node.func, ast.Name) and node.func.id == 'SGDRegressor':
            for keyword in node.keywords:
                if keyword.arg == 'loss' and isinstance(keyword.value, ast.Constant) and keyword.value.value == 'squared_loss':
                    keyword.value = ast.Constant('squared_error')
        return node


def probe(directory, modernize=False):
    topology = inventory(directory)
    namespace = dict(np=np, pd=pd, sp=sparse, csr_matrix=sparse.csr_matrix,
        BaseEstimator=BaseEstimator, TransformerMixin=TransformerMixin,
        DictVectorizer=DictVectorizer, CountVectorizer=CountVectorizer, TfidfVectorizer=TfidfVectorizer,
        SGDRegressor=SGDRegressor, MinMaxScaler=MinMaxScaler, MaxAbsScaler=MaxAbsScaler,
        make_pipeline=make_pipeline, FeatureUnionMP=SerialUnion,
        make_union_mp=lambda *branches, **kwargs: SerialUnion(list(enumerate(branches))),
        Pool=SerialPool, tqdm=lambda values, **kwargs: values, logger=logging.getLogger('mercari-probe'),
        array=array, re=re, defaultdict=defaultdict, partial=partial,
        _make_int_array=lambda: array.array('i'), UNK='unk',
        ITEM_DESCRIPTION_MAX_LENGTH=1050, NAME_MAX_LENGTH=100, BRAND_NAME_MAX_LENGTH=100)
    exec(compile((directory/'mercari/text_utils.py').read_text(), '<source-text>', 'exec'), namespace)
    def try_float(value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0
    namespace['try_float'] = try_float
    tree = ast.parse((directory/'mercari/transformers.py').read_text())
    if modernize:
        tree = ast.fix_missing_locations(ModernApiNames().visit(tree))
    definitions = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), '<source-transformers>', 'exec'), namespace)
    if modernize:
        for name in ('ReportShape', 'SparseMatrixOptimize', 'FalseBrands', 'FeaturesPatterns'):
            namespace[name].__sklearn_is_fitted__ = lambda self: True
    # These names are public source interface bindings, not a private template.
    # Every value below is generated solely for this probe.
    train = pd.DataFrame([dict(name=f'synthetic sensor {i%4+1} cm',
        item_description=f'synthetic measurement {i%3+1} cm stable signal',
        category_name='synthetic/devices/sensors', brand_name=f'maker{i%2}',
        shipping=i%2, item_condition_id=i%3+1) for i in range(160)])
    query = train.iloc[:8].copy(deep=True)
    boundaries = {}
    unseen = query.copy(deep=True)
    for column in ('name', 'item_description', 'category_name', 'brand_name'):
        unseen[column] = 'unseen synthetic token'
    boundaries['unseen_text_and_categories'] = unseen
    missing = query.copy(deep=True)
    for column in missing.columns:
        missing[column] = None
    boundaries['all_missing_values'] = missing
    blank = query.copy(deep=True)
    for column in ('name', 'item_description', 'category_name', 'brand_name'):
        blank[column] = ''
    boundaries['empty_text_values'] = blank
    short_category = query.copy(deep=True)
    short_category['category_name'] = 'synthetic'
    boundaries['short_category_hierarchy'] = short_category
    boundaries['zero_query_rows'] = query.iloc[:0].copy(deep=True)
    targets = np.linspace(1., 10., len(train))
    cases = []
    for backend, filename in [('tensorflow', 'datasets_tf.py'), ('mxnet', 'datasets_mx.py')]:
        tree = ast.parse((directory/'mercari'/filename).read_text())
        definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), '<source-feature-recipes>', 'exec'), namespace)
        for group, function in enumerate(definitions, 1):
            case = dict(backend=backend, group=group, passed=False)
            try:
                pipeline = namespace[function.name](n_jobs=1)
                fitted = pipeline.fit_transform(train.copy(deep=True), targets)
                predicted = pipeline.transform(query.copy(deep=True))
                assert sparse.issparse(fitted) and sparse.issparse(predicted)
                assert fitted.shape[0] == 160 and predicted.shape == (8, fitted.shape[1])
                assert np.isfinite(fitted.data).all() and np.isfinite(predicted.data).all()
                reverse = pipeline.transform(query.iloc[::-1].copy(deep=True))[::-1].tocsr()
                split = sparse.vstack([pipeline.transform(query.iloc[:3].copy(deep=True)),
                                       pipeline.transform(query.iloc[3:].copy(deep=True))]).tocsr()
                for alternate in (reverse, split):
                    assert alternate.shape == predicted.shape
                    assert (alternate != predicted).nnz == 0
                case.update(passed=True, synthetic_feature_count=fitted.shape[1])
                case['synthetic_row_order_and_batch_partition_invariant'] = True
                case['held_out_boundaries'] = []
                for label, frame in boundaries.items():
                    boundary = dict(synthetic_case=label, passed=False)
                    try:
                        result = pipeline.transform(frame.copy(deep=True))
                        assert sparse.issparse(result) and result.shape == (len(frame), fitted.shape[1])
                        assert np.isfinite(result.data).all()
                        boundary['passed'] = True
                    except Exception as error:
                        boundary.update(error_type=type(error).__name__, error=str(error))
                    # A held-out transform, even one that fails, must not change
                    # subsequent predictions for the unchanged reference query.
                    replay = pipeline.transform(query.copy(deep=True))
                    np.testing.assert_array_equal(replay.indptr, predicted.indptr)
                    np.testing.assert_array_equal(replay.indices, predicted.indices)
                    np.testing.assert_array_equal(replay.data, predicted.data)
                    boundary['reference_query_unchanged'] = True
                    case['held_out_boundaries'].append(boundary)
            except Exception as error:
                case.update(passed=False, error_type=type(error).__name__, error=str(error))
            cases.append(case)
    return dict(format='mercari-feature-runtime-probe.v1', approved=False, catalog_mutations=0,
        synthetic_only=True, worker_processes=0, modern_api_names=modernize, source_commit=topology['source_commit'],
        source_sha256=topology['source_sha256'], cases=cases,
        runtime_versions={name: version(name) for name in ('numpy', 'scipy', 'pandas', 'scikit-learn', 'Unidecode', 'nltk')},
        compatibility_shims=['Pool.map executes serially.',
            'Source feature union replaced by serial branch selection, deep-copied inputs and sparse concatenation; weighted unions are not qualified.',
            'Removed sklearn integer-array helper replaced by array.array of C integers.',
            'Source config constants supplied explicitly; source imports bypassed to avoid logger files and private-library import failures.'] + ([
            'Explicit regex=True for regex replacements carrying flags; source ix label slicing becomes loc.',
            'SGD squared_loss spelling becomes squared_error; current optimizer defaults remain unqualified against historical defaults.',
            'Stateless ReportShape, SparseMatrixOptimize, FalseBrands and FeaturesPatterns advertise fitted status for modern sklearn pipeline checks.'] if modernize else []),
        limits=['Diagnostic execution in current dependencies only, not a promoted provider implementation or historical environment proof.',
                'No source dataset, real-data accuracy, training ensemble or full original CDG approval.',
                'Held-out boundary outcomes apply only to the recorded synthetic cases; arbitrary token populations, training on missing values and all legacy feature semantics remain unqualified.',
                'Unchanged reference-query replay checks observable inference stability after boundary transforms; it does not prove that every internal object is immutable.'],
        probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-directory', type=Path, required=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        directory = parser.parse_args().source_directory
        result = dict(format='mercari-feature-runtime-comparison.v1', approved=False,
                      baseline=probe(directory), modern_api=probe(directory, modernize=True))
    (ROOT/'docs/reviews/competition_mercari_feature_runtime_probe.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: dict(pipelines_passed=sum(c['passed'] for c in result[k]['cases']),
        boundary_cases=sum(len(c.get('held_out_boundaries', [])) for c in result[k]['cases']),
        boundaries_passed=sum(b['passed'] for c in result[k]['cases'] for b in c.get('held_out_boundaries', [])))
        for k in ('baseline', 'modern_api')}))
