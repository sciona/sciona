"""Exact provider semantics and software dependency review for Tier 3 publication."""
import argparse
import ast
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import xml.etree.ElementTree as ET

from scripts.stage_connectomics_filter_providers import reviewed_plan
from scripts.review_decomposed_connectomics import ROOT, require, sha, review as integration_review

TESTS = ('tests/test_ordered_circular_filter.py', 'tests/test_decomposed_connectomics.py',
         'tests/test_decomposed_connectomics_review.py', 'tests/test_connectomics_provider_intake.py',
         'tests/test_connectomics_runtime.py')
TEST_REPORT = ROOT/'docs/reviews/connectomics_filter_provider_tests.json'
REPORT = ROOT/'docs/reviews/connectomics_filter_provider_semantics.json'
SOURCE_FILES = ('sciona/connectomics_runtime.py', 'sciona/connectomics_source.py',
                'sciona/decomposed_connectomics_runtime.py')
PROVIDER_FILES = ('sciona-atoms/src/sciona/atoms/numerical/spatial/circular_filter.py',
                  'sciona-atoms-ml/src/sciona/atoms/ml/connectomics_filter_execution.py')

CONTRACTS = {
    'ordered_circular_filter': dict(
        operation='Add weighted, wrapped shifts of an array in the order supplied by the caller.',
        plain_limits='Use nonempty arrays of finite numbers stored in 32 or 64 bit form. Choose the axis, whole-number shifts and weights explicitly. Weights may be negative and shifts may repeat. The result has the same shape and number format. The input is unchanged. Overflow is rejected.',
        validity_bounds='Nonempty finite float32/float64 array, ndim >= 1. Nonnegative integer axis within ndim; bool rejected. Equal nonempty 1D integer shifts and finite real weights; bool controls rejected. Each weight must be representable in input dtype. Python-float coefficients preserve weak-scalar float32 arithmetic. Ordered weighted rolls are accumulated left to right, without normalization or regrouping. Nonfinite terms or partial sums are rejected. No units, axis meaning, sampling rate or domain policy is inferred.',
        parameters={
            'values':'Finite nonempty float32 or float64 array. Any dimensionality of at least one; copied output retains shape and dtype.',
            'shifts':'Nonempty one-dimensional integer shifts; positive shifts move values toward increasing indices, wrapping at the boundary. Repeats are allowed.',
            'weights':'One-dimensional finite real weights, same length as shifts and representable in input dtype. Signed weights allowed; order affects rounding.',
            'axis':'Explicit nonnegative integer axis less than values.ndim; booleans rejected.'},
        outputs='Fresh array of the same shape and dtype; no caller array mutation.',
        dependencies=['numpy', 'numbers (Python standard library)', 'sciona.ghost.registry (registration only)'],
        reuse='Domain independent. Callers supply axis and all filter taps; no Connectomics names or policies occur in the implementation.'),
    'filter_configuration': dict(
        operation='Check the prepared input and return the four filter settings used by this source recipe.',
        plain_limits='This step uses the source recipe’s fixed weights and time axis. It copies and checks the input again. It returns the values, axis and four pairs of shifts and weights in a fixed order. It does not apply the filters.',
        validity_bounds='Requires Connectomics Prepared; revalidates the complete version-1 payload contract through prepare. Returns a fresh read-only Fortran float32 signals array, axis=0, and f1/f2/f3/f4 shift/weight lists. Output order is values, axis, f1_shifts, f1_weights, f2_shifts, f2_weights, f3_shifts, f3_weights, f4_shifts, f4_weights. Tap policies are source-specific and do not define the generic numerical atom.',
        parameters={'prepared':'Prepared object with finite nonnegative float32-compatible time-by-node signals, >=3 nodes and samples>=max(5,nodes+1), nonconstant channels, simple/tuned mode and boolean directivity. Revalidated and copied.'},
        outputs='Ten ordered outputs; complete tap values and ordering qualified by source parity and actual graph execution.',
        dependencies=['sciona.decomposed_connectomics_runtime.filter_configuration', 'sciona.connectomics_runtime.Prepared', 'sciona.connectomics_runtime.prepare', 'numpy', 'sciona.ghost.registry (registration only)'],
        reuse='Source recipe adapter. The circular numerical operation remains a separately bound generic provider.'),
    'execute_prefiltered': dict(
        operation='Run the full source recipe with the four supplied filter results and the caller’s chosen method.',
        plain_limits='Use the checked graph to supply all four filter results for the same input. This step checks their shape and number format, but does not prove that their weights were correct. It runs every source threshold and optionally adds the time order score. Invalid or degenerate results are rejected. These scores are not calibrated probabilities.',
        validity_bounds='Revalidated Prepared plus four finite float32 ndarrays exactly matching signals.shape, including f3/f4 in simple mode. Filter snapshots are copied and bound to equal source signal values; each source callback gets a fresh copy. Filter correctness is a graph-binding obligation, not inferred from shape. Executes complete 120-entry corrected sweeps: simple 240 historical PCA fits, tuned 480. Includes repeated thresholds, source weights, historical sample-count variance, 80% components, and normalization after aggregation. Optional directivity uses n_jobs=1 and 0.997/0.003 blend. Nonfinite filtered signals, nonpositive residual covariance and degenerate score normalization fail closed. Hash-verified locally provisioned source required.',
        parameters={
            'prepared':'Prepared object revalidated through the same version-1 input contract; explicit simple/tuned mode and boolean directivity.',
            **{f:'Finite float32 ndarray matching prepared.signals.shape; must be the corresponding source filter output for those exact signals. A copy is retained. The qualified graph supplies this binding.' for f in ('f1','f2','f3','f4')}},
        outputs='Version-1 result dictionary with mode, directivity, pca_fits and finite square score lists. Synthetic source parity does not establish scientific validity or calibrated probabilities.',
        dependencies=['sciona.decomposed_connectomics_runtime.execute_prefiltered', 'sciona.decomposed_connectomics_runtime.checked_filter', 'sciona.connectomics_runtime.Prepared', 'sciona.connectomics_runtime.prepare', 'sciona.connectomics_source.source_namespace', 'hash-pinned source function namespace and historical PCA class', 'numpy', 'scipy', 'scikit-learn base/validation interfaces', 'joblib', 'sciona.ghost.registry (registration only)'],
        reuse='Source recipe adapter; it consumes numerical arrays through explicit ports. Generic filters are upstream graph providers, not hidden calls from this adapter.'),
}


def run_tests():
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONPATH=str(ROOT))
    with tempfile.TemporaryDirectory(prefix='connectomics-provider-tests-') as temporary:
        junit = Path(temporary)/'results.xml'
        result = subprocess.run([str(ROOT/'.venv/bin/python'), '-m', 'pytest', '-q', *TESTS,
                                 '--junitxml='+str(junit)], cwd=ROOT, env=env, capture_output=True, text=True)
        require(result.returncode == 0, 'Provider tests failed: '+result.stdout[-2500:]+result.stderr[-1000:])
        suites = ET.parse(junit).getroot().findall('testsuite')
        counts = {k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ('tests','failures','errors','skipped')}
        require(counts == dict(tests=67, failures=0, errors=0, skipped=0), 'Expected complete 67-test provider qualification')
    report = dict(passed=True, **counts, synthetic_only=True,
                  test_sha256={p:sha(ROOT/p) for p in TESTS},
                  runner_sha256=sha(Path(__file__)))
    TEST_REPORT.write_text(json.dumps(report, indent=2)+'\n')
    return report


def review():
    intake = reviewed_plan()
    integration = integration_review()
    tests = json.loads(TEST_REPORT.read_text())
    require(tests == dict(passed=True, tests=67, failures=0, errors=0, skipped=0, synthetic_only=True,
                         test_sha256={p:sha(ROOT/p) for p in TESTS}, runner_sha256=sha(Path(__file__))),
            'Provider test evidence missing or stale')
    dependencies = dict(line.split('==') for line in (ROOT/'requirements/connectomics-execution.txt').read_text().splitlines())
    require(all(importlib.metadata.version(k) == v for k,v in dependencies.items()), 'Runtime package versions differ')
    files = {p: ROOT/p for p in SOURCE_FILES}
    files.update({p: ROOT.parent/p for p in PROVIDER_FILES})
    imports = {}
    for name, path in files.items():
        tree = ast.parse(path.read_text())
        imports[name] = sorted({ast.unparse(n) for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))})
    # Every registered provider call is explicit in the graph. Runtime adapters
    # call plain matcher functions; those functions are hashed software dependencies,
    # not fictitious calls to the similarly named registered prepare wrapper.
    internal_atom_imports = [line for lines in imports.values() for line in lines if 'sciona.atoms.' in line]
    require(not internal_atom_imports, 'Unexpected internal registered-provider import')
    records = []
    for record in intake['providers']:
        name = record['runtime_fqdn'].rsplit('.',1)[1]
        contract = CONTRACTS[name]
        require(set(contract['parameters']) == {p['name'] for p in record['interfaces'] if p['direction']=='input'}, 'Parameter semantic coverage differs')
        records.append(dict(record, **contract, internal_registered_provider_dependencies=[]))
    return dict(format='connectomics-filter-provider-semantics.v1', passed=True, approved=False,
        review_source='automated', proposed_tier=3, verdict='acceptable_with_limits',
        providers=records, graph_digest=intake['graph_digest'],
        software_sha256={name:sha(path) for name,path in files.items()}, imports=imports,
        package_versions=dependencies, python_version=platform.python_version(),
        source_namespace=dict(
            source_pins_sha256=sha(ROOT/'docs/reviews/competition_connectomics_source_pins.json'),
            execution='Only pinned historical PCA class and pinned function definitions are compiled. Source module imports, main routine, data loaders and rank helper are not executed.',
            callbacks='f1/f2/f3/f4 are replaced with checked prefiltered snapshots; PCA and scale are replaced with finite/degeneracy guards; print is suppressed.',
            namespace_libraries=['numpy','scipy.linalg','math.sqrt','sklearn.base.BaseEstimator','sklearn.base.TransformerMixin','sklearn.utils.as_float_array','itertools.chain','joblib.Parallel','joblib.delayed','joblib.cpu_count'],
            parallelism='Graph nodes execute serially. Optional source directivity explicitly receives n_jobs=1. Validation caps native pools at one; deployment must configure its own native thread limits.'),
        evidence_sha256={p.name:sha(p) for p in (TEST_REPORT, ROOT/'docs/reviews/connectomics_decomposed_semantic_review.json',
            ROOT/'docs/reviews/connectomics_decomposed_graph_execution.json', ROOT/'docs/reviews/connectomics_generic_circular_filter_source_parity.json',
            ROOT/'docs/reviews/connectomics_filter_provider_intake_plan.json')},
        intake_stage_mapping=integration['source_stage_mapping'],
        limitations=['Automated Tier 3 review only; no Tier 1 human review or Tier 2 usage evidence.',
                    'Synthetic numerical and full-sweep source agreement; no historical accuracy or biological validity claim.',
                    'Recipe adapters require the hash-verified source cache and matching Python packages.',
                    'Direct consumers of execute_prefiltered are responsible for filter provenance; approved graph bindings establish that relationship.',
                    'The generic operation permits other domains, axes and signed taps; users determine their scientific interpretation.',
                    'This review does not change catalog approval; publication needs matching metadata and transaction gates.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-tests', action='store_true')
    if parser.parse_args().run_tests:
        run_tests()
    result = review()
    REPORT.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, providers=len(result['providers']), tests=67, approved=False)))
