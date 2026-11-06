"""Exact provider contracts and dependency qualification; no catalog mutations."""
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
from scripts.stage_adversarial_normalizer_providers import reviewed_plan
from scripts.review_decomposed_adversarial import ROOT, require, sha, review as integration_review

TESTS = ('tests/test_row_scale.py', 'tests/test_decomposed_adversarial_updates.py',
         'tests/test_injected_adversarial_update.py', 'tests/test_decomposed_adversarial_review.py',
         'tests/test_adversarial_provider_intake.py', 'tests/test_runner_callable_injection.py',
         'tests/test_visualizer_runner.py', 'tests/test_execution_graph_codec.py')
TEST_REPORT = ROOT/'docs/reviews/adversarial_normalizer_provider_tests.json'
REPORT = ROOT/'docs/reviews/adversarial_normalizer_provider_semantics.json'
PROVIDER_FILES = ('sciona-atoms/src/sciona/atoms/numerical/statistics/row_scale.py',
                  'sciona-atoms-dl/src/sciona/atoms/dl/adversarial_normalized_execution.py')

CONTRACTS = {
    'normalize_rows_by_scale': dict(
        operation='Divide each row by its average absolute size or by its population standard deviation.',
        plain_limits='Use a nonempty table of finite numbers stored in 32 or 64 bit form. Choose the scale explicitly. Do not subtract the row average. The result keeps the shape and number format. The input is unchanged. Zero scales and nonfinite results are rejected.',
        validity_bounds='Nonempty finite 2D NumPy float32/float64 ndarray. Statistic is exactly mean_absolute or population_std. Divide each row by its mean absolute value or population standard deviation (ddof=0), reduced in the input dtype. The numerator is never centered. Preserve dtype and shape; return a fresh array. Reject zero/nonfinite divisors, nonfinite inputs and nonfinite outputs without any stabilizing epsilon. Axis meaning, units, sample interpretation and downstream domain are caller policy.',
        parameters={'values':'Nonempty finite two-dimensional float32/float64 ndarray; no implicit conversion from lists or integer arrays.',
                    'statistic':'Exact string mean_absolute or population_std; selects the per-row divisor, not numerator centering.'},
        outputs='Fresh finite array of the same shape and dtype; caller values are unchanged.',
        dependencies=['numpy', 'sciona.ghost.registry (registration only)'],
        reuse='Domain independent: rows and columns have no image, model, competition or task meaning in this operation.'),
    'execute_with_normalizer': dict(
        operation='Run the full source attack using the row scaling function supplied by the graph.',
        plain_limits='Use checked image inputs, explicit model settings, and the reviewed scaling function. Keep all model branches, repeated updates, padding, and output rules. Only real images are returned. The bound on normalized values does not guarantee a bound on saved image pixels. No trained weights are downloaded.',
        validity_bounds='Prepared version-1 decoded dictionary from adversarial_prepare, with exact rgb, targets, config and initializations fields, plus an explicit callable accepting values and statistic. The callable must implement the qualified uncentered mean-absolute/population-std row scaling, preserving finite float32 rows; callable identity is a graph binding obligation. Full CPU Torch source-shaped ensembles: non-targeted eight models and caller-positive iteration count (default 10); targeted epsilon>=8 five models/20 steps; targeted epsilon<8 two models/40 steps. Frozen inferred or supplied labels; main/auxiliary objectives, source momentum, round/sign policy, step clipping and fixed-original projection preserved. Source ten-entry padding and real-entry trimming/PNG min-max stretching retained. Zero/nonfinite normalization rejects. Explicit random or caller-decoded Slim states only; exact state shapes validated by model construction. No historical TensorFlow binary or pretrained quality guarantee.',
        parameters={'prepared':'Decoded dictionary from the unchanged version-1 preparation provider. Image layout is nonempty uint8 NHWC 299x299x3 before source padding/normalization; source mode, finite epsilon in [0,255], nonnegative momentum, targets and exact branch initialization scopes must satisfy its contract.',
                    'normalizer':'Explicit callable accepting (values: ndarray, statistic: str) and returning same-shape/dtype finite rows under the qualified uncentered scaling contract. No hidden fallback. The approved graph must bind the qualified generic provider.'},
        outputs='Version-1 JSON-compatible dictionary: encoded normalized real images, encoded labels, base64 real-entry PNGs, per-batch losses, initialization kinds, and saved_pixel_bound_guaranteed=False.',
        dependencies=['sciona.decomposed_adversarial_injected.execute', 'reviewed source model/loss/state/boundary modules', 'sciona.dfdc_codec', 'graph-injected row-normalization callable', 'numpy', 'torch', 'Pillow', 'sciona.ghost.registry (registration only)'],
        reuse='Source recipe adapter with an explicit callable port. Generic row scaling is independently selectable and contains no source recipe policy.'),
}


def run_tests():
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONPATH=str(ROOT))
    with tempfile.TemporaryDirectory(prefix='adversarial-provider-tests-') as temporary:
        junit = Path(temporary)/'results.xml'
        result = subprocess.run([str(ROOT/'.venv/bin/python'), '-m', 'pytest', '-q', *TESTS,
                                 '--junitxml='+str(junit)], cwd=ROOT, env=env, capture_output=True, text=True)
        require(result.returncode == 0, 'Provider tests failed: '+result.stdout[-2500:]+result.stderr[-1000:])
        suites = ET.parse(junit).getroot().findall('testsuite')
        counts = {k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ('tests','failures','errors','skipped')}
        require(counts == dict(tests=114, failures=0, errors=0, skipped=0), 'Expected complete 114-test provider qualification')
    report = dict(passed=True, **counts, synthetic_only=True,
                  test_sha256={p:sha(ROOT/p) for p in TESTS},
                  runner_sha256=sha(Path(__file__)))
    TEST_REPORT.write_text(json.dumps(report, indent=2)+'\n')
    return report


def review():
    intake = reviewed_plan()
    integration = integration_review()
    tests = json.loads(TEST_REPORT.read_text())
    require(tests == dict(passed=True, tests=114, failures=0, errors=0, skipped=0, synthetic_only=True,
        test_sha256={p:sha(ROOT/p) for p in TESTS}, runner_sha256=sha(Path(__file__))),
        'Provider test evidence missing or stale')
    requirements = ROOT/'requirements/adversarial-execution.txt'
    dependencies = dict(line.split('==') for line in requirements.read_text().splitlines()
                        if line.strip() and not line.startswith('#'))
    require(all(importlib.metadata.version(k)==v for k,v in dependencies.items()), 'Runtime packages changed')
    files = {str(p.relative_to(ROOT)):p for p in (ROOT/'sciona').glob('adversarial_*.py')}
    files.update({str(p.relative_to(ROOT)):p for p in (ROOT/'sciona').glob('adversarial_*topology.json')})
    files.update({p:ROOT/p for p in ('sciona/decomposed_adversarial_injected.py',
        'sciona/decomposed_adversarial_graph.py', 'sciona/dfdc_codec.py',
        'sciona/visualizer/runner.py', 'sciona/services/execution_graph_codec.py',
        'requirements/adversarial-execution.txt')})
    files.update({p:ROOT.parent/p for p in PROVIDER_FILES})
    imports = {name:sorted({ast.unparse(n) for n in ast.walk(ast.parse(path.read_text()))
                           if isinstance(n, (ast.Import, ast.ImportFrom))})
               for name,path in files.items() if path.suffix=='.py'}
    direct = ['sciona/decomposed_adversarial_injected.py', *PROVIDER_FILES]
    require(not any('sciona.atoms.' in line for name in direct for line in imports[name]),
            'Unexpected hidden registered-provider import')
    records = []
    for record in intake['providers']:
        contract = CONTRACTS[record['runtime_fqdn'].rsplit('.',1)[1]]
        require(set(contract['parameters']) == {p['name'] for p in record['interfaces'] if p['direction']=='input'},
                'Parameter semantic coverage differs')
        records.append(dict(record, **contract, internal_registered_provider_dependencies=[],
            callable_dependency_contracts=({'normalizer':'Graph must bind the reviewed row-scaling provider; this is a callable input, not a hardcoded internal atom import.'}
                                          if record['runtime_fqdn'].endswith('.execute_with_normalizer') else {})))
    evidence = ('adversarial_normalizer_provider_tests.json', 'adversarial_decomposed_semantic_review.json',
                'decomposed_adversarial_graph.json', 'adversarial_generic_row_scale_source_parity.json',
                'adversarial_normalizer_provider_intake_plan.json')
    return dict(format='adversarial-normalizer-provider-semantics.v1', passed=True, approved=False,
        review_source='automated', proposed_tier=3, verdict='acceptable_with_limits',
        providers=records, graph_digest=intake['graph_digest'],
        software_sha256={name:sha(path) for name,path in sorted(files.items())}, imports=imports,
        package_versions=dependencies, python_version=platform.python_version(),
        evidence_sha256={name:sha(ROOT/'docs/reviews'/name) for name in evidence},
        intake_stage_mapping=integration['source_stage_mapping'],
        limitations=[
            'Automated Tier 3 review only; no Tier 1 human certification or Tier 2 usage qualification.',
            'Generic scaling is domain independent; users determine the meaning of rows, columns, units and downstream use.',
            'Source adapter requires a qualified callable binding. Shape checks do not establish that an arbitrary callback implements the source equations.',
            'Full serialized graph evidence uses synthetic explicit model states; no pretrained effectiveness or historical TensorFlow binary parity claim.',
            'Source PNG min-max stretching does not preserve the normalized epsilon bound in saved pixels.',
            'No source runtime downloads. Numerical packages are pinned; clean installation remains untested.',
            'Validation runs serially with native pools and Torch intra/inter-op threads at one. Deployment must configure its own thread limits.',
            'Software modules and callable contracts are recorded separately from internal registered atom dependencies.',
            'This review does not approve catalog records; native metadata, transaction and served-selection gates remain.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-tests', action='store_true')
    if parser.parse_args().run_tests:
        run_tests()
    result = review()
    REPORT.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, providers=len(result['providers']), tests=114, approved=False)))
