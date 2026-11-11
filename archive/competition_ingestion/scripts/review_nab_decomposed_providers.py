"""Qualify provider semantics and software dependencies without publishing."""
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

from scripts.decomposed_nab_provider_contracts import CONTRACTS
from scripts.stage_nab_decomposed_providers import reviewed_plan
from scripts.review_decomposed_nab import ROOT, require, sha, review as integration_review

TESTS = tuple(str(p.relative_to(ROOT)) for p in sorted((ROOT/'tests').glob('test_nab_*.py'))) + tuple(
    'tests/test_decomposed_nab_'+name+'.py' for name in ('runtime', 'review'))
TEST_REPORT = ROOT/'docs/reviews/nab_decomposed_provider_tests.json'
REPORT = ROOT/'docs/reviews/nab_decomposed_provider_semantics.json'
PROVIDER_FILES = ('sciona-atoms/src/sciona/atoms/numerical/statistics/prior_threshold.py',
                 'sciona-atoms-ml/src/sciona/atoms/ml/decomposed_nab_execution.py')


def run_tests():
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONPATH=str(ROOT))
    with tempfile.TemporaryDirectory(prefix='nab-provider-tests-') as temporary:
        junit = Path(temporary)/'results.xml'
        result = subprocess.run([str(ROOT/'.venv/bin/python'), '-m', 'pytest', '-q', *TESTS,
                                 '--junitxml='+str(junit)], cwd=ROOT, env=env, capture_output=True, text=True)
        require(result.returncode == 0, 'Provider tests failed: '+result.stdout[-2500:]+result.stderr[-1000:])
        suites = ET.parse(junit).getroot().findall('testsuite')
        counts = {k: sum(int(s.attrib.get(k, 0)) for s in suites)
                  for k in ('tests', 'failures', 'errors', 'skipped')}
        require(counts == dict(tests=79, failures=0, errors=0, skipped=0), 'Complete 79-test qualification required')
    report = dict(passed=True, **counts, synthetic_only=True,
        test_sha256={p: sha(ROOT/p) for p in TESTS}, runner_sha256=sha(Path(__file__)))
    TEST_REPORT.write_text(json.dumps(report, indent=2)+'\n')
    return report


def review():
    intake = reviewed_plan()
    integration = integration_review()
    require(json.loads(TEST_REPORT.read_text()) == dict(passed=True, tests=79, failures=0,
        errors=0, skipped=0, synthetic_only=True, test_sha256={p: sha(ROOT/p) for p in TESTS},
        runner_sha256=sha(Path(__file__))), 'Provider test evidence missing or stale')
    from scripts.audit_decomposed_nab_packages import audit as package_audit
    packages = package_audit()
    require(packages == json.loads((ROOT/'docs/reviews/decomposed_nab_package_files.json').read_text()),
            'Installed package closure or files changed')
    dependencies = integration['package_versions']
    files = {str(p.relative_to(ROOT)): p for p in (ROOT/'sciona').glob('nab_*.py')}
    files.update({p: ROOT/p for p in ('sciona/decomposed_nab_runtime.py',
        'sciona/decomposed_nab_graph.py', 'sciona/visualizer/runner.py',
        'sciona/services/execution_graph_codec.py',
        'scripts/audit_decomposed_nab_packages.py', 'scripts/review_decomposed_nab.py',
        'scripts/stage_nab_decomposed_providers.py',
        'scripts/decomposed_nab_provider_contracts.py', 'scripts/review_nab_decomposed_providers.py')})
    files.update({p: ROOT.parent/p for p in PROVIDER_FILES})
    imports = {name: sorted({ast.unparse(n) for n in ast.walk(ast.parse(path.read_text()))
        if isinstance(n, (ast.Import, ast.ImportFrom))})
        for name, path in files.items() if path.suffix == '.py'}
    direct = ['sciona/decomposed_nab_runtime.py', *PROVIDER_FILES]
    require(not any('sciona.atoms.' in line for name in direct for line in imports[name]),
            'Unexpected hidden registered-provider import')
    records = []
    for record in intake['providers']:
        contract = CONTRACTS[record['runtime_fqdn'].rsplit('.', 1)[1]]
        require(set(contract['parameters']) == {p['name'] for p in record['interfaces'] if p['direction'] == 'input'},
                'Parameter semantic coverage differs')
        records.append(dict(record, **contract, internal_registered_provider_dependencies=[],
                            callable_dependency_contracts={}))
    require(len(records) == len(CONTRACTS) == 5, 'Five provider contracts required')
    evidence = ('nab_decomposed_provider_tests.json', 'nab_decomposed_semantic_review.json',
        'decomposed_nab_graph_execution.json', 'nab_decomposed_provider_intake_gates.json',
        'decomposed_nab_package_files.json', 'nab_decomposed_provider_intake_plan.json')
    return dict(format='nab-decomposed-provider-semantics.v1', passed=True, approved=False,
        review_source='automated', proposed_tier=3, verdict='acceptable_with_limits',
        providers=records, graph_digest=intake['graph_digest'],
        software_sha256={name: sha(path) for name, path in sorted(files.items())}, imports=imports,
        package_versions=dependencies, python_version=platform.python_version(),
        evidence_sha256={name: sha(ROOT/'docs/reviews'/name) for name in evidence},
        intake_stage_mapping=integration['source_stage_mapping'],
        limitations=integration['limitations']+[
            'Automated Tier 3 only; no Tier 1 human certification or Tier 2 usage qualification.',
            'Rolling prior location-scale threshold is domain independent. The recipe supplies history controls and multiplier three through explicit graph edges.',
            'Synthetic execution and pinned scoring comparisons do not establish historical detector or generalization accuracy parity.',
            'Installed package hashes were checked against their own RECORD files, not independently authenticated publisher signatures. Clean deployment qualification is not claimed.',
            'Validation bounds native thread pools and forest jobs to one; deployment must configure its own resource limits.',
            'Pinned public NAB software is a scoring qualification reference. The runtime uses an independently implemented corrected scorer and explicit isolation-forest recipe.',
            'Software dependencies are separate from registered atom dependencies. This review alone does not approve catalog records.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-tests', action='store_true')
    if parser.parse_args().run_tests:
        run_tests()
    result = review()
    REPORT.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, providers=len(result['providers']), tests=79, approved=False)))
