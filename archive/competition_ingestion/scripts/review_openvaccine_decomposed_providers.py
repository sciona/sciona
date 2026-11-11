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

from scripts.decomposed_openvaccine_provider_contracts import CONTRACTS
from scripts.stage_openvaccine_decomposed_providers import reviewed_plan
from scripts.review_decomposed_openvaccine_implementation import ROOT, require, sha, review as integration_review

TESTS = tuple(str(p.relative_to(ROOT)) for p in sorted((ROOT/'tests').glob('test_openvaccine_*.py'))) + tuple(
    'tests/test_decomposed_openvaccine_'+name+'.py' for name in ('runtime', 'ensemble', 'interfaces', 'implementation'))
TEST_REPORT = ROOT/'docs/reviews/openvaccine_decomposed_provider_tests.json'
REPORT = ROOT/'docs/reviews/openvaccine_decomposed_provider_semantics.json'
PROVIDER_FILES = ('sciona-atoms-ml/src/sciona/atoms/ml/decomposed_openvaccine_execution.py',)


def run_tests():
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONPATH=str(ROOT))
    with tempfile.TemporaryDirectory(prefix='openvaccine-provider-tests-') as temporary:
        junit = Path(temporary)/'results.xml'
        result = subprocess.run([str(ROOT/'.venv/bin/python'), '-m', 'pytest', '-q', *TESTS,
                                 '--junitxml='+str(junit)], cwd=ROOT, env=env, capture_output=True, text=True)
        require(result.returncode == 0, 'Provider tests failed: '+result.stdout[-2500:]+result.stderr[-1000:])
        suites = ET.parse(junit).getroot().findall('testsuite')
        counts = {k: sum(int(s.attrib.get(k, 0)) for s in suites)
                  for k in ('tests', 'failures', 'errors', 'skipped')}
        require(counts == dict(tests=190, failures=0, errors=0, skipped=0), 'Complete 190-test qualification required')
    report = dict(passed=True, **counts, synthetic_only=True,
        test_sha256={p: sha(ROOT/p) for p in TESTS}, runner_sha256=sha(Path(__file__)))
    TEST_REPORT.write_text(json.dumps(report, indent=2)+'\n')
    return report


def review():
    intake = reviewed_plan()
    integration = integration_review()
    require(json.loads(TEST_REPORT.read_text()) == dict(passed=True, tests=190, failures=0,
        errors=0, skipped=0, synthetic_only=True, test_sha256={p: sha(ROOT/p) for p in TESTS},
        runner_sha256=sha(Path(__file__))), 'Provider test evidence missing or stale')
    from scripts.audit_decomposed_openvaccine_packages import audit as package_audit
    packages = package_audit()
    require(packages == json.loads((ROOT/'docs/reviews/decomposed_openvaccine_package_files.json').read_text()),
            'Installed package closure or files changed')
    dependencies = integration['runtime']
    files = {str(p.relative_to(ROOT)): p for p in (ROOT/'sciona').glob('openvaccine_*.py')}
    files.update({p: ROOT/p for p in ('sciona/decomposed_openvaccine_runtime.py',
        'sciona/decomposed_openvaccine_graph.py', 'sciona/decomposed_openvaccine_ensemble.py', 'sciona/visualizer/runner.py',
        'sciona/services/execution_graph_codec.py',
        'scripts/audit_decomposed_openvaccine_packages.py', 'scripts/review_decomposed_openvaccine_implementation.py',
        'scripts/stage_openvaccine_decomposed_providers.py',
        'scripts/decomposed_openvaccine_provider_contracts.py', 'scripts/review_openvaccine_decomposed_providers.py')})
    files.update({p: ROOT.parent/p for p in PROVIDER_FILES})
    imports = {name: sorted({ast.unparse(n) for n in ast.walk(ast.parse(path.read_text()))
        if isinstance(n, (ast.Import, ast.ImportFrom))})
        for name, path in files.items() if path.suffix == '.py'}
    direct = ['sciona/decomposed_openvaccine_runtime.py', 'sciona/decomposed_openvaccine_ensemble.py', *PROVIDER_FILES]
    require(not any('sciona.atoms.' in line for name in direct for line in imports[name]),
            'Unexpected hidden registered-provider import')
    records = []
    for record in intake['providers']:
        contract = CONTRACTS[record['runtime_fqdn'].rsplit('.', 1)[1]]
        require(set(contract['parameters']) == {p['name'] for p in record['interfaces'] if p['direction'] == 'input'},
                'Parameter semantic coverage differs')
        records.append(dict(record, **contract, internal_registered_provider_dependencies=[],
                            callable_dependency_contracts={}))
    require(len(records) == len(CONTRACTS) == 4, 'Four provider contracts required')
    evidence = ('openvaccine_decomposed_provider_tests.json', 'decomposed_openvaccine_implementation_review.json',
        'decomposed_openvaccine_graph_execution.json', 'openvaccine_decomposed_provider_intake_gates.json',
        'decomposed_openvaccine_package_files.json', 'openvaccine_decomposed_provider_intake_plan.json')
    return dict(format='openvaccine-decomposed-provider-semantics.v1', passed=True, approved=False,
        review_source='automated', proposed_tier=3, verdict='acceptable_with_limits',
        providers=records, graph_digest=intake['graph_digest'],
        software_sha256={name: sha(path) for name, path in sorted(files.items())}, imports=imports,
        package_versions=dependencies, python_version=platform.python_version(),
        evidence_sha256={name: sha(ROOT/'docs/reviews'/name) for name in evidence},
        source_qualification_complete=False,
        original_cdg_publication_qualified=False,
        limitations=[
            'Automated Tier 3 implementation semantics only; no Tier 1 human certification or Tier 2 usage qualification.',
            'Four recipe adapters preserve all three populations, full shared autoencoder pretraining, twenty split models, all caller-configured steps and refinement rounds, checkpoint restoration and unclipped teacher moments. RNA folding is separate from compatible dense-feature training; the generic ordered sum remains independently bound and domain independent.',
            'Recovered source is inference-only; the explicit training reconstruction uses caller-defined populations and controls. This review qualifies the stated implementation contracts, not historical optimization schedules, the original winning clustered blend or original-CDG publication.',
            'Fresh execution exactly matches the existing reconstruction on generated populations; historical population membership and competitive accuracy are unproven.',
            'Installed package hashes were checked against their own RECORD files. Clean deployment and independently authenticated publisher signatures are not qualified.',
            'Native and TensorFlow pools were bounded to one thread during full validation; models train serially. Deployment must configure its own resource limits and isolate process-global seed changes.',
            'Software imports are separate from registered-provider dependencies. Same-invocation intermediate provenance is required but not authenticated by structural checks.',
            'Catalog eligibility, publication transaction gates and existing-provider metadata remain separate checks.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-tests', action='store_true')
    if parser.parse_args().run_tests:
        run_tests()
    result = review()
    REPORT.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, providers=len(result['providers']), tests=190, approved=False)))
