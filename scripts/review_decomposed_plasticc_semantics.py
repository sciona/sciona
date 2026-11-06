"""Automated semantic qualification of proposed PLAsTiCC Tier3 providers."""
import base64
import hashlib
import importlib.metadata as metadata
import inspect
import json
import os
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from scripts.review_decomposed_plasticc_interfaces import review as interfaces, pinned, ROOT, sha
from scripts.validate_decomposed_plasticc_graph_v2 import validate as execute_graph
from scripts.decomposed_plasticc_provider_contracts import CONTRACTS

TESTS = ['tests/test_decomposed_plasticc_'+name+'.py' for name in
         ('gp', 'gp_fit', 'sampling', 'training', 'prediction', 'observations', 'registered', 'graph')]
PINNED = {
    'docs/reviews/decomposed_plasticc_interfaces.json': '81a97318',
    'docs/reviews/decomposed_plasticc_graph_execution_v2.json': 'b5a0a8be',
}
LIMITS = [
    'Automated Tier3 qualification only; no human Tier1 review or usage-qualified Tier2 designation.',
    'Publication remains separate and must bind these exact providers, contracts, software and graph.',
    'Complete configured synthetic runs use three folds, one augmentation attempt per original and a60-tree ceiling; historical scale and historical competition accuracy are not established.',
    'Generic GP coordinates, categorical reference grouping and arbitrary-width multiclass fitting have separate synthetic evidence. Astronomy augmentation, features, redshift weights and final class policy remain explicit restrictions.',
    'Mutable state and callable types do not authenticate same-invocation provenance; exact graph bindings and qualified inputs are required.',
    'Validation influences early stopping. OOF diagnostics do not establish unbiased final generalization or unknown-class detection.',
    'The source normalization uses unshifted exponentials; nonfinite outcomes reject rather than switching algorithms. GP optimizer failure restores initial effective parameters but preserves reported diagnostics.',
    'Installed files are checked against their own distribution RECORD; publisher authentication, clean deployment and other platforms are not certified.',
    'Reuse review covers the already-published Matern32 provider and retained whole-pipeline implementation, not exhaustive equivalence with every catalog item.',
]


def packages():
    pending = ['numpy', 'scipy', 'pandas', 'lightgbm', 'scikit-learn', 'astropy', 'george', 'tqdm', 'icontract', 'pydantic', 'threadpoolctl']
    result = {}
    while pending:
        name = canonicalize_name(pending.pop())
        if name in result: continue
        dist = metadata.distribution(name); requirements = []
        for raw in dist.requires or []:
            req = Requirement(raw)
            if req.marker and not req.marker.evaluate({'extra': ''}): continue
            if req.extras or req.url: raise ValueError('Unreviewed extra or URL requirement')
            if not req.specifier.contains(metadata.version(req.name), prereleases=True):
                raise ValueError('Unsatisfied runtime requirement: '+str(req))
            requirements.append(str(req)); pending.append(req.name)
        verified = 0
        for relative in dist.files or []:
            if relative.hash is None: continue
            with dist.locate_file(relative).open('rb') as stream:
                digest = hashlib.file_digest(stream, relative.hash.mode).digest()
            if base64.urlsafe_b64encode(digest).rstrip(b'=').decode() != relative.hash.value:
                raise ValueError('Installed RECORD mismatch: '+name)
            verified += 1
        if not verified: raise ValueError('No hashed files: '+name)
        result[name] = dict(version=dist.version, hashed_files_verified=verified, active_requirements=sorted(requirements))
    return dict(sorted(result.items()))


def review():
    frozen = {path: pinned(path, commit) for path, commit in PINNED.items()}
    original = frozen['docs/reviews/decomposed_plasticc_interfaces.json']
    if interfaces() != original: raise ValueError('Reviewed interfaces or catalog state changed')
    execution = frozen['docs/reviews/decomposed_plasticc_graph_execution_v2.json']
    paths = {p: ROOT/p for p in TESTS + list(execution['implementation_sha256']) + list(original['code_sha256'])}
    for name in ('decomposed_plasticc_provider_contracts.py', 'review_decomposed_plasticc_semantics.py'):
        paths['scripts/'+name] = ROOT/'scripts'/name
    for name in ('decomposed_plasticc_numeric.py', 'decomposed_plasticc_lifecycle.py', 'sklearn/gaussian_process/kernels/atoms.py'):
        key = 'sciona-atoms-ml/src/sciona/atoms/ml/'+name
        paths[key] = ROOT.parent/key
    before = {p: sha(path) for p, path in paths.items()}
    installed = packages()
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MPLCONFIGDIR='/private/tmp/sciona-mpl-cache', PYTHONPATH=str(ROOT))
    with tempfile.TemporaryDirectory(prefix='plasticc-semantic-') as directory:
        xml = Path(directory)/'tests.xml'
        run = subprocess.run([str(ROOT/'.venv/bin/python'), '-m', 'pytest', '-q', *TESTS, '--junitxml='+str(xml)],
            cwd=ROOT, env=environment, capture_output=True, text=True)
        if run.returncode: raise ValueError('Synthetic qualification failed: '+run.stdout[-4000:]+run.stderr[-1000:])
        suites = list(ET.parse(xml).getroot().iter('testsuite'))
        counts = {k: sum(int(s.get(k, '0')) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
    if counts != dict(tests=67, failures=0, errors=0, skipped=0):
        raise ValueError('Exact complete67-test suite required: '+str(counts))
    if execute_graph() != execution: raise ValueError('Complete registered graph evidence changed')
    providers = []
    for record in original['new_providers']:
        name = record['runtime_fqdn'].rsplit('.', 1)[-1]
        contract = CONTRACTS[name]
        names = [p['name'] for p in record['interfaces'] if p['direction'] == 'input']
        if list(contract['parameters']) != names or not all(contract.values()) or not all(contract['parameters'].values()):
            raise ValueError('Exact complete semantic parameter contract required')
        providers.append(dict(record, **contract))
    if len(providers) != len(CONTRACTS) or len(providers) != 24: raise ValueError('Complete24-provider qualification required')
    if before != {p: sha(path) for p, path in paths.items()} or packages() != installed:
        raise ValueError('Qualified software changed during review')
    if interfaces() != original: raise ValueError('Catalog state changed during review')
    return dict(format='decomposed-plasticc-semantics.v1', passed=True, approved=False, proposed_tier=3,
        verdict='acceptable_with_limits', catalog_mutations=0, synthetic_only=True, providers=providers,
        reused_providers=original['existing_providers'], tests=counts, graph_digest=execution['graph_sha256'],
        graph_cases=execution['cases'], software_sha256=before, installed_packages=installed,
        evidence_sha256={path: sha(ROOT/path) for path in PINNED}, limitations=LIMITS,
        resource_policy=dict(serial_tests=True, serial_models=True, native_threads=1))


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/decomposed_plasticc_semantics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, tests=result['tests'], providers=len(result['providers']), approved=False)))
