"""Qualify decomposed Web Traffic semantics using synthetic evidence only."""
import base64
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
import numpy as np
from scripts.audit_decomposed_webtraffic_readiness import ROOT, sha
from scripts.review_decomposed_webtraffic_interfaces import review as interfaces
from scripts.validate_decomposed_webtraffic_graph import validate as execute_graph

TESTS = ['tests/test_panel_output.py'] + [
    'tests/test_decomposed_webtraffic_'+s+'.py' for s in ('features','lifecycle','contract')]
PINNED = {
    'docs/reviews/decomposed_webtraffic_interfaces.json': 'bf3e77d7',
    'docs/reviews/decomposed_webtraffic_graph_execution.json': '15f322b8',
    'docs/reviews/webtraffic_original_output_scope.json': '1c0dbbbd',
    'docs/reviews/webtraffic_current_baseline.json': '5c3e73c0',
}
CANDIDATES = {
    'sciona-atoms-signal/src/sciona/atoms/riemannian_bci/covariance_features/atoms.py':
        'autocorrelation_covariance_matrix returns a channel-by-lag Hankel covariance matrix. It does not return weighted neighboring-lag correlations with row support masks and population normalization.',
    'sciona-atoms-dl/src/sciona/atoms/dl/graph/atoms.py':
        'feature_clip_standardize adds 1e-5 to population standard deviation and clips outputs. Correlations preserve undefined constant populations; popularity standardizes row medians with sample standard deviation. Neither is interchangeable with the clipped feature transform.',
    'sciona-atoms-ml/src/sciona/atoms/ml/sklearn/neural_network/mlp_optimizers/atoms.py':
        'mlp_adam_updates has related epsilon-hat Adam mathematics, but casts to float64, computes beta powers from an integer step, and returns deltas without global gradient clipping. The qualified float32 lifecycle maintains explicit beta-power tensors, clips global norm at 10, and observes three models and EMA phases. Substitution requires a separately qualified adapter and numerical-order review.',
    'sciona-atoms-dl/src/sciona/atoms/dl/webtraffic_execution.py':
        'Existing whole prepare/train/forecast providers remain the unchanged parity reference. Their reuse as whole stages would conceal the separately selectable numeric and output operations.',
}
LIMITS = [
    'Automated Tier 3 semantic qualification only; provider publication and original CDG approval remain pending.',
    'Candidate review covers the named nearby implementations, not an exhaustive catalog equivalence proof.',
    'Numeric correlation, row-median normalization and integer suffix alignment accept domain-independent inputs. Page grammar, calendar lag policy, fixed s32 dimensions and three-model lifecycle retain explicit recipe assumptions.',
    'Synthetic short schedules execute every configured update, checkpoint and ensemble member. They do not establish historical training duration, winning accuracy or generalization.',
    'NumPy/PyTorch random streams and explicit EMA observation phases are declared adaptations; no identical-seed TensorFlow replay is claimed.',
    'The output boundary accepts a prealigned identifier matrix and suffix coordinates. Original key-file parsing, historical cutoff constants and file serialization are not implemented by this graph.',
    'State and feature validation does not authenticate same-invocation provenance. Training streams are mutable and single-use; failed consumption requires fresh initialization.',
    'Installed files match their own RECORD hashes. Independent publisher authentication, clean deployment and other platforms are not certified.',
]


def packages():
    pending = ['numpy', 'pandas', 'torch', 'icontract', 'pydantic', 'threadpoolctl']
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
            path = dist.locate_file(relative)
            with path.open('rb') as stream: digest = hashlib.file_digest(stream, relative.hash.mode).digest()
            if base64.urlsafe_b64encode(digest).rstrip(b'=').decode() != relative.hash.value:
                raise ValueError('Installed RECORD mismatch: '+name)
            verified += 1
        if not verified: raise ValueError('No hashed files: '+name)
        result[name] = dict(version=dist.version, hashed_files_verified=verified, active_requirements=sorted(requirements))
    return dict(sorted(result.items()))



def review():
    frozen = {}
    for path, commit in PINNED.items():
        raw = (ROOT/path).read_bytes()
        if raw != subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT):
            raise ValueError('Immutable evidence changed: '+path)
        frozen[path] = json.loads(raw)
    original = frozen['docs/reviews/decomposed_webtraffic_interfaces.json']
    current = interfaces()
    if current != original:
        raise ValueError('Previously reviewed interfaces changed')
    execution_evidence = frozen['docs/reviews/decomposed_webtraffic_graph_execution.json']
    paths = {p: ROOT/p for p in TESTS}
    for p in execution_evidence['implementation_sha256']: paths[p] = ROOT/p
    for p in original['code_sha256']: paths[p] = ROOT/p
    paths['scripts/review_decomposed_webtraffic_semantics.py'] = Path(__file__)
    for p in CANDIDATES: paths[p] = ROOT.parent/p
    provider = 'sciona-atoms-dl/src/sciona/atoms/dl/decomposed_webtraffic_execution.py'
    paths[provider] = ROOT.parent/provider
    before = {key:sha(path) for key,path in paths.items()}
    installed = packages()
    # Concrete distinction: candidate repairs constant columns whereas the
    # source numerical operation deliberately preserves undefined populations.
    from sciona.atoms.dl.graph.atoms import feature_clip_standardize
    from sciona.decomposed_webtraffic_features import correlations
    values = np.ones((3,20), dtype=np.float64)
    candidate = feature_clip_standardize(values)
    actual = correlations(values, np.zeros(3,dtype=np.int64),
        np.full(3,19,dtype=np.int64), [2], [1.], 0)
    if not np.all(candidate == 0) or not np.isnan(actual).all():
        raise ValueError('Constant-population semantic distinction changed')
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
        MPLCONFIGDIR='/private/tmp/sciona-mpl-cache', PYTHONPATH=str(ROOT))
    with tempfile.TemporaryDirectory(prefix='webtraffic-semantic-') as directory:
        xml = Path(directory)/'tests.xml'
        run = subprocess.run([str(ROOT/'.venv/bin/python'),'-m','pytest','-q',*TESTS,
            '--junitxml='+str(xml)],cwd=ROOT,env=environment,capture_output=True,text=True)
        if run.returncode:
            raise ValueError('Synthetic qualification failed: '+run.stdout[-3000:]+run.stderr[-1000:])
        suites = list(ET.parse(xml).getroot().iter('testsuite'))
        counts = {key:sum(int(s.get(key,'0')) for s in suites)
            for key in ('tests','failures','errors','skipped')}
    if counts != dict(tests=52,failures=0,errors=0,skipped=0):
        raise ValueError('Complete 52-test suite required')
    execution = execute_graph()
    if execution != execution_evidence:
        raise ValueError('Full graph evidence differs')
    if before != {key:sha(path) for key,path in paths.items()} or installed != packages():
        raise ValueError('Qualification software changed')
    if current != interfaces():
        raise ValueError('Provider inventory changed')
    return dict(format='decomposed-webtraffic-semantics.v1',passed=True,approved=False,
        proposed_tier=3,catalog_mutations=0,synthetic_only=True,
        providers=current['new_providers'],existing_providers=current['existing_providers'],
        tests=counts,execution=execution,candidate_review=CANDIDATES,
        candidate_checks=dict(constant_population_semantics_differ=True),
        software_sha256=before,installed_packages=installed,
        evidence_sha256={p:sha(ROOT/p) for p in PINNED},
        resource_policy=dict(serial_tests=True,serial_workers=True,native_threads=1),
        limitations=LIMITS)


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/decomposed_webtraffic_semantics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,providers=len(result['providers']),tests=result['tests'],
        graph_cases=len(result['execution']['cases']),approved=False)))
