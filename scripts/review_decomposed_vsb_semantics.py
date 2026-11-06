"""Qualify decomposed VSB semantics using synthetic evidence only."""
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
from scripts.audit_decomposed_vsb_readiness import ROOT, sha
from scripts.review_decomposed_vsb_interfaces import review as interfaces
from scripts.validate_decomposed_vsb_graph import validate as execute_graph

TESTS = ['tests/test_decomposed_vsb_'+s+'.py' for s in ('features','training','contract')]
TESTS += ['tests/test_vsb_'+s+'.py' for s in
          ('baseline','peaks','descriptors','measurement','threshold','training','execution')]
TESTS += ['tests/test_lfilter_provider_reuse.py']
PINNED = {
    'docs/reviews/decomposed_vsb_interfaces.json': '3b0ee7d1',
    'docs/reviews/decomposed_vsb_graph_execution.json': '5dacb736',
    'docs/reviews/vsb_original_scope.json': '6e093e16',
    'docs/reviews/vsb_current_baseline.json': '1c44c993',
}
CANDIDATES = {
    'sciona-atoms/src/sciona/atoms/numpy/fft.py':
        'Registered rfft returns a complete spectrum with configurable axis, length, normalization and output buffer. Fundamental is a validated finite real matrix projection to the first positive coefficient along axis0; rfft alone is not a drop-in replacement. Synthetic comparison verifies the exact projection relation. No claim that this wrapper is a distinct Fourier algorithm.',
    'sciona-atoms/src/sciona/atoms/scipy/signal.py':
        'Existing lfilter is actually reused as an explicit callable node with canonical Tier3 approval. Matching coefficient and delay construction and finite residual checking remain separate adapters.',
    'sciona-atoms-ml/src/sciona/atoms/ml/model_selection/nested_stratified_masks.py':
        'stratified_inner_masks uses unshuffled balanced StratifiedKFold over an outer selection, returning two boolean masks. Repeated_folds uses class-specific RandomState assignments to five bins, two rotating heldouts and three index vectors. The partition policies and outputs are not interchangeable.',
    'sciona-atoms-ml/src/sciona/atoms/ml/model_selection/stratified_kfold_cross_validation/atoms.py':
        'compute_stratified_kfold_indices intends balanced train/test pairs, not the qualified random three-way repeated lifecycle. Its implementation passes targets to the StratifiedKFold constructor and its postcondition references an undefined folds name; it is not selected as an execution dependency.',
    'sciona-atoms-ml/src/sciona/atoms/ml/vsb_execution.py':
        'Existing prepare/execute providers remain unchanged reference implementations. Reusing the whole execute stage would conceal individually selectable feature, fold, threshold and grouping operations.',
}
LIMITS = [
    'Automated Tier3 qualification only; provider publication and original CDG approval remain pending.',
    'Candidate review covers the named nearby implementations, not exhaustive catalog semantic equivalence.',
    'FFT projection, numeric peak/phase operations, MCC threshold selection and generic-width fold fitting are domain-independent. Consecutive triples, nine-feature order, fixed tree hyperparameters and five-way coverage are explicit recipe restrictions.',
    'Synthetic cases execute all125 models with the10000-round ceiling and100-round stopping patience; early stopping determines actual iterations. No historical accuracy, historical floating-point replay or generalization claim.',
    'Both heldouts affect early stopping; threshold fitting uses in-training scores. The optimizer uses >= while final decisions use strict > and repeat across three signals.',
    'Numeric shape checks and callability do not authenticate same-invocation provenance or callable semantics; exact registered graph bindings are required.',
    'Actual-length waveform clipping and one-cycle FFT interpretation are declared independent generalizations. Source notebook code is not copied or licensed by this review.',
    'Installed files match their own RECORD hashes; independent publisher authentication, clean deployment and other platforms are not certified.',
]


def packages():
    pending = ['numpy', 'scipy', 'pandas', 'lightgbm', 'scikit-learn', 'icontract', 'pydantic', 'threadpoolctl']
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
    original = frozen['docs/reviews/decomposed_vsb_interfaces.json']
    current = interfaces()
    if current != original:
        raise ValueError('Previously reviewed interfaces changed')
    execution_evidence = frozen['docs/reviews/decomposed_vsb_graph_execution.json']
    paths = {p: ROOT/p for p in TESTS}
    for p in execution_evidence['implementation_sha256']: paths[p] = ROOT/p
    for p in original['code_sha256']: paths[p] = ROOT/p
    paths['scripts/review_decomposed_vsb_semantics.py'] = Path(__file__)
    for p in CANDIDATES: paths[p] = ROOT.parent/p
    provider = 'sciona-atoms-ml/src/sciona/atoms/ml/decomposed_vsb_execution.py'
    paths[provider] = ROOT.parent/provider
    before = {key:sha(path) for key,path in paths.items()}
    installed = packages()
    from sciona.atoms.numpy.fft import rfft
    from sciona.vsb_measurement import fundamental
    from sciona.atoms.ml.model_selection.nested_stratified_masks import stratified_inner_masks
    from sciona.vsb_threshold import repeated_folds
    values = np.random.default_rng(18).normal(size=(129,7))
    spectrum = rfft(values, axis=0)
    if spectrum.shape != (65,7) or not np.array_equal(spectrum[1], fundamental(values)):
        raise ValueError('FFT projection relation changed')
    labels = np.repeat(np.array([0,1], dtype=np.int64),100)
    _, candidate = stratified_inner_masks(labels, np.ones(200,dtype=bool), 5, 0)
    plan = repeated_folds(labels, seed=123948, repetitions=1)[0]
    if np.array_equal(np.flatnonzero(candidate), plan[1]) or len(plan) != 3:
        raise ValueError('Distinct fold policy check failed')
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
        MPLCONFIGDIR='/private/tmp/sciona-mpl-cache', PYTHONPATH=str(ROOT))
    with tempfile.TemporaryDirectory(prefix='vsb-semantic-') as directory:
        xml = Path(directory)/'tests.xml'
        run = subprocess.run([str(ROOT/'.venv/bin/python'),'-m','pytest','-q',*TESTS,
            '--junitxml='+str(xml)],cwd=ROOT,env=environment,capture_output=True,text=True)
        if run.returncode:
            raise ValueError('Synthetic qualification failed: '+run.stdout[-3000:]+run.stderr[-1000:])
        suites = list(ET.parse(xml).getroot().iter('testsuite'))
        counts = {key:sum(int(s.get(key,'0')) for s in suites)
            for key in ('tests','failures','errors','skipped')}
    if counts != dict(tests=151,failures=0,errors=0,skipped=0):
        raise ValueError('Complete 151-test suite required')
    execution = execute_graph()
    if execution != execution_evidence:
        raise ValueError('Full graph evidence differs')
    if before != {key:sha(path) for key,path in paths.items()} or installed != packages():
        raise ValueError('Qualification software changed')
    if current != interfaces():
        raise ValueError('Provider inventory changed')
    return dict(format='decomposed-vsb-semantics.v1',passed=True,approved=False,
        proposed_tier=3,catalog_mutations=0,synthetic_only=True,
        providers=current['new_providers'],existing_providers=current['existing_providers'],
        tests=counts,execution=execution,candidate_review=CANDIDATES,
        candidate_checks=dict(fft_projection_equal=True,fold_policy_differs=True,existing_filter_reused=True),
        software_sha256=before,installed_packages=installed,
        evidence_sha256={p:sha(ROOT/p) for p in PINNED},
        resource_policy=dict(serial_tests=True,serial_workers=True,native_threads=1),
        limitations=LIMITS)


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/decomposed_vsb_semantics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,providers=len(result['providers']),tests=result['tests'],
        graph_cases=len(result['execution']['cases']),approved=False)))
