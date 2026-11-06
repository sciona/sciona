"""Qualify the existing Matérn covariance provider for canonical Tier 3 publication."""
import json
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from scripts.stage_matern_canonical_records import reviewed_plan,ROOT,sha,atoms
from scripts.review_plasticc_matern_reuse import review as reuse
import base64
import hashlib
import importlib.metadata as metadata
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

LIMITS=[
    'Automated Tier3 qualification of normalized Matern32 covariance only; no human Tier1 or usage-qualified Tier2 designation.',
    'Qualified inputs are nonempty finite float64 coordinate matrices, optional matching query matrix, positive finite unsquared length scales and nu=1.5. Extreme scales or coordinates that overflow scaled distances are outside the qualified arithmetic range.',
    'Output is normalized covariance only. Amplitude scaling, measurement noise, GP optimization, optimizer fallback, cached state and prediction remain separate operations.',
    'Other smoothness values supported by the wider implementation are outside this review. Units and same-invocation provenance remain caller responsibilities.',
    'Synthetic covariance checks do not establish positive definiteness for arbitrary rounded or extreme inputs, application accuracy or historical fitting parity.',
    'Installed files match their own RECORD hashes; independent publisher authentication, clean deployment and other platforms are not certified.',
    'Legacy provider metadata and approval remain preserved. Canonical publication and PLAsTiCC original graph qualification remain separate actions.',
]


def packages():
    pending = ['numpy', 'scipy', 'george', 'icontract', 'threadpoolctl']
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
    intake=reviewed_plan()
    paths={p:ROOT/p for p in ('tests/test_matern_provider_reuse.py',
        'scripts/review_matern_provider_semantics.py','scripts/review_plasticc_matern_reuse.py',
        'scripts/stage_matern_canonical_records.py','scripts/audit_matern_provider_intake.py',
        'sciona/plasticc_observations.py')}
    paths['sciona-atoms-ml/src/sciona/atoms/ml/sklearn/gaussian_process/kernels/atoms.py']=ROOT.parent/'sciona-atoms-ml/src/sciona/atoms/ml/sklearn/gaussian_process/kernels/atoms.py'
    before={k:sha(p) for k,p in paths.items()};installed=packages();comparison=reuse()
    environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1',MPLCONFIGDIR='/private/tmp/sciona-mpl-cache',PYTHONPATH=str(ROOT))
    with tempfile.TemporaryDirectory(prefix='matern-synthetic-') as directory:
        xml=Path(directory)/'tests.xml'
        run=subprocess.run([str(ROOT/'.venv/bin/python'),'-m','pytest','-q','tests/test_matern_provider_reuse.py','--junitxml='+str(xml)],
            cwd=ROOT,env=environment,capture_output=True,text=True)
        if run.returncode:raise ValueError('Independent covariance tests failed: '+run.stdout[-3000:]+run.stderr[-1000:])
        suites=list(ET.parse(xml).getroot().iter('testsuite'))
        counts={k:sum(int(s.get(k,'0')) for s in suites) for k in ('tests','failures','errors','skipped')}
    if counts!=dict(tests=20,failures=0,errors=0,skipped=0):raise ValueError('Complete 20-test suite required')
    if before!={k:sha(p) for k,p in paths.items()} or installed!=packages():raise ValueError('Qualified software changed')
    if intake!=reviewed_plan():raise ValueError('Qualified intake changed')
    record=intake['records'][0]
    provider=dict(artifact_id=record['artifact']['artifact_id'],fqdn=record['artifact']['fqdn'],
        version_id=record['version']['version_id'],content_hash=record['version']['content_hash'],interfaces=record['interfaces'],
        operation='Compute normalized Matern32 covariance between numeric coordinate rows with explicit scalar or anisotropic length scales.',
        plain_description='Compare numeric points using their scaled distance and return their pairwise similarity values.',
        validity_bounds=' '.join(p['constraints'] for p in record['interfaces']),
        reuse='Domain-independent numeric covariance with explicit coordinate scales; no astronomy bands, redshift, labels or observation-table semantics.',
        registered_provider_dependencies=[])
    return dict(format='matern-provider-semantics.v1',passed=True,approved=False,proposed_tier=3,
        verdict='acceptable_with_limits',catalog_mutations=0,synthetic_only=True,providers=[provider],tests=counts,
        covariance_reuse_cases=comparison['cases'],software_sha256=before,installed_packages=installed,
        intake_plan_sha256=sha(ROOT/'docs/reviews/matern_canonical_intake_plan.json'),
        resource_policy=dict(serial_tests=True,native_threads=1),limitations=LIMITS)


if __name__=='__main__':
    result=review()
    (ROOT/'docs/reviews/matern_provider_semantics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,tests=result['tests'],covariance_cases=len(result['covariance_reuse_cases']),approved=False)))
