"""Qualify the existing reusable filter for canonical Tier 3 publication."""
import json
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from scripts.stage_lfilter_canonical_records import reviewed_plan,ROOT,sha,atoms
from scripts.review_vsb_lfilter_reuse import review as reuse
import base64
import hashlib
import importlib.metadata as metadata
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

LIMITS=[
    'Automated Tier 3 qualification of the explicit real rank-one subset only; no human Tier 1 or usage-qualified Tier 2 designation.',
    'Finite nonempty float64 coefficients and signal, nonzero leading denominator, axis -1 or 0, and compatible finite optional delay state are caller responsibilities where the existing provider does not enforce them.',
    'The wrapper returns only filtered samples, discarding final delay state even when initial delay is supplied. It is not a state-returning streaming interface.',
    'Unstable filters and numerical overflow may produce nonfinite results without rejection. This qualification does not promise stability or bounded outputs; composition adapters must check required finiteness.',
    'Synthetic recurrence and baseline parity do not establish performance on real signals, historical precision, or application-specific validity.',
    'Installed files match their own RECORD hashes; independent publisher authentication, clean deployment and other platforms are not certified.',
    'Legacy provider metadata and approval remain preserved. Canonical publication and VSB original graph qualification remain separate actions.',
]


def packages():
    pending = ['numpy', 'scipy', 'icontract', 'threadpoolctl']
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
    paths={p:ROOT/p for p in ('tests/test_lfilter_provider_reuse.py',
        'scripts/review_lfilter_provider_semantics.py','scripts/review_vsb_lfilter_reuse.py',
        'scripts/stage_lfilter_canonical_records.py','scripts/audit_lfilter_provider_intake.py',
        'sciona/vsb_baseline.py')}
    paths['sciona-atoms/src/sciona/atoms/scipy/signal.py']=ROOT.parent/'sciona-atoms/src/sciona/atoms/scipy/signal.py'
    before={k:sha(p) for k,p in paths.items()};installed=packages();comparison=reuse()
    environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1',MPLCONFIGDIR='/private/tmp/sciona-mpl-cache',PYTHONPATH=str(ROOT))
    with tempfile.TemporaryDirectory(prefix='lfilter-synthetic-') as directory:
        xml=Path(directory)/'tests.xml'
        run=subprocess.run([str(ROOT/'.venv/bin/python'),'-m','pytest','-q','tests/test_lfilter_provider_reuse.py','--junitxml='+str(xml)],
            cwd=ROOT,env=environment,capture_output=True,text=True)
        if run.returncode:raise ValueError('Independent recurrence tests failed: '+run.stdout[-3000:]+run.stderr[-1000:])
        suites=list(ET.parse(xml).getroot().iter('testsuite'))
        counts={k:sum(int(s.get(k,'0')) for s in suites) for k in ('tests','failures','errors','skipped')}
    if counts!=dict(tests=40,failures=0,errors=0,skipped=0):raise ValueError('Complete 40-test suite required')
    if before!={k:sha(p) for k,p in paths.items()} or installed!=packages():raise ValueError('Qualified software changed')
    if intake!=reviewed_plan():raise ValueError('Qualified intake changed')
    record=intake['records'][0]
    provider=dict(artifact_id=record['artifact']['artifact_id'],fqdn=record['artifact']['fqdn'],
        version_id=record['version']['version_id'],content_hash=record['version']['content_hash'],interfaces=record['interfaces'],
        operation='Apply a real linear digital filter with explicit numerator, denominator, axis and optional initial delay state.',
        plain_description='Filter a numeric sequence using supplied coefficients and, optionally, a starting delay state. Return the filtered sequence.',
        validity_bounds=' '.join(p['constraints'] for p in record['interfaces']),
        reuse='Domain-independent linear sequence filtering; no power-line, measurement-group, peak or classification semantics.',
        registered_provider_dependencies=[])
    return dict(format='lfilter-provider-semantics.v1',passed=True,approved=False,proposed_tier=3,
        verdict='acceptable_with_limits',catalog_mutations=0,synthetic_only=True,providers=[provider],tests=counts,
        baseline_reuse_cases=comparison['synthetic_cases'],software_sha256=before,installed_packages=installed,
        intake_plan_sha256=sha(ROOT/'docs/reviews/lfilter_canonical_intake_plan.json'),
        resource_policy=dict(serial_tests=True,native_threads=1),limitations=LIMITS)


if __name__=='__main__':
    result=review()
    (ROOT/'docs/reviews/lfilter_provider_semantics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,tests=result['tests'],baseline_cases=len(result['baseline_reuse_cases']),approved=False)))
