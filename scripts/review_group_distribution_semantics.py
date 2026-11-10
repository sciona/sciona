"""Qualify explicit whole-group distribution scoring for automated Tier 3 publication."""
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
from sciona.atoms.ml import group_distribution_matching as atoms
from sciona.ghost import registry, abstract, dimensions
from sciona.atoms.ml.constrained_ml.decorrelation import atoms as ks_atoms, witnesses as ks_witnesses
from scripts.plan_group_distribution_intake import check_dependency
from scripts.import_group_distribution_drafts import reviewed_plan, ROOT, PLAN, sha
from scripts.import_flavours_provider_drafts import connect

LIMITS = [
    'Automated Tier 3 only; no human Tier 1 certification or Tier 2 usage qualification.',
    'Caller supplies aligned features and coordinates, exact group labels, and explicit candidate masks.',
    'Equal row weights preserve group size effects. No p-values, feature screening, subset generation, averaging or selection policy.',
    'Marginal agreement does not establish joint agreement, predictive accuracy, or valid causal/temporal validation.',
    'Query-driven reference selection is transductive and requires the reference population at decision time.',
    'Synthetic continuous and discrete examples qualify reusable scoring, not the complete historical competition workflow.',
    'Installed RECORD hashes and active constraints are checked; independent package provenance and other platforms are not certified.',
]




def packages():
    pending = ['numpy', 'pydantic', 'scipy', 'icontract']
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
    intake = reviewed_plan()
    provider_repo = ROOT.parent/'sciona-atoms-ml'
    tests = [provider_repo/'tests/test_group_distribution_matching_provider.py']
    paths = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else 'provider/'+p.name: p for p in tests}
    for relative in ('scripts/review_group_distribution_semantics.py',
                     'scripts/import_group_distribution_drafts.py', 'scripts/plan_group_distribution_intake.py',
                     'scripts/validate_lanl_group_distribution_matching.py'):
        paths[relative] = ROOT/relative
    for module in (atoms, ks_atoms, ks_witnesses, registry, abstract, dimensions):
        paths[module.__name__] = Path(inspect.getfile(module))
    before = {key: sha(path) for key, path in paths.items()}
    installed = packages()
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MPLCONFIGDIR='/private/tmp/sciona-mpl-cache', PYTHONPATH=str(ROOT))
    with tempfile.TemporaryDirectory(prefix='group-distribution-review-') as directory:
        xml = Path(directory)/'tests.xml'
        result = subprocess.run([str(ROOT/'.venv/bin/python'), '-m', 'pytest', '-q', *map(str, tests), '--junitxml='+str(xml)],
                                cwd=ROOT, env=environment, capture_output=True, text=True)
        if result.returncode:
            raise ValueError('Provider qualification tests failed: '+result.stdout[-3000:]+result.stderr[-1000:])
        suites = list(ET.parse(xml).getroot().iter('testsuite'))
        counts = {key: sum(int(s.get(key, '0')) for s in suites) for key in ('tests', 'failures', 'errors', 'skipped')}
    if counts != dict(tests=11, failures=0, errors=0, skipped=0):
        raise ValueError('All 11 focused qualification tests required')
    records = []
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        for dependency in intake['registered_provider_dependencies']:
            check_dependency(db, dependency)
        for item in intake['new_providers']:
            for prefix, key in [('atom', 'atom_id'), ('artifact', 'artifact_id')]:
                actual = db.execute(f'''SELECT a.fqdn,v.version_id::text,v.content_hash,v.trust_tier
                    FROM {prefix}s a JOIN {prefix}_versions v USING({key}) WHERE a.{key}=%s AND v.is_latest''',
                    (item['atom_id'],)).fetchall()
                if actual != [dict(fqdn=item['runtime_fqdn'], version_id=item['version_id'], content_hash=item['content_hash'], trust_tier=3)]:
                    raise ValueError('Exact current provider version required')
                ports = db.execute(f'''SELECT direction,ordinal,name,type_desc,required,default_value_repr,constraints
                    FROM {prefix}_io_specs WHERE version_id=%s ORDER BY direction,ordinal''', (item['version_id'],)).fetchall()
                if ports != sorted(item['interfaces'], key=lambda p: (p['direction'], p['ordinal'])):
                    raise ValueError('Exact stored provider interfaces required')
            dependencies = db.execute("""SELECT dependency_artifact_fqdn,dependency_content_hash,
                dependency_role,optional,port_name,binding_metadata FROM artifact_dependencies
                WHERE dependent_version_id=%s ORDER BY dependency_artifact_fqdn""", (item['version_id'],)).fetchall()
            expected = [dict(dependency_artifact_fqdn=d['fqdn'], dependency_content_hash=d['content_hash'],
                dependency_role='logic_atom', optional=False, port_name='',
                binding_metadata=dict(source_version_id=d['version_id'], scope='Mandatory runtime KS calculation with equal row weights.'))
                for d in intake['registered_provider_dependencies']]
            if dependencies != expected:
                raise ValueError('Exact mandatory runtime dependency binding required')
            records.append(dict(artifact_id=item['atom_id'], fqdn=item['runtime_fqdn'], version_id=item['version_id'],
                content_hash=item['content_hash'], interfaces=sorted(item['interfaces'], key=lambda p:(p['direction'],p['ordinal'])),
                operation='Compute per-feature two-sided empirical KS distances for each explicitly supplied whole-group subset against a shared reference.',
                plain_description='Compare each chosen collection of groups with a reference population, one feature at a time.',
                validity_bounds=' '.join(p['constraints'] for p in item['interfaces']),
                reuse='Reusable for continuous measurement batches and discrete production counts, with explicit feature alignment and group membership. Source-domain screening and model-selection policies remain separate.',
                registered_provider_dependencies=intake['registered_provider_dependencies']))
    if before != {key: sha(path) for key,path in paths.items()} or installed != packages() or intake != reviewed_plan():
        raise ValueError('Software or qualification inputs changed during review')
    return dict(format='group-distribution-semantics.v1', passed=True, approved=False, proposed_tier=3,
        verdict='acceptable_with_limits', catalog_mutations=0, synthetic_only=True, providers=records,
        tests=counts, software_sha256=before, installed_packages=installed,
        intake_plan_sha256=sha(PLAN), connected_execution_commit=intake['execution_evidence_commit'],
        connected_execution_sha256=intake['execution_evidence_sha256'],
        resource_policy=dict(native_threads=1, test_workers=1), limitations=LIMITS)


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/group_distribution_semantics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, providers=len(result['providers']), tests=result['tests'],
                         packages=len(result['installed_packages']), approved=False)))
