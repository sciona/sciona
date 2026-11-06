"""Qualify reusable ordered resource selection for automated Tier 3 use."""
import json
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from scripts.stage_trackml_commit_providers import reviewed_plan, ROOT, sha, connect
from scripts.review_trackml_commit_reuse import review as reuse
from scripts.review_trackml_geometry_provider_semantics import packages

LIMITS = [
    'Automated Tier3 qualification only; no human Tier1 or usage-qualified Tier2 designation.',
    'Caller supplies an int64 matrix of nonnegative resource IDs and a nonempty one-dimensional boolean used mask. Every ID must index the mask. Zero is an empty-slot sentinel and claimed[0] is always true.',
    'Counts must be integers with min_nhits >= 1, max_nloss >= 0 and max_nrows >= 0. The loss fraction must be finite in [0,1]. Existing preconditions do not enforce the full domain; caller boundaries must do so.',
    'Rows are processed in caller-supplied order. Greedy selection is order-dependent and is not a globally optimal allocation. Resource IDs carry no particle-specific meaning.',
    'Duplicate IDs within one row count as repeated positions, not unique resources. Qualification does not promise within-row deduplication.',
    'The minimum retained count and maximum loss count/fraction are inclusive. The accepted-row budget is strict: after max_nrows acceptances no more rows are accepted. Zero budget accepts no rows.',
    'Provider copies both inputs and returns selected rows and the updated claimed mask. Rejected rows claim no resources. Reserving skipped rows and in-place mutation are separate recipe policies.',
    'Empty matrices and all-zero rows are accepted by the provider. Pinned source asserts on all-zero rows, so source parity only applies to nonempty rows.',
    'Provider defaults max_nloss=100 and max_nrows=100000 differ from source None, which resolves to matrix width and row count. Translate source defaults explicitly.',
    'Symbolic metadata describes only loss_fraction=nloss/nhits, not the ordered selection algorithm. Its positive max_nrows bound excludes zero even though concrete execution supports zero. Witnesses report output shapes and dtypes only.',
    'Candidate ordering, identifier assignment, submission formatting and fill policy remain caller operations. No historical competition accuracy or global optimization guarantee is claimed.',
    'Installed RECORD validation checks current package files, not publisher authentication or clean deployment on another platform.',
    'Provider publication and original TrackML CDG qualification remain separate actions.',
]


def staged(intake):
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        for record in intake['providers']:
            identity, version = record['atom_id'], record['version_id']
            for prefix, key, status in [('atom', 'atom_id', 'approved' if record['existing_legacy'] else 'flagged'), ('artifact', 'artifact_id', 'draft')]:
                row = db.execute(f'''SELECT a.fqdn,a.status,a.is_publishable,v.content_hash,v.trust_tier,v.is_latest
                    FROM {prefix}s a JOIN {prefix}_versions v USING({key}) WHERE a.{key}=%s AND v.version_id=%s''',
                    (identity, version)).fetchone()
                if row != dict(fqdn=record['catalog_fqdn'], status=status, is_publishable=False,
                               content_hash=record['content_hash'], trust_tier=3, is_latest=True):
                    raise ValueError('Exact staged provider required')
                ports = db.execute(f'''SELECT direction,name,ordinal,type_desc,required,default_value_repr,constraints
                    FROM {prefix}_io_specs WHERE version_id=%s ORDER BY direction,ordinal''', (version,)).fetchall()
                if ports != ([] if prefix=='atom' and record['existing_legacy'] else record['interfaces']):
                    raise ValueError('Exact staged interface required')
            if db.execute('SELECT 1 FROM artifact_dependencies WHERE dependent_version_id=%s', (version,)).fetchone():
                raise ValueError('Unexpected registered provider dependency')
            if db.execute("SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND passed=false AND status IN ('failed','completed')", (version,)).fetchone():
                raise ValueError('Completed failed audit exists')


def review():
    intake = reviewed_plan()
    staged(intake)
    paths = {p: ROOT/p for p in [
        'tests/test_commit_provider_semantics.py', 'tests/test_decomposed_trackml_commit.py',
        'scripts/review_trackml_commit_provider_semantics.py', 'scripts/review_trackml_commit_reuse.py',
        'scripts/review_trackml_geometry_provider_semantics.py', 'scripts/stage_trackml_commit_providers.py',
        'sciona/trackml_candidates.py', 'sciona/decomposed_trackml_commit_lifecycle.py', 'sciona/ghost/registry.py']}
    for relative in ['particle_tracking/track_matching/atoms.py',
                     'particle_tracking/track_matching/expressions.py', 'particle_tracking/track_matching/witnesses.py',
                     'particle_tracking/source_tracking/solution_LICENSE.source']:
        key = 'sciona-atoms-physics/src/sciona/atoms/'+relative
        paths[key] = ROOT.parent/key
    before = {k: sha(p) for k, p in paths.items()}
    installed = packages()
    comparison = reuse()
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MPLCONFIGDIR='/private/tmp/sciona-mpl-cache', PYTHONPATH=str(ROOT))
    with tempfile.TemporaryDirectory(prefix='commit-synthetic-') as directory:
        xml = Path(directory)/'tests.xml'
        run = subprocess.run([str(ROOT/'.venv/bin/python'), '-m', 'pytest', '-q',
            'tests/test_commit_provider_semantics.py', 'tests/test_decomposed_trackml_commit.py',
            '--junitxml='+str(xml)], cwd=ROOT, env=environment, capture_output=True, text=True)
        if run.returncode:
            raise ValueError('Decision contract tests failed: '+run.stdout[-3000:]+run.stderr[-1000:])
        suites = list(ET.parse(xml).getroot().iter('testsuite'))
        counts = {k: sum(int(s.get(k, '0')) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
    if counts != dict(tests=36, failures=0, errors=0, skipped=0):
        raise ValueError('Complete 36-test qualification required')
    if before != {k: sha(p) for k, p in paths.items()} or installed != packages() or intake != reviewed_plan():
        raise ValueError('Qualified software or intake changed')
    staged(intake)
    providers = []
    for record in intake['providers']:
        providers.append(dict(artifact_id=record['atom_id'], fqdn=record['catalog_fqdn'],
            runtime_fqdn=record['runtime_fqdn'], version_id=record['version_id'], content_hash=record['content_hash'],
            interfaces=record['interfaces'],
            operation='Select ordered rows of resource IDs subject to a minimum retained position count, maximum already-claimed count and fraction, and accepted-row budget; return copied selections and updated claims.',
            plain_description='Choose rows in their supplied order while removing resources already taken by earlier choices, subject to supplied count and loss limits.',
            validity_bounds=' '.join(dict.fromkeys(p['constraints'] for p in record['interfaces'])),
            reuse='Pure ordered resource allocation over integer IDs and explicit count/fraction controls; no event, detector, row-ranking or output-format policy.',
            registered_provider_dependencies=[]))
    return dict(format='trackml-commit-provider-semantics.v1', passed=True, approved=False, proposed_tier=3,
        verdict='acceptable_with_limits', catalog_mutations=0, synthetic_only=True, providers=providers,
        tests=counts, exact_source_cases=dict(batches=comparison['exact_source_batches']),
        source_comparison_sha256=comparison['implementation_sha256'],
        source_differences=dict(reservation=comparison['reservation_counterexample'], empty_rows=comparison['empty_row_difference']),
        software_sha256=before, installed_packages=installed,
        intake_plan_sha256=sha(ROOT/'docs/reviews/trackml_commit_provider_intake_plan.json'),
        resource_policy=dict(serial_tests=True, native_threads=1), limitations=LIMITS)


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/trackml_commit_provider_semantics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, tests=result['tests'], providers=len(result['providers']), approved=False)))
