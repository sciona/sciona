"""Qualify existing helix direction for canonical automated Tier 3 use."""
import json
from scripts.stage_trackml_direction_canonical_records import reviewed_plan, ROOT, sha, legacy_digest
from scripts.review_trackml_direction_reuse import review as reuse
from scripts.review_trackml_geometry_provider_semantics import packages
from scripts.import_flavours_provider_drafts import connect
from uuid import UUID

LIMITS = [
    'Automated Tier3 qualification only; no human Tier1 or usage-qualified Tier2 designation.',
    'Nonempty aligned finite float64 vectors in consistent caller-defined length units. No general broadcasting contract.',
    'Output is axial difference except at equal heights, where it is pitch times transverse rotation sign. Zero is possible and callers must handle it.',
    'The existing provider does not independently enforce every shape, dtype or finite-intermediate precondition. Qualified callers must enforce these conditions.',
    'Finite input can overflow the fallback cross product while yielding a finite sign. Such arithmetic is outside qualification; no floating-point error bound is supplied.',
    'The registered symbolic equation describes only the ordinary axial-difference branch. The abstract witness supplies shape only; neither proves the equal-height fallback.',
    'Geometry is reusable across applications with consistent coordinates and units. No detector layout, field strength, candidate ranking or calibration map is embedded.',
    'Installed RECORD integrity validates current dependency files, not wheel publisher authentication or deployment on other platforms.',
    'Legacy approval and history remain preserved. Canonical approval, recipe integration and original-CDG publication are separate gates.',
]


def staged(intake):
    record = intake['records'][0]
    identity, version = record['artifact']['artifact_id'], record['version']['version_id']
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        if legacy_digest(db,[UUID(identity)]) != intake['legacy_sha256']:
            raise ValueError('Legacy history changed')
        row=db.execute('''SELECT a.status,a.is_publishable,v.content_hash,v.trust_tier,v.is_latest
            FROM artifacts a JOIN artifact_versions v USING(artifact_id)
            WHERE a.artifact_id=%s AND v.version_id=%s''',(identity,version)).fetchone()
        if row!=dict(status='draft',is_publishable=False,content_hash=record['version']['content_hash'],trust_tier=3,is_latest=True):
            raise ValueError('Exact staged canonical provider required')
        ports=db.execute('''SELECT direction,name,ordinal,type_desc,required,default_value_repr,constraints
            FROM artifact_io_specs WHERE version_id=%s ORDER BY direction,ordinal''',(version,)).fetchall()
        if ports!=record['interfaces']: raise ValueError('Exact staged interface required')
        if db.execute('SELECT 1 FROM artifact_dependencies WHERE dependent_version_id=%s',(version,)).fetchone():
            raise ValueError('Unexpected registered dependency')
        if db.execute("SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND passed=false AND status IN ('completed','failed')",(version,)).fetchone():
            raise ValueError('Completed failed audit exists')


def review():
    intake=reviewed_plan();staged(intake)
    paths={p:ROOT/p for p in ('scripts/review_trackml_direction_provider_semantics.py',
        'scripts/review_trackml_geometry_provider_semantics.py','scripts/review_trackml_direction_reuse.py',
        'scripts/stage_trackml_direction_canonical_records.py','scripts/audit_trackml_direction_intake.py',
        'tests/test_helix_direction_reuse.py','sciona/trackml_nearest.py','sciona/ghost/decorators.py',
        'sciona/ghost/registry.py','sciona/ghost/dimensions.py')}
    for name in ('atoms.py','expressions.py','witnesses.py','__init__.py'):
        key='sciona-atoms-physics/src/sciona/atoms/particle_tracking/helix_geometry/'+name
        paths[key]=ROOT.parent/key
    before={k:sha(p) for k,p in paths.items()}; installed=packages()
    comparison=reuse()
    if before!={k:sha(p) for k,p in paths.items()} or installed!=packages() or intake!=reviewed_plan():
        raise ValueError('Qualified software or intake changed')
    staged(intake)
    record=intake['records'][0]
    provider=dict(artifact_id=record['artifact']['artifact_id'],fqdn=record['artifact']['fqdn'],
        version_id=record['version']['version_id'],content_hash=record['version']['content_hash'],interfaces=record['interfaces'],
        operation='Determine axial motion sense from two positions, a transverse center and signed pitch, including the equal-height rotation fallback.',
        plain_description='Determine the direction of travel along a spiral from two points.',
        validity_bounds=' '.join(dict.fromkeys(p['constraints'] for p in record['interfaces'])),
        reuse='Numeric geometry with caller-defined coordinates and units; detector topology, field strength and tracking policy remain outside the provider.',
        registered_provider_dependencies=[])
    return dict(format='trackml_direction-provider-semantics.v1',passed=True,approved=False,proposed_tier=3,
        verdict='acceptable_with_limits',catalog_mutations=0,synthetic_only=True,providers=[provider],
        tests=comparison['tests'],exact_source_cases=comparison['source_parity_cases'],
        symbolic_scope='Axial-difference branch only; equal-height fallback checked concretely.',
        software_sha256=before,installed_packages=installed,
        intake_plan_sha256=sha(ROOT/'docs/reviews/trackml_direction_canonical_intake_plan.json'),
        resource_policy=dict(serial_tests=True,native_threads=1),limitations=LIMITS)


if __name__=='__main__':
    result=review()
    (ROOT/'docs/reviews/trackml_direction_provider_semantics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,tests=result['tests'],providers=1,approved=False)))
