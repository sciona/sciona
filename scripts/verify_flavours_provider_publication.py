"""Fresh-connection verification of provider serving and historical preservation."""
import json
from uuid import UUID,uuid5
from scripts import promote_flavours_providers as publisher
from scripts.inventory_flavours_decomposed_providers import inventory
from scripts.audit_flavours_original_intake_scope import audit
from scripts.plan_flavours_provider_intake import sha


def verify():
    plan=publisher.qualify_plan()
    identities=[UUID(r['artifact_id']) for r in plan['records']]
    with publisher.connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        publisher.check(db,plan,published=True)
        after=publisher.snapshot(db,identities)
        for table in ('atom_versions','artifact_versions','atom_io_specs','artifact_io_specs'):
            if after[table]!=plan['before'][table]:raise ValueError('Version history or interfaces changed')
        preserved=[UUID(r['artifact_id']) for r in plan['records'] if r['existing_approved']]
        if publisher.snapshot(db,preserved)!=plan['preserved_approved']:raise ValueError('Existing approved provider history changed')
        native=0
        for record in plan['records']:
            identity,version=UUID(record['artifact_id']),UUID(record['version_id'])
            for table,key in [('catalog_atoms_served','atom_id'),('catalog_artifacts_served','artifact_id')]:
                if not db.execute(f'SELECT 1 FROM {table} WHERE {key}=%s AND fqdn=%s',(identity,record['catalog_fqdn'])).fetchone():
                    raise ValueError('Exact provider not served')
            if record['existing_approved']:continue
            eligible=db.execute('SELECT atom_is_publishable(%s) AND artifact_is_publishable(%s) AS eligible',(identity,identity)).fetchone()['eligible']
            if not eligible:raise ValueError('Native eligibility failed')
            native+=1
            details=db.execute('SELECT details FROM artifact_audit_evidence WHERE evidence_id=%s',(uuid5(version,publisher.RUNNER),)).fetchone()['details']
            previous=[r for r in plan['rollups'] if r['identity']==str(identity) and r['before']]
            if details['provider']!=record or details['previous_rollups']!=previous or details['limitations']!=plan['qualification']['limitations']:
                raise ValueError('Exact review and previous rollups not retained')
    current=inventory()
    if current['both_catalogs_exact_approved']!=39:raise ValueError('Not all exact providers approved')
    original=audit()
    retained=json.loads((publisher.ROOT/'docs/reviews/flavours_original_intake_scope.json').read_text())
    if original['original_history_sha256']!=retained['original_history_sha256']:raise ValueError('Original CDG history changed')
    return dict(passed=True,read_only=True,catalog_mutations=0,trust_tier=3,
        exact_providers_approved_and_served=39,new_canonical_approvals=37,native_eligibility_verified=native,
        existing_approved_providers_unchanged=2,all_provider_versions_and_ports_preserved=True,
        previous_rollups_retained_in_versioned_evidence=True,original_cdg_history_unchanged=True,
        original_cdg_approved=False,plan_sha256=sha(publisher.PLAN),verifier_sha256=sha(__file__),
        next_step='Stage and execute the complete decomposed graph under the original CDG identity before its promotion.')


if __name__=='__main__':
    result=verify()
    (publisher.ROOT/'docs/reviews/flavours_provider_publication_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
