"""Verify approved resource_route providers independently of original CDG status."""
import json
import subprocess
from uuid import UUID

from scripts.import_flavours_provider_drafts import ROOT, connect
from scripts.plan_flavours_provider_intake import sha
from scripts.promote_openvaccine_decomposed_providers import snapshot, PORT_FIELDS
from scripts.review_video_numeric_runtime import review as runtime_review

PLAN = ROOT/'docs/reviews/resource_route_approved_provider_state.json'
EVIDENCE = {
    'docs/reviews/resource_route_current_runtime.json': '51723116',
    'docs/reviews/resource_route_provider_semantics.json': '51723116',
    'docs/reviews/resource_route_provider_metadata_plan.json': '046aa252',
}


def sources():
    evidence = {}
    for path, commit in EVIDENCE.items():
        raw = (ROOT/path).read_bytes()
        if raw != subprocess.check_output(['git', 'show', commit+':'+path], cwd=ROOT):
            raise ValueError('Immutable provider qualification changed')
        evidence[path] = json.loads(raw)
    graph = evidence['docs/reviews/resource_route_current_runtime.json']
    semantics = evidence['docs/reviews/resource_route_provider_semantics.json']
    for path, digest in graph['software_sha256'].items():
        if sha(ROOT/path) != digest:
            raise ValueError('Qualified provider source changed: '+path)
    if sha(ROOT/'scripts/review_resource_route_provider_semantics.py') != semantics['reviewer_sha256']:
        raise ValueError('Qualified semantic reviewer changed')
    if runtime_review() != graph['runtime']:
        raise ValueError('Qualified numerical runtime changed')
    return evidence


def check(db, plan):
    for record in plan['records']:
        identity = record['artifact_id']
        module, symbol = record['runtime_fqdn'].rsplit('.', 1)
        if db.execute('SELECT import_module,source_symbol FROM atoms WHERE atom_id=%s', (identity,)).fetchone() != dict(import_module=module, source_symbol=symbol):
            raise ValueError('Provider runtime binding differs')
        for prefix, key in [('atom', 'atom_id'), ('artifact', 'artifact_id')]:
            rows = db.execute(f'''SELECT a.fqdn,a.status,a.is_publishable,v.version_id::text,v.content_hash,v.trust_tier
                FROM {prefix}s a JOIN {prefix}_versions v USING({key}) WHERE a.{key}=%s AND v.is_latest''', (identity,)).fetchall()
            expected = dict(fqdn=record['catalog_fqdn'], status='approved', is_publishable=True,
                version_id=record['version_id'], content_hash=record['content_hash'], trust_tier=3)
            if rows != [expected]:
                raise ValueError('Exact approved provider version differs')
            ports = db.execute('SELECT '+','.join(PORT_FIELDS)+f' FROM {prefix}_io_specs WHERE version_id=%s ORDER BY direction,ordinal', (record['version_id'],)).fetchall()
            if ports != record['ports' if prefix == 'artifact' else 'legacy_ports']:
                raise ValueError('Exact provider interface differs')
            if not db.execute(f'SELECT {prefix}_is_publishable(%s) AS ready', (identity,)).fetchone()['ready']:
                raise ValueError('Provider native eligibility failed')
            if not db.execute(f'SELECT 1 FROM catalog_{prefix}s_served WHERE {key}=%s', (identity,)).fetchone():
                raise ValueError('Provider not served')
        if db.execute('SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND NOT passed', (record['version_id'],)).fetchone():
            raise ValueError('Failed provider evidence')
    if snapshot(db, [UUID(r['artifact_id']) for r in plan['records']]) != plan['catalog_snapshot']:
        raise ValueError('Frozen approved provider metadata changed')
    if dependencies(db, plan['records']) != plan['dependencies']:
        raise ValueError('Frozen provider dependency closure changed')


def dependencies(db, records):
    return db.execute('''SELECT dependent_version_id::text,dependency_artifact_fqdn,dependency_content_hash,
        port_name,dependency_role,optional,binding_metadata FROM artifact_dependencies
        WHERE dependent_version_id=ANY(%s) ORDER BY dependent_version_id,dependency_artifact_fqdn,port_name''',
        ([UUID(r['version_id']) for r in records],)).fetchall()


def qualify_plan():
    plan = json.loads(PLAN.read_text())
    sources()
    if plan['checker_sha256'] != sha(__file__) or plan['evidence_sha256'] != {p: sha(ROOT/p) for p in EVIDENCE}:
        raise ValueError('Approved provider verification changed')
    return plan


def contracts():
    return {r['runtime_fqdn']: dict(r, ports=r['usage_ports']) for r in qualify_plan()['records']}


def freeze():
    evidence = sources()
    from scripts import complete_resource_route_provider_metadata as repair
    publication = evidence['docs/reviews/resource_route_provider_metadata_plan.json']
    if repair.qualified_plan() != publication or repair.apply()['rows_created'] != 0:
        raise ValueError('Exact additive metadata must already be present')
    records = []
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        repair.check(db,publication['records'],require_eligible=True)
        for binding in publication['records']:
            record = {k:binding[k] for k in ('runtime_fqdn','artifact_id','version_id','content_hash','catalog_fqdn')}
            record.update(ports=binding['artifact_ports'],legacy_ports=binding['atom_ports'],usage_ports=binding['ports'])
            records.append(record)
        if len(records)!=5:raise ValueError('Five route providers required')
        result = dict(passed=True, catalog_mutations=0, records=sorted(records, key=lambda r: r['runtime_fqdn']),
            checker_sha256=sha(__file__), evidence_sha256={p: sha(ROOT/p) for p in EVIDENCE},
            catalog_snapshot=snapshot(db, [UUID(r['artifact_id']) for r in records]), dependencies=dependencies(db, records))
        check(db, result)
    return result


if __name__ == '__main__':
    result = freeze()
    PLAN.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(approved_providers=len(result['records']), catalog_mutations=0)))
