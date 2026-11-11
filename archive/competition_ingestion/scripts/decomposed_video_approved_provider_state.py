"""Freeze exact approved video bindings independently of original CDG state."""
import json
from uuid import UUID

from scripts.audit_decomposed_video_multilabel_readiness import ROOT, sha
from scripts.import_flavours_provider_drafts import connect
from scripts.promote_video_sharded_recipe import qualify_plan as adapter_plan, check as check_adapter
from scripts.promote_video_sharded_recipe import check_dependencies, PORT_FIELDS, snapshot

PLAN = ROOT/'docs/reviews/video_approved_provider_state.json'


def check(db, plan):
    published = adapter_plan()
    if sha(ROOT/'docs/reviews/video_sharded_recipe_publication_plan.json') != plan['adapter_plan_sha256']:
        raise ValueError('Qualified adapter publication changed')
    check_dependencies(db, published, published=True)
    for record in plan['records']:
        for prefix, key in [('atom', 'atom_id'), ('artifact', 'artifact_id')]:
            actual = db.execute(f'''SELECT a.fqdn,a.status,a.is_publishable,v.version_id::text,v.content_hash,v.trust_tier
                FROM {prefix}s a JOIN {prefix}_versions v USING({key}) WHERE a.{key}=%s AND v.is_latest''',
                (record['artifact_id'],)).fetchall()
            expected = dict(fqdn=record['catalog_fqdn'], status='approved', is_publishable=True,
                version_id=record['version_id'], content_hash=record['content_hash'], trust_tier=3)
            if actual != [expected]:
                raise ValueError('Exact approved provider version differs')
            ports = db.execute('SELECT '+','.join(PORT_FIELDS)+f' FROM {prefix}_io_specs WHERE version_id=%s ORDER BY direction,ordinal',
                (record['version_id'],)).fetchall()
            if ports != record['ports' if prefix == 'artifact' else 'legacy_ports']:
                raise ValueError('Qualified provider interface differs')
            if not db.execute(f'SELECT {prefix}_is_publishable(%s) AS eligible', (record['artifact_id'],)).fetchone()['eligible']:
                raise ValueError('Provider not natively eligible: '+prefix+' '+record['catalog_fqdn'])
            if not db.execute(f'SELECT 1 FROM catalog_{prefix}s_served WHERE {key}=%s', (record['artifact_id'],)).fetchone():
                raise ValueError('Provider not served')
        if db.execute('SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND NOT passed', (record['version_id'],)).fetchone():
            raise ValueError('Failed provider audit')
    ids = [UUID(r['artifact_id']) for r in plan['records']]
    if 'catalog_snapshot' in plan and snapshot(db, ids) != plan['catalog_snapshot']:
        raise ValueError('Frozen provider metadata changed')
    dependencies = db.execute('''SELECT dependent_version_id::text,dependency_artifact_fqdn,dependency_content_hash,
        port_name,dependency_role,optional,binding_metadata FROM artifact_dependencies
        WHERE dependent_version_id=ANY(%s) ORDER BY dependent_version_id,dependency_artifact_fqdn,port_name''',
        ([UUID(r['version_id']) for r in plan['records']],)).fetchall()
    if dependencies != plan['dependencies']:
        raise ValueError('Frozen provider dependency closure changed')


def qualify_plan():
    plan = json.loads(PLAN.read_text())
    if plan['checker_sha256'] != sha(__file__):
        raise ValueError('Provider checker changed')
    # Recheck every provider source and installed package against publication evidence.
    adapter_plan()
    from sciona.catalog_video_multilabel_graph import build_catalog_video_multilabel_graph
    if {n.matched_primitive for n in build_catalog_video_multilabel_graph().nodes} != {r['runtime_fqdn'] for r in plan['records']}:
        raise ValueError('Graph provider inventory changed')
    return plan


def contracts():
    return {r['runtime_fqdn']:r for r in qualify_plan()['records']}


def freeze():
    source = json.loads((ROOT/'docs/reviews/video_graph_contracts.json').read_text())
    records = []
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        check_adapter(db, adapter_plan(), published=True)
        for binding in source['records']:
            record = dict(runtime_fqdn=binding['runtime_fqdn'], catalog_fqdn=binding['fqdn'],
                artifact_id=binding['artifact_id'], version_id=binding['version_id'], content_hash=binding['content_hash'],
                ports=binding['ports'], legacy_ports=binding['legacy_ports'])
            records.append(record)
        if len(records) != 10:raise ValueError('All ten provider bindings required')
        result = dict(passed=True, records=sorted(records, key=lambda r:r['runtime_fqdn']), checker_sha256=sha(__file__),
            adapter_plan_sha256=sha(ROOT/'docs/reviews/video_sharded_recipe_publication_plan.json'))
        result['dependencies'] = db.execute('''SELECT dependent_version_id::text,dependency_artifact_fqdn,dependency_content_hash,
            port_name,dependency_role,optional,binding_metadata FROM artifact_dependencies
            WHERE dependent_version_id=ANY(%s) ORDER BY dependent_version_id,dependency_artifact_fqdn,port_name''',
            ([UUID(r['version_id']) for r in records],)).fetchall()
        result['catalog_snapshot'] = snapshot(db, [UUID(r['artifact_id']) for r in records])
        check(db, result)
    return result


if __name__ == '__main__':
    result = freeze()
    PLAN.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(approved_providers=len(result['records']), catalog_mutations=0)))
