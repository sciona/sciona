"""Freeze and verify exact approved providers independently of original CDG state."""
import json
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5
from sciona.decomposed_m5u_graph import build_decomposed_m5u_graph
from scripts.review_decomposed_m5u_implementation import ROOT, review, sha, require
from scripts.promote_m5u_decomposed_providers import snapshot, PORT_FIELDS
from scripts.import_flavours_provider_drafts import connect

PLAN = ROOT/'docs/reviews/m5u_approved_provider_state.json'


def records():
    qualified = review()
    identities = {
        'sciona.atoms.ml.m5u_execution.m5u_prepare': '33489a4f-b0df-5dc4-9822-d7ac86deb049',
        'sciona.atoms.numerical.statistics.ordered_sum.ordered_weighted_sum': '4f486f1c-82db-50db-b5d2-188aeaa8cc4a',
    }
    result = []
    for record in qualified['providers']:
        identity = identities.get(record['runtime_fqdn']) or str(uuid5(NAMESPACE_URL, 'sciona-provider-draft:'+record['catalog_fqdn']))
        ports = [dict(p) for p in record['interfaces']]
        if record['classification'] == 'generic_operation':
            # Catalog names remain domain independent. The graph's output
            # alias is applied by ordinal when constructing its binding.
            require(record['runtime_fqdn'] == 'sciona.atoms.numerical.statistics.ordered_sum.ordered_weighted_sum',
                    'Unexpected generic provider')
            for port in ports:
                require(port['type_desc'] == 'ndarray', 'Unexpected graph array type')
                port['type_desc'] = 'numpy.ndarray'
                if port['direction'] == 'output':
                    require(port['name'] == 'total' and port['ordinal'] == 0, 'Unexpected graph output alias')
                    port['name'] = 'scores'
        result.append(dict(record, artifact_id=identity, ports=ports))
    require(len(result) == 6 and len({r['artifact_id'] for r in result}) == 6,
            'Six distinct qualified providers required')
    return result


def check(db, plan):
    ids = [UUID(r['artifact_id']) for r in plan['records']]
    for r in plan['records']:
        for prefix, key in [('atom','atom_id'), ('artifact','artifact_id')]:
            rows = db.execute(f'''SELECT a.fqdn,a.status,a.is_publishable,v.version_id::text,v.content_hash,v.trust_tier
                FROM {prefix}s a JOIN {prefix}_versions v USING({key}) WHERE a.{key}=%s AND v.is_latest''', (UUID(r['artifact_id']),)).fetchall()
            require(rows == [dict(fqdn=r['catalog_fqdn'],status='approved',is_publishable=True,
                version_id=r['version_id'],content_hash=r['content_hash'],trust_tier=3)], 'Exact approved provider version differs')
            ports = db.execute('SELECT '+','.join(PORT_FIELDS)+f' FROM {prefix}_io_specs WHERE version_id=%s ORDER BY direction,ordinal', (UUID(r['version_id']),)).fetchall()
            require(ports == r['ports'], 'Qualified provider interface differs')
            require(db.execute(f'SELECT {prefix}_is_publishable(%s) AS yes', (UUID(r['artifact_id']),)).fetchone()['yes'], 'Provider not natively eligible')
        require(not db.execute('SELECT 1 FROM artifact_dependencies WHERE dependent_version_id=%s', (UUID(r['version_id']),)).fetchone(), 'Unexpected internal provider dependency')
        require(not db.execute('SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND NOT passed', (UUID(r['version_id']),)).fetchone(), 'Failed provider audit')
    if 'catalog_snapshot' in plan:
        require(snapshot(db, ids) == plan['catalog_snapshot'], 'Approved provider metadata or history changed')
    for table, key in [('catalog_atoms_served','atom_id'),('catalog_artifacts_served','artifact_id')]:
        require(db.execute(f'SELECT count(DISTINCT {key}) AS n FROM {table} WHERE {key}=ANY(%s)', (ids,)).fetchone()['n'] == 6, 'Not all providers served')


def qualify_plan():
    frozen = json.loads(PLAN.read_text())
    require(frozen['records'] == records() and frozen['checker_sha256'] == sha(Path(__file__)), 'Approved provider qualification drift')
    return frozen


def contracts():
    return {r['runtime_fqdn']:r for r in qualify_plan()['records']}


if __name__ == '__main__':
    from scripts import promote_m5u_decomposed_providers as publication
    published = publication.qualify_plan()
    result = dict(passed=True, records=records(), checker_sha256=sha(Path(__file__)))
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        # Verify publication now, then freeze only providers so later original-CDG
        # staging cannot invalidate the approved provider snapshot.
        publication.check(db, published, published=True)
        from scripts.import_residual_execution_drafts import comparable
        new_ids = [UUID(r['artifact_id']) for r in published['records']]
        require(publication.preserved_history(db, new_ids) == published['preserved_approved'],
                'Published provider history changed')
        for item in published['inserts']:
            actual = db.execute('SELECT '+','.join(item['row'])+' FROM '+item['table']+' WHERE '+
                ' AND '.join(k+'=%s' for k in item['keys']), tuple(item['keys'].values())).fetchall()
            require(comparable(actual) == [item['row']], 'Published provider metadata changed')
        for item in published['rollups']:
            actual = db.execute('SELECT '+','.join(item['fields'])+' FROM '+item['table']+
                ' WHERE '+item['key']+'=%s', (item['identity'],)).fetchone()
            require(actual == item['fields'], 'Published provider review changed')
        from scripts.repair_decomposed_m5u_metadata import RUNNER as repair_runner, IDENTITY, CONTRACT, PLAIN
        for prefix, key in [('atom','atom_id'), ('artifact','artifact_id')]:
            descriptions = db.execute(f'SELECT kind,content FROM {prefix}_descriptions WHERE {key}=%s AND language=%s ORDER BY kind',
                                      (IDENTITY, 'en')).fetchall()
            require(descriptions == [dict(kind='dejargonized',content=PLAIN),dict(kind='technical',content=CONTRACT)],
                    'Preparation descriptions differ')
            parameters = db.execute(f'SELECT name,type_desc,required,constraints_json FROM {prefix}_parameters WHERE {key}=%s',
                                    (IDENTITY,)).fetchall()
            require(parameters == [dict(name='payload',type_desc='dict',required=True,
                constraints_json=dict(parameter_contract=CONTRACT,validity_bounds=CONTRACT,
                                      review_status='automated_tier_3_pass_with_limits'))], 'Preparation parameter contract differs')
        require(db.execute('SELECT count(*) AS n FROM artifact_audit_evidence WHERE artifact_id=%s AND runner_version=%s AND passed',
                           (IDENTITY, repair_runner)).fetchone()['n'] == 1, 'Preparation repair evidence missing')
        check(db, result)
        result['catalog_snapshot'] = snapshot(db, [UUID(r['artifact_id']) for r in result['records']])
    PLAN.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(approved_providers=6, catalog_mutations=0)))
