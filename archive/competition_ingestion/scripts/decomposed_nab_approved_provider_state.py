"""Freeze and verify exact approved providers independently of original CDG state."""
import json
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5
from sciona.decomposed_nab_graph import build_decomposed_nab_graph
from scripts.review_decomposed_nab import ROOT, review, sha, require
from scripts.promote_nab_decomposed_providers import snapshot, PORT_FIELDS
from scripts.import_flavours_provider_drafts import connect

PLAN = ROOT/'docs/reviews/nab_approved_provider_state.json'


def records():
    qualified = review()
    graph = build_decomposed_nab_graph()
    result = []
    for record in qualified['providers']:
        node = next(n for n in graph.nodes if n.matched_primitive == record['runtime_fqdn'])
        ports = []
        for direction in ('input', 'output'):
            for ordinal, p in enumerate(getattr(node, direction+'s')):
                ports.append(dict(direction=direction, name=p.name, ordinal=ordinal, type_desc=p.type_desc,
                    required=p.required, default_value_repr=p.default_value_repr or '', constraints=p.constraints))
        result.append(dict(record, artifact_id=('58493a86-5102-5855-a249-33e00553bc26' if record['runtime_fqdn'].endswith('.nab_prepare') else str(uuid5(NAMESPACE_URL, 'sciona-provider-draft:'+record['catalog_fqdn']))), ports=ports))
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
    from scripts import promote_nab_decomposed_providers as publication
    published = publication.qualify_plan()
    result = dict(passed=True, records=records(), checker_sha256=sha(Path(__file__)))
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        # Preparation metadata was deliberately repaired after the five-provider
        # publication. Verify those five providers without freezing CDG state.
        from scripts.import_residual_execution_drafts import comparable
        five_ids = [UUID(r['artifact_id']) for r in published['records']]
        require(publication.preserved_history(db, five_ids) == published['preserved_approved'],
                'Published provider history changed')
        for item in published['inserts']:
            actual = db.execute('SELECT '+','.join(item['row'])+' FROM '+item['table']+' WHERE '+
                ' AND '.join(k+'=%s' for k in item['keys']), tuple(item['keys'].values())).fetchall()
            require(comparable(actual) == [item['row']], 'Published provider metadata changed')
        for item in published['rollups']:
            actual = db.execute('SELECT '+','.join(item['fields'])+' FROM '+item['table']+
                ' WHERE '+item['key']+'=%s', (item['identity'],)).fetchone()
            require(actual == item['fields'], 'Published provider review changed')
        from scripts.repair_decomposed_nab_metadata import RUNNER as repair_runner, IDENTITY, CONTRACT, PLAIN
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
