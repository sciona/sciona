"""Verify approved image_tabular providers independently of original CDG status."""
import json
import subprocess
from uuid import UUID

from scripts.import_flavours_provider_drafts import ROOT, connect
from scripts.plan_flavours_provider_intake import sha
from scripts.promote_openvaccine_decomposed_providers import snapshot, PORT_FIELDS
from scripts.review_video_numeric_runtime import review as runtime_review

PLAN = ROOT/'docs/reviews/image_tabular_approved_provider_state.json'
EVIDENCE = {
    'docs/reviews/image_tabular_complete_runtime.json': 'a03ab602',
    'docs/reviews/image_tabular_provider_reuse.json': 'a03ab602',
    'docs/reviews/image_tabular_provider_semantics.json': '2baee159',
    'docs/reviews/complete_image_tabular_provider_publication_plan.json': 'b2acf755',
}


def sources():
    evidence = {}
    for path, commit in EVIDENCE.items():
        raw = (ROOT/path).read_bytes()
        if raw != subprocess.check_output(['git', 'show', commit+':'+path], cwd=ROOT):
            raise ValueError('Immutable provider qualification changed')
        evidence[path] = json.loads(raw)
    graph = evidence['docs/reviews/image_tabular_complete_runtime.json']
    semantics = evidence['docs/reviews/image_tabular_provider_semantics.json']
    for path, digest in graph['software_sha256'].items():
        if sha(ROOT/path) != digest:
            raise ValueError('Qualified provider source changed: '+path)
    if sha(ROOT/'scripts/review_image_tabular_provider_semantics.py') != semantics['reviewer_sha256']:
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
    return {r['runtime_fqdn']: r for r in qualify_plan()['records'] if r['recipe_adapter']}


def freeze():
    evidence = sources()
    publication = evidence['docs/reviews/complete_image_tabular_provider_publication_plan.json']
    graph = evidence['docs/reviews/image_tabular_complete_runtime.json']
    records = []
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        # Only the initial freeze checks the old original-CDG preservation gate.
        from scripts.promote_complete_image_tabular_providers import qualify_plan as published_plan, check as published_check
        if published_plan() != publication:
            raise ValueError('Publication plan differs')
        published_check(db, publication, published=True)
        bindings = list(publication['records'])
        for reused in evidence['docs/reviews/image_tabular_provider_reuse.json']['reused']:
            source = reused['source']
            catalog = reused['catalogs']['artifact']
            bindings.append(dict(source, artifact_id=catalog['identity'], fqdn=source['catalog_fqdn'], interfaces=catalog['ports']))
        for binding in bindings:
            record = {k: binding[k] for k in ('runtime_fqdn', 'artifact_id', 'version_id', 'content_hash')}
            record.update(catalog_fqdn=binding['fqdn'], recipe_adapter='.image_tabular_complete_execution.' in binding['runtime_fqdn'])
            for prefix, key in [('atom', 'legacy_ports'), ('artifact', 'ports')]:
                record[key] = db.execute('SELECT '+','.join(PORT_FIELDS)+f' FROM {prefix}_io_specs WHERE version_id=%s ORDER BY direction,ordinal', (binding['version_id'],)).fetchall()
            if record['ports'] != binding['interfaces'] or record['legacy_ports'] != binding['interfaces']:
                raise ValueError('Reviewed provider interfaces differ')
            records.append(record)
        if len(records) != 6 or len({r['artifact_id'] for r in records}) != 6:
            raise ValueError('Four adapters and two reusable F1 providers required')
        result = dict(passed=True, catalog_mutations=0, records=sorted(records, key=lambda r: r['runtime_fqdn']),
            checker_sha256=sha(__file__), evidence_sha256={p: sha(ROOT/p) for p in EVIDENCE},
            catalog_snapshot=snapshot(db, [UUID(r['artifact_id']) for r in records]), dependencies=dependencies(db, records))
        check(db, result)
    return result


if __name__ == '__main__':
    result = freeze()
    PLAN.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(approved_providers=len(result['records']), catalog_mutations=0)))
