"""Read-only reconciliation of reusable F1 providers and new signal adapters."""
import json
from scripts.validate_biosignal_complete_runtime import ROOT, source_inventory, sha
from scripts.import_flavours_provider_drafts import connect
from scripts.promote_openvaccine_decomposed_providers import PORT_FIELDS


def audit():
    report = json.loads((ROOT/'docs/reviews/biosignal_complete_runtime.json').read_text())
    records = source_inventory()
    if report['passed'] is not True or report['providers'] != records:
        raise ValueError('Qualified source inventory differs')
    for path, digest in report['software_sha256'].items():
        if sha(ROOT/path) != digest:
            raise ValueError('Qualified software changed')
    reused = []
    new = []
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        for record in records:
            states = {}
            for prefix, key in [('atom', 'atom_id'), ('artifact', 'artifact_id')]:
                rows = db.execute(f'''SELECT a.{key}::text AS identity,a.fqdn,a.status,a.is_publishable,
                    v.version_id::text,v.content_hash,v.trust_tier,{prefix}_is_publishable(a.{key}) AS eligible,
                    EXISTS(SELECT 1 FROM catalog_{prefix}s_served s WHERE s.{key}=a.{key}) AS served
                    FROM {prefix}s a JOIN {prefix}_versions v USING({key}) WHERE a.fqdn=%s AND v.is_latest''',
                    (record['catalog_fqdn'],)).fetchall()
                if record['role'] == 'recipe_adapter':
                    if rows or db.execute(f'SELECT 1 FROM {prefix}s WHERE fqdn=%s', (record['catalog_fqdn'],)).fetchone():
                        raise ValueError('New adapter identity already exists')
                    continue
                if len(rows) != 1:
                    raise ValueError('Unique reusable provider required')
                row = rows[0]
                if (row['status'] != 'approved' or row['is_publishable'] is not True or row['eligible'] is not True
                    or row['served'] is not True or row['trust_tier'] != 3
                    or any(row[k] != record[k] for k in ('version_id', 'content_hash'))):
                    raise ValueError('Exact approved reusable provider unavailable')
                row['ports'] = db.execute('SELECT '+','.join(PORT_FIELDS)+f' FROM {prefix}_io_specs '
                    'WHERE version_id=%s ORDER BY direction,ordinal', (record['version_id'],)).fetchall()
                states[prefix] = row
            if record['role'] == 'recipe_adapter':
                new.append(record)
            else:
                if states['atom']['identity'] != states['artifact']['identity']:
                    raise ValueError('Provider identities differ across catalogs')
                module, symbol = record['runtime_fqdn'].rsplit('.', 1)
                actual = db.execute('SELECT import_module,source_symbol FROM atoms WHERE atom_id=%s',
                    (states['atom']['identity'],)).fetchone()
                if actual != dict(import_module=module, source_symbol=symbol):
                    raise ValueError('Reusable runtime binding differs')
                if db.execute('SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND NOT passed',
                    (record['version_id'],)).fetchone():
                    raise ValueError('Reusable provider has failed evidence')
                reused.append(dict(source=record, catalogs=states))
    if len(reused) != 2 or len(new) != 4:
        raise ValueError('Two existing numeric providers and four new adapters required')
    return dict(passed=True, read_only=True, catalog_mutations=0, reused=reused, new_adapters=new,
        runtime_report_sha256=sha(ROOT/'docs/reviews/biosignal_complete_runtime.json'), auditor_sha256=sha(__file__),
        limits=['This audit confirms existing numeric version identities, eligibility, serving and runtime bindings.',
            'Snapshot evidence is not permission to skip later live checks, semantic contracts or publication gates.',
            'Four recipe adapters remain absent; the original CDG is not approved by this audit.'])


if __name__ == '__main__':
    result = audit()
    (ROOT/'docs/reviews/biosignal_provider_reuse.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, existing_approved=len(result['reused']), new_adapters=len(result['new_adapters']), catalog_mutations=0)))
