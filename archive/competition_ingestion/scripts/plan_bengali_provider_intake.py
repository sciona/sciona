"""Freeze unapproved Bengali provider identities and complete port contracts."""
import json
from uuid import NAMESPACE_URL, uuid5

from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from scripts.import_flavours_provider_drafts import connect
from scripts.import_residual_execution_drafts import comparable
from scripts import review_bengali_provider_contracts as contracts

ROOT = contracts.ROOT


def plan():
    reviewed = contracts.review()
    report_path = ROOT/'docs/reviews/bengali_provider_contracts.json'
    if reviewed != json.loads(report_path.read_text()):
        raise ValueError('Reviewed provider contracts changed')
    directory = ROOT.parent/'sciona-atoms-dl'
    parsed = {s.import_module+'.'+s.source_symbol: s for s in _parse_registered_atoms(
        repo=ProviderRepo(directory.name, directory), artifact_root=directory/'src/sciona/atoms')}
    records = []
    with connect() as db:
        db.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        owners = db.execute("SELECT DISTINCT a.owner_id::text,a.source_repo_id::text FROM atoms a "
            "JOIN atom_source_repositories r USING(source_repo_id) WHERE r.repo_name='sciona-atoms-dl'").fetchall()
        if len(owners) != 1:
            raise ValueError('Unique provider ownership required')
        for provider in reviewed['providers']:
            runtime = provider['runtime_fqdn']
            spec = parsed[runtime]
            identity = uuid5(NAMESPACE_URL, 'sciona-provider-draft:'+spec.fqdn)
            for table, key in [('atoms', 'atom_id'), ('artifacts', 'artifact_id')]:
                if db.execute(f'SELECT 1 FROM {table} WHERE fqdn=%s OR {key}=%s',
                              (spec.fqdn, identity)).fetchone():
                    raise ValueError('New provider identity already exists')
            for table in ('atom_versions', 'artifact_versions'):
                if db.execute(f'SELECT 1 FROM {table} WHERE version_id=%s', (spec.version_id,)).fetchone():
                    raise ValueError('Version identity collision')
            fields = {name: getattr(spec, name) for name in [
                'fqdn', 'namespace_root', 'namespace_path', 'repo_name', 'source_module_path',
                'import_module', 'source_symbol', 'description', 'domain_tags', 'source_kind', 'is_ffi']}
            ports = []
            for direction, key in [('input', 'inputs'), ('output', 'outputs')]:
                for ordinal, port in enumerate(provider[key]):
                    ports.append(dict(direction=direction, ordinal=ordinal, name=port['name'],
                        type_desc=port['type_desc'], required=port.get('required', True),
                        default_value_repr='', constraints=port['constraints']))
            records.append(dict(runtime_fqdn=runtime, atom_id=str(identity),
                version_id=str(spec.version_id), content_hash=spec.content_hash,
                provider_sha256=contracts.sha(spec.file_path), semver=spec.semver,
                fingerprint=comparable(spec.fingerprint), seed_fields=comparable(fields), interfaces=ports))
    if len(records) != 14 or len({r['version_id'] for r in records}) != 14:
        raise ValueError('Fourteen unique provider versions required')
    return dict(format='bengali-provider-intake-plan.v1', passed=True, applied=False,
        approved=False, catalog_mutations=0, owner=owners[0], new_providers=records,
        new_provider_count=len(records), interface_rows_per_catalog=sum(len(r['interfaces']) for r in records),
        execution_evidence_commit=contracts.COMMIT,
        execution_evidence_sha256=reviewed['execution_qualification']['report_sha256'],
        graph_sha256=reviewed['execution_qualification']['graph_sha256'],
        provider_contracts_sha256=contracts.sha(report_path),
        limitations=[
            'Draft plan only. Canonical status must be draft; legacy status flagged; neither may be served.',
            'Source licensing, notices and initialization permissions remain unresolved; no approval is implied.',
            'Before applying, revalidate identities and contracts and prove transaction rollback with injected failures.',
            'Dependencies, plain descriptions, parameter metadata and semantic approval remain subsequent requirements.',
        ])


if __name__ == '__main__':
    result = plan()
    (ROOT/'docs/reviews/bengali_provider_intake_plan.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('passed', 'applied', 'new_provider_count', 'interface_rows_per_catalog')}))
