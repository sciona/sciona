"""Freeze new DSB provider drafts; preserve every existing catalog identity."""
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from scripts.inventory_dsb_decomposed_providers import inventory, ROOT, sha
from scripts.plan_dsb_provider_interfaces import plan as interface_plan
from scripts.import_flavours_provider_drafts import connect
from scripts.import_residual_execution_drafts import comparable

PLAN = ROOT/'docs/reviews/dsb_provider_intake_plan.json'
REPOS = ('sciona-atoms', 'sciona-atoms-dl')
FIELDS = ('fqdn','namespace_root','namespace_path','repo_name','source_module_path',
          'import_module','source_symbol','description','domain_tags','source_kind','is_ffi')


def specs():
    result = {}
    for name in REPOS:
        directory = ROOT.parent/name
        for spec in _parse_registered_atoms(repo=ProviderRepo(name,directory),
                                           artifact_root=directory/'src/sciona/atoms'):
            runtime = spec.import_module+'.'+spec.source_symbol
            if runtime in result:
                raise ValueError('Duplicate runtime identity')
            result[runtime] = spec
    return result


def catalog_ports(record):
    result = []
    for port in record['ports']:
        row = {k:port[k] for k in ('direction','name','ordinal','type_desc','required','default_value_repr')}
        row['constraints'] = json.dumps({k:v for k,v in port.items() if k not in row},sort_keys=True)
        result.append(row)
    return result


def plan():
    current = inventory()
    contracts = interface_plan()
    path = ROOT/'docs/reviews/dsb_provider_interface_plan.json'
    if contracts != json.loads(path.read_text()):
        raise ValueError('Interface plan changed')
    interfaces = {r['runtime_fqdn']:r for r in contracts['provider_contracts']}
    parsed = specs()
    new = []; existing = []; owners = {}
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        for repo in REPOS:
            rows = db.execute('''SELECT DISTINCT a.owner_id::text,a.source_repo_id::text
                FROM atoms a JOIN atom_source_repositories r USING(source_repo_id)
                WHERE r.repo_name=%s''',(repo,)).fetchall()
            if len(rows)!=1:
                raise ValueError('Unique repository ownership required: '+repo)
            owners[repo] = rows[0]
        for record in current['providers']:
            runtime = record['runtime_fqdn']; spec = parsed[runtime]
            if record['atom_latest'] or record['artifact_latest']:
                existing.append(record)
                continue
            for table in ('atoms','artifacts'):
                if db.execute(f'SELECT 1 FROM {table} WHERE fqdn=%s',(spec.fqdn,)).fetchone():
                    raise ValueError('Identity exists without latest version')
            for table in ('atom_versions','artifact_versions'):
                if db.execute(f'SELECT 1 FROM {table} WHERE version_id=%s',(spec.version_id,)).fetchone():
                    raise ValueError('Version identity collision')
            new.append(dict(record,atom_id=str(uuid5(NAMESPACE_URL,'sciona-provider-draft:'+spec.fqdn)),
                seed_fields={k:getattr(spec,k) for k in FIELDS},semver=spec.semver,
                fingerprint=spec.fingerprint,owner=owners[spec.repo_name],
                interfaces=catalog_ports(interfaces[runtime])))
    if len(new)!=33 or len(existing)!=1:
        raise ValueError('Reviewed 33-new/one-existing split changed')
    return comparable(dict(approved=False,catalog_mutations=0,graph_digest=current['graph_digest'],
        new_provider_count=len(new),existing_provider_count=len(existing),owners=owners,
        new_providers=new,existing_providers=existing,
        interface_plan_sha256=sha(path),planner_sha256=sha(Path(__file__)),
        execution_evidence_sha256=current['full_graph_evidence_sha256'],
        limitations=['Draft identities and interfaces only; no approval or serving.',
                    'Existing approved identity requires separate eligibility repair.',
                    'Dependency closure and semantic publication review remain required.']))


if __name__=='__main__':
    report=plan()
    PLAN.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('new_provider_count','existing_provider_count','catalog_mutations')}))
