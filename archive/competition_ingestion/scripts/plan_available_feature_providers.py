"""Freeze reusable provider identities, semantic ports and existing evidence."""
import hashlib
import inspect
import json
from pathlib import Path
from uuid import NAMESPACE_URL,uuid5

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

import sciona.atoms.ml.tabular.available_features as provider
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from sciona.available_feature_contracts import contracts
from sciona.ghost.registry import REGISTRY

ROOT=Path(__file__).resolve().parents[1]
SOURCE_VERSION='0e37a912-48ef-5551-90a3-08290faccb9a'
MODULE='sciona.atoms.ml.tabular.available_features'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence():
    directory=ROOT/'docs/reviews'
    discovery=json.loads((directory/'available_feature_providers.json').read_text())
    tests=json.loads((directory/'available_feature_provider_tests.json').read_text())
    native=json.loads((directory/'competition_nasa_first_native_wrapper.json').read_text())
    if not all(report['passed'] for report in [discovery,tests,native]) or discovery['provider_count']!=13 or tests['tests_passed']!=137:
        raise ValueError('Complete current provider evidence required')
    if sha(Path(provider.__file__))!=discovery['provider_source_sha256']:
        raise ValueError('Provider source drift')
    if sha(ROOT/'scripts/validate_available_feature_providers.py')!=discovery['validator_sha256']:
        raise ValueError('Discovery validator drift')
    closure={**native['local_source_closure_sha256'],**tests['implementation_sha256'],**tests['test_sha256']}
    for name,digest in closure.items():
        if sha(ROOT/name)!=digest:raise ValueError('Qualification source drift: '+name)
    return dict(reports={name:sha(directory/name) for name in ['available_feature_providers.json',
        'available_feature_provider_tests.json','competition_nasa_first_native_wrapper.json']},
        source_closure=closure,provider_source_sha256=sha(Path(provider.__file__)),
        contract_sha256=sha(ROOT/'sciona/available_feature_contracts.py'),planner_sha256=sha(Path(__file__)))


def plan():
    reviewed=evidence();repo=ROOT.parent/'sciona-atoms-ml'
    specs=[spec for spec in _parse_registered_atoms(repo=ProviderRepo('sciona-atoms-ml',repo),artifact_root=repo/'src/sciona/atoms') if spec.import_module==MODULE]
    if len(specs)!=13:raise ValueError('Exactly thirteen registered feature providers required')
    atoms=[]
    for spec in sorted(specs,key=lambda s:s.source_symbol):
        runtime=MODULE+'.'+spec.source_symbol
        function=REGISTRY[runtime]['impl']
        if Path(inspect.getsourcefile(function)).resolve()!=Path(provider.__file__).resolve():raise ValueError('Provider shadowing')
        inputs,outputs=contracts(spec.source_symbol,function)
        if [p['name'] for p in inputs]!=list(inspect.signature(function).parameters):raise ValueError('Callable port drift')
        atoms.append(dict(fqdn=spec.fqdn,artifact_id=str(uuid5(NAMESPACE_URL,'sciona-provider-draft:'+spec.fqdn)),
            version_id=str(spec.version_id),content_hash=spec.content_hash,runtime_fqdn=runtime,
            description=inspect.getdoc(function),inputs=inputs,outputs=outputs))
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        source=db.execute('SELECT a.artifact_id,a.fqdn,v.version_id,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s',(SOURCE_VERSION,)).fetchone()
        if not source:raise ValueError('Original source intake missing')
        provenance={k:str(v) for k,v in source.items()}
        for atom in atoms:
            for table,key in [('artifacts','artifact_id'),('atoms','atom_id')]:
                rows=db.execute(f'SELECT {key} AS identity FROM {table} WHERE fqdn=%s',(atom['fqdn'],)).fetchall()
                if len(rows)>1 or any(str(row['identity'])!=atom['artifact_id'] for row in rows):raise ValueError('Provider identity conflict')
            rows=db.execute('SELECT artifact_id,content_hash FROM artifact_versions WHERE version_id=%s',(atom['version_id'],)).fetchall()
            if any(str(row['artifact_id'])!=atom['artifact_id'] or row['content_hash']!=atom['content_hash'] for row in rows):raise ValueError('Provider version conflict')
    result=dict(approved=False,catalog_mutations=0,trust_tier_candidate=3,atoms=atoms,
        source_provenance=provenance,evidence=reviewed,
        scope='Domain-neutral materialized feature operations with explicit availability, units and runtime policies. Source intake remains unchanged.',
        limitations=['Static symbolic propagation is unsupported.','Draft planning is not publication or approval.'])
    prior=ROOT/'docs/reviews/available_feature_provider_plan.json'
    if prior.exists() and json.loads(prior.read_text())['source_provenance']!=provenance:raise ValueError('Frozen source provenance drift')
    return result


if __name__=='__main__':
    result=plan()
    (ROOT/'docs/reviews/available_feature_provider_plan.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(planned_atoms=len(result['atoms']),input_ports=sum(len(a['inputs']) for a in result['atoms']),
        output_ports=sum(len(a['outputs']) for a in result['atoms']),catalog_mutations=0,approved=False)))
