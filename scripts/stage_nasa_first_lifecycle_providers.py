"""Atomically stage qualified lifecycle provider drafts; default is rollback."""
import argparse
import json
from uuid import UUID,uuid5

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import AtomSeedRow,_parse_registered_atoms
from scripts.import_residual_execution_drafts import ensure_row
from scripts.plan_nasa_first_lifecycle_providers import ROOT,plan,evidence

RUNNER='nasa-first-lifecycle-provider-draft.v1'


def stage(apply=False,fault_after=None):
    proposed=plan();repo=ROOT.parent/'sciona-atoms-ml'
    specs={spec.fqdn:spec for spec in _parse_registered_atoms(repo=ProviderRepo('sciona-atoms-ml',repo),
        artifact_root=repo/'src/sciona/atoms') if spec.fqdn in {atom['fqdn'] for atom in proposed['atoms']}}
    created=0
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c statement_timeout=30000 -c lock_timeout=10000') as db:
        db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(RUNNER,))
        source=db.execute('SELECT content_hash FROM artifact_versions WHERE version_id=%s FOR SHARE',
            (proposed['source_provenance']['version_id'],)).fetchone()
        if not source or source['content_hash']!=proposed['source_provenance']['content_hash']:raise ValueError('Source provenance drift')
        owners=db.execute("SELECT DISTINCT a.owner_id,a.source_repo_id FROM atoms a JOIN atom_source_repositories r USING(source_repo_id) WHERE r.repo_name='sciona-atoms-ml'").fetchall()
        if len(owners)!=1:raise ValueError('Ambiguous provider ownership')
        columns={r['column_name'] for r in db.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='artifacts'")}

        def ensure(table,key,row):
            nonlocal created
            created+=ensure_row(db,table,key,row)
            if fault_after is not None and created>=fault_after:raise RuntimeError('Injected draft write failure')

        for atom in proposed['atoms']:
            spec=specs[atom['fqdn']]
            if str(spec.version_id)!=atom['version_id'] or spec.content_hash!=atom['content_hash']:raise ValueError('Provider source drift')
            identity=UUID(atom['artifact_id']);version=UUID(atom['version_id'])
            fields={key:getattr(spec,key) for key in ['fqdn','namespace_root','namespace_path','repo_name','source_module_path',
                'import_module','source_symbol','description','domain_tags','source_kind','is_ffi']}
            fields.update(source_package=spec.namespace_root,status='flagged',is_publishable=False)
            legacy=AtomSeedRow(**fields).as_dict(owner_id=str(owners[0]['owner_id']),source_repo_id=str(owners[0]['source_repo_id']))
            legacy['atom_id']=identity
            ensure('atoms',{'atom_id':identity},legacy)
            canonical={key:value for key,value in legacy.items() if key in columns and key not in {'created_at','updated_at'}}
            canonical.update(artifact_id=identity,artifact_kind='atom',status='draft')
            ensure('artifacts',{'artifact_id':identity},canonical)
            for table,key in [('atom_versions','atom_id'),('artifact_versions','artifact_id')]:
                ensure(table,{'version_id':version},dict(version_id=version,**{key:identity},content_hash=spec.content_hash,
                    semver=spec.semver,is_latest=True,trust_tier=3,s3_key='',fingerprint=spec.fingerprint))
            for direction,ports in [('input',atom['inputs']),('output',atom['outputs'])]:
                for ordinal,port in enumerate(ports):
                    common=dict(version_id=version,direction=direction,ordinal=ordinal,
                        **{key:port[key] for key in ['name','type_desc','constraints','required','default_value_repr']})
                    for table,key in [('artifact_io_specs','artifact_id'),('atom_io_specs','atom_id')]:
                        row={key:identity,**common}
                        if key=='artifact_id':row['dim_signature']=port['dim_signature']
                        ensure(table,{key:identity,'version_id':version,'direction':direction,'name':port['name']},row)
            audit_id=uuid5(version,RUNNER)
            ensure('artifact_audit_evidence',{'evidence_id':audit_id},dict(evidence_id=audit_id,artifact_id=identity,
                version_id=version,audit_type='asset_integrity_check',passed=True,status='completed',source_kind='automated',
                runner_version=RUNNER,details=Jsonb(dict(scope='Non-publishable draft identity, ports and evidence binding only.',
                    qualification=proposed['evidence'],source_provenance=proposed['source_provenance']))))
            for view,key in [('catalog_artifacts_served','artifact_id'),('catalog_atoms_served','atom_id')]:
                if db.execute(f'SELECT 1 FROM {view} WHERE {key}=%s',(identity,)).fetchone():raise ValueError('Draft unexpectedly served')
        if evidence()!=proposed['evidence']:raise ValueError('Qualification evidence changed during staging')
        if not apply:db.rollback()
    return dict(applied=apply,rows_created=created,draft_atoms=len(proposed['atoms']),approved=False,served=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');parser.add_argument('--fault-after',type=int)
    args=parser.parse_args();print(json.dumps(stage(args.apply,args.fault_after)))
