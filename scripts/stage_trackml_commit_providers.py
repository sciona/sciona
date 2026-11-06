"""Stage the reusable selection interface while preserving legacy catalog history."""
import argparse
import importlib
import inspect
import json
import subprocess
from uuid import UUID, uuid5
from psycopg.types.json import Jsonb
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from sciona.decomposed_trackml_commit_graph import build_decomposed_trackml_commit_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from scripts.stage_trackml_coordinate_providers import ROOT, sha, connect, comparable, ensure_row, FIELDS, preserved
from scripts.stage_trackml_nearest_canonical_records import legacy_digest
from scripts.promote_openvaccine_decomposed_providers import snapshot

RUNNER = 'trackml-commit-provider-intake.v1'
PLAN = ROOT/'docs/reviews/trackml_commit_provider_intake_plan.json'
GATES = ROOT/'docs/reviews/trackml_commit_provider_intake_gates.json'
IDENTITIES = {'greedy_track_commit': '8f90d8d6-1269-42fe-8f16-2079a27856ad'}
LEGACY_IDS = [UUID(v) for v in IDENTITIES.values()]


def local_plan():
    path = 'docs/reviews/decomposed_trackml_commit_graph_execution.json'
    raw = (ROOT/path).read_bytes()
    if raw != subprocess.check_output(['git','show','ffdada98:'+path], cwd=ROOT):
        raise ValueError('Immutable graph evidence changed')
    report = json.loads(raw)
    if not report['passed'] or not report['exact_registered_bindings'] or report['explicit_callable_edges'] != 15:
        raise ValueError('Qualified graph bindings required')
    for name, digest in report['implementation_sha256'].items():
        if sha(ROOT/name) != digest: raise ValueError('Qualified implementation changed')
    graph = build_decomposed_trackml_commit_graph()
    if encode_execution_graph(graph)[0] != report['graph_sha256']: raise ValueError('Qualified graph changed')
    nodes = {n.matched_primitive:n for n in graph.nodes if n.node_id == 'commit'}
    directory = ROOT.parent/'sciona-atoms-physics'
    records = []
    for spec in _parse_registered_atoms(repo=ProviderRepo(directory.name,directory), artifact_root=directory/'src/sciona/atoms'):
        runtime = spec.import_module+'.'+spec.source_symbol
        if runtime not in nodes: continue
        if sha(spec.file_path) != report['all_provider_sha256'][runtime]: raise ValueError('Executed provider changed')
        fn = getattr(importlib.import_module(spec.import_module), spec.source_symbol)
        signature = inspect.signature(fn); node = nodes[runtime]
        if list(signature.parameters) != [p.name for p in node.inputs]: raise ValueError('Graph input order differs')
        bounds = ('Integer resource matrix (N,M) with nonnegative int64 IDs strictly below the nonempty boolean used-mask length. Zero is an empty-slot sentinel. Counts are integers: min_nhits >= 1, max_nloss >= 0, max_nrows >= 0; max_loss_fraction is finite in [0,1]. Rows are processed in supplied order. Duplicate IDs within a row count as repeated positions, not unique resources. Provider copies inputs; caller owns ordering and any mutation or reservation of skipped rows.')
        ports = []
        for i,(name,p) in enumerate(signature.parameters.items()):
            scalar = name not in ('hit_matrix','used')
            kind = ('float' if name == 'max_loss_fraction' else 'int') if scalar else 'ndarray'
            required = not scalar
            default = repr(p.default) if scalar else ''
            if p.kind != p.POSITIONAL_OR_KEYWORD or (p.default is p.empty) != required:
                raise ValueError('Unexpected signature')
            port = node.inputs[i]
            if (port.type_desc,port.required,port.default_value_repr) != (kind,required,default):
                raise ValueError('Input contract differs')
            ports.append(dict(direction='input',name=name,ordinal=i,type_desc=kind,required=required,default_value_repr=default,constraints=bounds))
        if [p.name for p in node.outputs] != ['selected','claimed']:
            raise ValueError('Output contract differs')
        for i,port in enumerate(node.outputs):
            if port.type_desc != 'ndarray': raise ValueError('Output type differs')
            description = ('Returns a copied int64 selection matrix and boolean claimed mask with the input shapes. Rejected rows and already-claimed positions are zeroed. Accepted rows claim their resource IDs; rejected rows claim nothing. The sentinel mask entry is always true. Ordered greedy selection is not a globally optimal allocation. Empty matrices and all-zero rows are accepted by this atom, unlike the stricter source recipe.')
            ports.append(dict(direction='output',name=port.name,ordinal=i,type_desc='ndarray',required=True,default_value_repr='',constraints=description))
        records.append(dict(atom_id=IDENTITIES[spec.source_symbol],
            existing_legacy=True, catalog_fqdn=spec.fqdn,runtime_fqdn=runtime,
            version_id=str(spec.version_id),content_hash=spec.content_hash,fingerprint=comparable(spec.fingerprint),
            semver=spec.semver,seed_fields=comparable({k:getattr(spec,k) for k in FIELDS}),
            interfaces=ports,source_sha256=sha(spec.file_path)))
    if len(records)!=1: raise ValueError('One Selection provider required')
    return dict(providers=sorted(records,key=lambda r:r['runtime_fqdn']),stager_sha256=sha(__file__),graph_sha256=report['graph_sha256'],graph_review_sha256=sha(ROOT/path))


def plan():
    result=local_plan(); ids=[UUID(r['atom_id']) for r in result['providers']]
    inserts=[]
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        columns={r['column_name'] for r in db.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='artifacts'")}
        for r in result['providers']:
            identity,version=r['atom_id'],r['version_id']
            if db.execute('SELECT 1 FROM artifacts WHERE artifact_id=%s OR fqdn=%s',(identity,r['catalog_fqdn'])).fetchone(): raise ValueError('Canonical identity already exists')
            if db.execute('SELECT 1 FROM artifact_versions WHERE version_id=%s OR content_hash=%s',(version,r['content_hash'])).fetchone(): raise ValueError('Canonical implementation already exists')
            old=db.execute('SELECT to_jsonb(t) AS row FROM atoms t WHERE atom_id=%s',(identity,)).fetchone()
            if not old: raise ValueError('Existing commit identity missing')
            atom=comparable(old['row'])
            if atom['fqdn']!=r['catalog_fqdn'] or atom['import_module']+'.'+atom['source_symbol']!=r['runtime_fqdn'] or atom['status']!='approved' or atom['is_publishable']:
                raise ValueError('Selection identity/state differs')
            versions=db.execute('SELECT version_id::text,content_hash,fingerprint,trust_tier FROM atom_versions WHERE atom_id=%s AND is_latest',(identity,)).fetchall()
            if comparable(versions)!=[dict(version_id=version,content_hash=r['content_hash'],fingerprint=r['fingerprint'],trust_tier=3)]: raise ValueError('Selection fingerprint differs')
            version_row=comparable(db.execute('SELECT version_id,content_hash,semver,s3_key,fingerprint,trust_tier,is_latest FROM atom_versions WHERE version_id=%s',(version,)).fetchone())
            prefixes=[('artifact','artifact_id')]
            artifact={k:v for k,v in atom.items() if k in columns and k not in ('created_at','updated_at')}
            artifact.update(artifact_id=identity,artifact_kind='atom',status='draft',is_publishable=False)
            inserts.append(dict(table='artifacts',keys={'artifact_id':identity},row=artifact))
            for prefix,key in prefixes:
                inserts.append(dict(table=prefix+'_versions',keys={'version_id':version},row=dict(version_row,**{key:identity})))
                for port in r['interfaces']:
                    row=dict(port,version_id=version,**{key:identity})
                    inserts.append(dict(table=prefix+'_io_specs',keys={k:row[k] for k in (key,'version_id','direction','name')},row=row))
        result.update(inserts=inserts,before=snapshot(db,ids),legacy_sha256=legacy_digest(db,LEGACY_IDS),originals_sha256=preserved(db))
    return result


def reviewed_plan():
    frozen=json.loads(PLAN.read_text()); current=local_plan()
    if any(frozen[k]!=v for k,v in current.items()): raise ValueError('Qualified intake changed')
    return frozen


class InjectedFailure(RuntimeError): pass


def stage(apply=False,fail_after=None):
    p=reviewed_plan(); ids=[UUID(r['atom_id']) for r in p['providers']]; writes=0
    with connect() as db:
        db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(RUNNER,))
        db.execute('SELECT atom_id FROM atoms WHERE atom_id=ANY(%s) FOR SHARE',(LEGACY_IDS,)).fetchall()
        if legacy_digest(db,LEGACY_IDS)!=p['legacy_sha256'] or preserved(db)!=p['originals_sha256']: raise ValueError('Preserved catalog changed')
        def insert(table,keys,row):
            nonlocal writes
            writes+=int(ensure_row(db,table,keys,row))
            if fail_after is not None and writes>=fail_after: raise InjectedFailure('Injected intake failure')
        for item in p['inserts']: insert(item['table'],item['keys'],item['row'])
        for r in p['providers']:
            for prefix in ['artifact']:
                ports=db.execute(f'SELECT direction,name,ordinal,type_desc,required,default_value_repr,constraints FROM {prefix}_io_specs WHERE version_id=%s ORDER BY direction,ordinal',(r['version_id'],)).fetchall()
                if ports!=r['interfaces']: raise ValueError('Stored interfaces differ')
            eid=uuid5(UUID(r['version_id']),RUNNER)
            insert('artifact_audit_evidence',{'evidence_id':eid},dict(evidence_id=eid,artifact_id=r['atom_id'],version_id=r['version_id'],audit_type='asset_integrity_check',passed=True,status='completed',source_kind='automated',runner_version=RUNNER,
                details=Jsonb(dict(plan_sha256=sha(PLAN),source_sha256=r['source_sha256'],scope='Exact source and interface draft intake; no publication approval.'))))
        if db.execute("SELECT 1 FROM artifacts WHERE artifact_id=ANY(%s) AND (status<>'draft' OR is_publishable)",(ids,)).fetchone(): raise ValueError('Unexpected canonical approval')
        for table,key in [('catalog_atoms_served','atom_id'),('catalog_artifacts_served','artifact_id')]:
            if db.execute(f'SELECT 1 FROM {table} WHERE {key}=ANY(%s)',(ids,)).fetchone(): raise ValueError('Draft unexpectedly served')
        if legacy_digest(db,LEGACY_IDS)!=p['legacy_sha256'] or preserved(db)!=p['originals_sha256']: raise ValueError('Preserved catalog modified')
        if reviewed_plan()!=p: raise ValueError('Intake changed during transaction')
        if fail_after is not None: raise ValueError('Fault not reached')
        if not apply: db.rollback()
    return dict(applied=apply,approved=False,rows_created=writes,served_drafts=0,legacy_selection_preserved=True,originals_preserved=True)


def unchanged():
    p=reviewed_plan()
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        if snapshot(db,[UUID(r['atom_id']) for r in p['providers']])!=p['before'] or preserved(db)!=p['originals_sha256']: raise ValueError('Rollback incomplete')


def validate():
    unchanged(); dry=stage(); unchanged()
    points=[1,dry['rows_created']//2,dry['rows_created']]
    for point in points:
        try: stage(fail_after=point)
        except InjectedFailure: pass
        else: raise ValueError('Fault not reached')
        unchanged()
    return dict(passed=True,catalog_mutations=0,dry_run=dry,rollback_points=points,all_rollbacks_complete=True,plan_sha256=sha(PLAN))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['plan','validate','apply'])
    action=parser.parse_args().action
    if action=='plan': result,path=plan(),PLAN
    elif action=='validate': result,path=validate(),GATES
    else:
        g=json.loads(GATES.read_text())
        if not g['passed'] or not g['all_rollbacks_complete'] or g['plan_sha256']!=sha(PLAN): raise ValueError('Matching rollback evidence required')
        first=stage(apply=True);repeat=stage(apply=True)
        if repeat['rows_created']: raise ValueError('Repeated intake changed rows')
        result,path=dict(first=first,repeat=repeat),ROOT/'docs/reviews/trackml_commit_provider_intake_applied.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(action=action,approved=False,report=path.name)))
