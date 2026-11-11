"""Freeze exact new-provider draft intake while preserving existing identities."""
import hashlib
import importlib
import inspect
import json
import types
from pathlib import Path
from typing import get_args,get_origin,get_type_hints,Union
from uuid import NAMESPACE_URL,uuid5
import numpy as np
import psycopg
from psycopg.rows import dict_row
from dotenv import dotenv_values
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from sciona.flavours_decomposed_evaluation import build_evaluated_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from scripts.inventory_flavours_decomposed_providers import inventory,ROOT


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def kind(annotation):
    origin=get_origin(annotation) or annotation
    if origin is np.ndarray:return 'numpy.ndarray'
    if origin in (list,dict,int,float,bool):return origin.__name__
    if origin in (Union,types.UnionType):
        kinds={kind(a) for a in get_args(annotation)}
        if len(kinds)==1:return kinds.pop()
    raise ValueError('Unsupported interface annotation: '+str(annotation))


def interfaces(function,nodes):
    signature=inspect.signature(function);hints=get_type_hints(function);ports=[]
    for ordinal,(name,parameter) in enumerate(signature.parameters.items()):
        if parameter.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD,inspect.Parameter.KEYWORD_ONLY):
            raise ValueError('Unsupported parameter kind')
        type_desc=kind(hints[name])
        for node in nodes:
            matching=[p for p in node.inputs if p.name==name]
            if len(matching)!=1 or matching[0].type_desc!=type_desc:raise ValueError('Graph input contract differs: '+name)
        ports.append(dict(direction='input',name=name,ordinal=ordinal,type_desc=type_desc,
            required=parameter.default is inspect.Parameter.empty,
            default_value_repr='' if parameter.default is inspect.Parameter.empty else repr(parameter.default),
            constraints='See exact versioned provider contract for shape, units, selection and numerical limits.'))
    if any({p.name for p in node.inputs}!=set(signature.parameters) for node in nodes):raise ValueError('Unexpected graph input')
    if nodes:
        outputs=[(p.name,p.type_desc) for p in nodes[0].outputs]
        if any([(p.name,p.type_desc) for p in n.outputs]!=outputs for n in nodes):raise ValueError('Repeated output contracts differ')
    elif function.__name__=='fit_configured_regression_model':outputs=[('state','dict'),('actual_rounds','int')]
    else:raise ValueError('Explicit indirect provider output contract required')
    annotation=hints['return']
    if annotation is not tuple:
        annotations=get_args(annotation) if get_origin(annotation) is tuple else (annotation,)
        if [kind(a) for a in annotations]!=[t for _,t in outputs]:raise ValueError('Output annotations differ from explicit graph contract')
    elif not nodes:raise ValueError('Bare tuple needs materialized graph output evidence')
    for ordinal,(name,type_desc) in enumerate(outputs):
        ports.append(dict(direction='output',name=name,ordinal=ordinal,type_desc=type_desc,required=True,default_value_repr='',
            constraints='Exact graph tuple ordinal; materialized outputs qualified by full evaluated execution.'))
    return ports


def plan():
    current=inventory();graph=build_evaluated_graph();digest,_,_=encode_execution_graph(graph)
    evidence_path=ROOT/'docs/reviews/flavours_decomposed_evaluation_execution.json'
    evidence=json.loads(evidence_path.read_text())
    if not (evidence['passed'] and evidence['synthetic_only'] and evidence['all_nodes_uncached'] and evidence['executed_nodes']==841
            and evidence['total_fits']==115 and evidence['graph_digest']==digest==current['graph_digest']):
        raise ValueError('Complete exact evaluated graph qualification required')
    for path,expected in evidence['source_sha256'].items():
        if sha(ROOT/path)!=expected:raise ValueError('Execution source drift: '+path)
    known=set(evidence['source_sha256'].values())|set(evidence['provider_sha256'].values())
    if any(r['provider_sha256'] not in known for r in current['providers']):raise ValueError('Unqualified provider source in closure')
    grouped={}
    for node in graph.nodes:grouped.setdefault(node.matched_primitive,[]).append(node)
    new=[r for r in current['providers'] if not r['canonical_latest'] and not r['legacy_latest']]
    existing=[r for r in current['providers'] if r not in new]
    if len(new)!=31 or len(existing)!=8:raise ValueError('Reviewed new/existing provider split changed')
    directory=ROOT.parent/'sciona-atoms-ml'
    parsed={s.import_module+'.'+s.source_symbol:s for s in _parse_registered_atoms(repo=ProviderRepo(directory.name,directory),artifact_root=directory/'src/sciona/atoms')}
    records=[]
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        owners=db.execute("SELECT DISTINCT a.owner_id::text,a.source_repo_id::text FROM atoms a JOIN atom_source_repositories r USING(source_repo_id) WHERE r.repo_name='sciona-atoms-ml'").fetchall()
        if len(owners)!=1:raise ValueError('Unique provider ownership required')
        for record in new:
            runtime=record['runtime_fqdn'];spec=parsed[runtime]
            if str(spec.version_id)!=record['version_id'] or spec.content_hash!=record['content_hash'] or sha(spec.file_path)!=record['provider_sha256']:
                raise ValueError('Provider identity drift')
            for table in ('atoms','artifacts'):
                if db.execute(f'SELECT 1 FROM {table} WHERE fqdn=%s',(spec.fqdn,)).fetchone():raise ValueError('New identity already exists')
            for table in ('atom_versions','artifact_versions'):
                if db.execute(f'SELECT 1 FROM {table} WHERE version_id=%s',(spec.version_id,)).fetchone():raise ValueError('Version identity collision')
            function=getattr(importlib.import_module(spec.import_module),spec.source_symbol)
            fields={name:getattr(spec,name) for name in ['fqdn','namespace_root','namespace_path','repo_name','source_module_path',
                'import_module','source_symbol','description','domain_tags','source_kind','is_ffi']}
            records.append(dict(record,atom_id=str(uuid5(NAMESPACE_URL,'sciona-provider-draft:'+spec.fqdn)),
                seed_fields=fields,semver=spec.semver,fingerprint=spec.fingerprint,
                interfaces=interfaces(function,grouped.get(runtime,[]))))
    return dict(passed=True,applied=False,catalog_mutations=0,approved=False,graph_digest=digest,
        execution_evidence_sha256=sha(evidence_path),owner=owners[0],new_providers=records,existing_providers=existing,
        new_provider_count=len(records),existing_provider_count=len(existing),interface_rows_per_catalog=sum(len(r['interfaces']) for r in records),
        limitations=['Draft intake plan only; no publication approval.','Existing providers are excluded from these mutations and require separate canonical/legacy reconciliation.','Dependency links and serving remain subsequent review steps after exact identities are staged.'])


if __name__=='__main__':
    result=plan()
    (ROOT/'docs/reviews/flavours_provider_intake_plan.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['passed','applied','new_provider_count','existing_provider_count','interface_rows_per_catalog']}))
