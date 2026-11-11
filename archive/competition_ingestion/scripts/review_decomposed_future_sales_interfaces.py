"""Review exact registration, callable ports, reuse contracts and catalog duplicates."""
import importlib
import inspect
import json
import subprocess
from pathlib import Path
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from sciona.decomposed_future_sales_graph import build_decomposed_future_sales_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from typing import get_type_hints, get_args, get_origin
import numpy as np
from sciona.architect.models import EdgeKind
from scripts.promote_connectomics_execution import boundary_ports, provider_ports
from scripts.decomposed_future_sales_provider_contracts import CONTRACTS
from scripts.import_flavours_provider_drafts import connect, ROOT
from scripts.audit_decomposed_future_sales_readiness import sha

PREFIX='sciona.atoms.ml.decomposed_future_sales_execution.'
EVIDENCE='docs/reviews/decomposed_future_sales_graph_execution.json'


def interfaces(function,node):
    kinds={object:'object',dict:'dict',np.ndarray:'ndarray',list:'list',int:'int',float:'float'}
    signature,hints=inspect.signature(function),get_type_hints(function)
    if list(signature.parameters)!=[p.name for p in node.inputs]:raise ValueError('Input order differs')
    ports=[]
    for ordinal,(name,param) in enumerate(signature.parameters.items()):
        port=node.inputs[ordinal]
        if param.kind!=inspect.Parameter.POSITIONAL_OR_KEYWORD or param.default is not inspect.Parameter.empty:
            raise ValueError('Required keyword input expected')
        if hints.get(name) not in kinds or port.type_desc!=kinds[hints[name]] or not port.required or port.default_value_repr not in ('',None):
            raise ValueError('Input annotation differs')
        ports.append(dict(direction='input',name=name,ordinal=ordinal,type_desc=port.type_desc,required=True,default_value_repr='',constraints=port.constraints))
    returns=get_args(hints['return']) if get_origin(hints['return']) is tuple else (hints['return'],)
    if any(t not in kinds for t in returns) or [kinds[t] for t in returns]!=[p.type_desc for p in node.outputs]:
        raise ValueError('Return annotation differs')
    ports.extend(dict(direction='output',name=p.name,ordinal=i,type_desc=p.type_desc,required=True,default_value_repr='',constraints=p.constraints) for i,p in enumerate(node.outputs))
    return ports


def review():
    raw=(ROOT/EVIDENCE).read_bytes()
    if raw!=subprocess.check_output(['git','show','aeabc8da:'+EVIDENCE],cwd=ROOT):
        raise ValueError('Committed full execution evidence changed')
    execution=json.loads(raw)
    if any(sha(ROOT/p)!=h for p,h in execution['code_sha256'].items()): raise ValueError('Executed implementation changed')
    graph=build_decomposed_future_sales_graph()
    if encode_execution_graph(graph)[0]!=execution['serialized_graph_sha256']: raise ValueError('Executed graph changed')
    wanted={n.matched_primitive for n in graph.nodes};specs={}
    for repo in ('sciona-atoms','sciona-atoms-ml'):
        directory=ROOT.parent/repo
        for spec in _parse_registered_atoms(repo=ProviderRepo(repo,directory),artifact_root=directory/'src/sciona/atoms'):
            name=spec.import_module+'.'+spec.source_symbol
            if name in wanted:
                if name in specs: raise ValueError('Ambiguous registration')
                specs[name]=spec
    if set(specs)!=wanted or len(specs)!=9: raise ValueError('Exact nine-provider inventory required')
    ports={};records=[]
    for node in graph.nodes:
        name=node.matched_primitive;spec=specs[name]
        module=importlib.import_module(spec.import_module);fn=getattr(module,spec.source_symbol)
        if Path(inspect.getfile(inspect.unwrap(fn))).resolve()!=spec.file_path.resolve(): raise ValueError('Runtime origin differs')
        if sha(spec.file_path)!=execution['provider_sha256'][name]: raise ValueError('Executed provider changed')
        ports[node.node_id]=provider_ports(node,fn)
        if not name.startswith(PREFIX): continue
        contract=CONTRACTS[spec.source_symbol]
        if list(contract['parameters'])!=list(inspect.signature(fn).parameters): raise ValueError('Semantic parameter inventory differs')
        if any(not value for value in contract.values()): raise ValueError('Incomplete semantic contract')
        try: module.concrete_only(*[object() for _ in node.inputs])
        except ValueError: pass
        else: raise ValueError('Recipe witness must reject symbolic execution')
        records.append(dict(runtime_fqdn=name,catalog_fqdn=spec.fqdn,version_id=str(spec.version_id),
            content_hash=spec.content_hash,provider_sha256=sha(spec.file_path),interfaces=interfaces(fn,node),**contract))
    if len(records)!=8 or len(CONTRACTS)!=8: raise ValueError('Eight complete recipe contracts required')
    callable_only={n.node_id for n in graph.nodes
        if any(e.source_id==n.node_id for e in graph.edges)
        and all(e.edge_kind==EdgeKind.CALLABLE_INJECTION for e in graph.edges if e.source_id==n.node_id)}
    concrete_graph=graph.model_copy(update={'nodes':[n for n in graph.nodes if n.node_id not in callable_only]})
    boundary=boundary_ports(concrete_graph,ports)
    if [p.name for p in boundary['input']]!=['payload'] or [p.name for p in boundary['output']]!=['result']:
        raise ValueError('Complete invocation boundary differs')
    existing=[]
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        for r in records:
            module,symbol=r['runtime_fqdn'].rsplit('.',1)
            if db.execute('SELECT 1 FROM atoms WHERE import_module=%s AND source_symbol=%s',(module,symbol)).fetchone():
                raise ValueError('Runtime implementation already cataloged under another identity')
            for prefix,key in [('atom','atom_id'),('artifact','artifact_id')]:
                found=db.execute(f'''SELECT DISTINCT a.{key} FROM {prefix}s a LEFT JOIN {prefix}_versions v USING({key})
                    WHERE a.fqdn=%s OR v.content_hash=%s OR v.version_id=%s''',
                    (r['catalog_fqdn'],r['content_hash'],r['version_id'])).fetchall()
                if found: raise ValueError('New adapter identity or implementation already exists')
        for name,spec in specs.items():
            if name.startswith(PREFIX): continue
            states={}
            for prefix,key in [('atom','atom_id'),('artifact','artifact_id')]:
                states[prefix]=db.execute(f'''SELECT a.{key}::text AS identity,a.status,a.is_publishable,
                    v.version_id::text,v.content_hash,v.trust_tier FROM {prefix}s a JOIN {prefix}_versions v USING({key})
                    WHERE a.fqdn=%s AND v.is_latest''',(spec.fqdn,)).fetchall()
                if prefix=='artifact' and not states[prefix]:
                    if db.execute('SELECT 1 FROM artifacts WHERE fqdn=%s',(spec.fqdn,)).fetchone():
                        raise ValueError('Canonical provider exists without a selected version')
                    continue
                if (len(states[prefix])!=1 or states[prefix][0]['status']!='approved'
                        or states[prefix][0]['is_publishable'] is not True
                        or states[prefix][0]['version_id']!=str(spec.version_id)
                        or states[prefix][0]['content_hash']!=spec.content_hash
                        or states[prefix][0]['trust_tier']!=3):
                    raise ValueError('Reused provider version is not approved and publishable')
            if states['artifact'] and states['atom'][0]['identity']!=states['artifact'][0]['identity']:
                raise ValueError('Reused provider identity differs between catalogs')
            identity=states['atom'][0]['identity']
            eligible=db.execute('SELECT atom_is_publishable(%s) AS eligible',(identity,)).fetchone()['eligible']
            legacy_ports=db.execute('''SELECT version_id::text,direction,name,ordinal,type_desc,required,default_value_repr,constraints
                FROM atom_io_specs WHERE atom_id=%s AND (version_id=%s OR version_id IS NULL)
                ORDER BY version_id NULLS FIRST,direction,ordinal''',(identity,str(spec.version_id))).fetchall()
            existing.append(dict(runtime_fqdn=name,catalog_fqdn=spec.fqdn,states=states,
                legacy_native_eligible=eligible,legacy_ports=legacy_ports,canonical_mirror_required=not bool(states['artifact'])))
    return dict(format='decomposed-future_sales-interfaces.v1',passed=True,approved=False,catalog_mutations=0,
        graph_digest=execution['serialized_graph_sha256'],new_providers=records,existing_providers=existing,
        root_inputs=['payload'],root_outputs=['result'],execution_evidence_sha256=sha(ROOT/EVIDENCE),
        code_sha256={p:sha(ROOT/p) for p in ['scripts/review_decomposed_future_sales_interfaces.py',
            'scripts/decomposed_future_sales_provider_contracts.py','scripts/promote_connectomics_execution.py']},
        limits=['Registration and interface qualification only; draft intake, provider publication and corrected original CDG approval remain pending.',
                'Duplicate checks cover runtime identity, version identity and content fingerprint. Semantic equivalence with differently named implementations requires separate review.',
                'Numeric grouping and training remain reusable; root entity roles and clipping choices remain explicit recipe policy.'])


if __name__=='__main__':
    report=review()
    (ROOT/'docs/reviews/decomposed_future_sales_interfaces.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,new_providers=len(report['new_providers']),existing_providers=len(report['existing_providers']),approved=False)))
