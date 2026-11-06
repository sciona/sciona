"""Freeze exact registered interfaces, reuse contracts and identity-duplicate checks."""
import importlib
import inspect
import json
import subprocess
from pathlib import Path
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from sciona.architect.models import EdgeKind
from sciona.decomposed_webtraffic_graph import build_decomposed_webtraffic_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from scripts.review_decomposed_future_sales_interfaces import interfaces
from scripts.decomposed_webtraffic_provider_contracts import CONTRACTS
from scripts.promote_connectomics_execution import boundary_ports,provider_ports
from scripts.import_flavours_provider_drafts import ROOT,connect
from scripts.audit_decomposed_webtraffic_readiness import sha

EVIDENCE='docs/reviews/decomposed_webtraffic_graph_execution.json'


def review():
    raw=(ROOT/EVIDENCE).read_bytes()
    if raw!=subprocess.check_output(['git','show','15f322b8:'+EVIDENCE],cwd=ROOT):raise ValueError('Committed graph execution changed')
    execution=json.loads(raw)
    if any(sha(ROOT/p)!=h for p,h in execution['implementation_sha256'].items()):raise ValueError('Executed implementation changed')
    graph=build_decomposed_webtraffic_graph()
    if encode_execution_graph(graph)[0]!=execution['graph_sha256']:raise ValueError('Executed graph changed')
    directory=ROOT.parent/'sciona-atoms-dl';wanted={n.matched_primitive for n in graph.nodes};specs={}
    for spec in _parse_registered_atoms(repo=ProviderRepo('sciona-atoms-dl',directory),artifact_root=directory/'src/sciona/atoms'):
        name=spec.import_module+'.'+spec.source_symbol
        if name not in wanted:continue
        if name in specs:raise ValueError('Ambiguous registration')
        specs[name]=spec
    if set(specs)!=wanted or len(specs)!=19 or len(CONTRACTS)!=19:raise ValueError('Exact19-provider inventory required')
    records=[];ports={}
    for node in graph.nodes:
        spec=specs[node.matched_primitive];module=importlib.import_module(spec.import_module);fn=getattr(module,spec.source_symbol)
        if Path(inspect.getfile(inspect.unwrap(fn))).resolve()!=spec.file_path.resolve():raise ValueError('Runtime origin differs')
        if sha(spec.file_path)!=execution['provider_sha256'][node.matched_primitive]:raise ValueError('Executed provider changed')
        contract=CONTRACTS[spec.source_symbol]
        if list(contract['parameters'])!=list(inspect.signature(fn).parameters) or any(not v for v in contract.values()):
            raise ValueError('Complete ordered semantic parameter inventory required')
        try:module.concrete_only(*[object() for _ in node.inputs])
        except ValueError:pass
        else:raise ValueError('Concrete-only witness must reject symbolic execution')
        ports[node.node_id]=provider_ports(node,fn)
        records.append(dict(runtime_fqdn=node.matched_primitive,catalog_fqdn=spec.fqdn,version_id=str(spec.version_id),
            content_hash=spec.content_hash,provider_sha256=sha(spec.file_path),interfaces=interfaces(fn,node),**contract))
    callable_only={n.node_id for n in graph.nodes if any(e.source_id==n.node_id for e in graph.edges)
        and all(e.edge_kind==EdgeKind.CALLABLE_INJECTION for e in graph.edges if e.source_id==n.node_id)}
    if callable_only!={'advance','save_checkpoint'}:raise ValueError('Exact callable boundary required')
    concrete=graph.model_copy(update={'nodes':[n for n in graph.nodes if n.node_id not in callable_only]})
    boundary=boundary_ports(concrete,ports)
    if [p.name for p in boundary['input']]!=['payload'] or [p.name for p in boundary['output']]!=['result']:
        raise ValueError('Complete root boundary differs')
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        for r in records:
            module,symbol=r['runtime_fqdn'].rsplit('.',1)
            if db.execute('SELECT 1 FROM atoms WHERE import_module=%s AND source_symbol=%s',(module,symbol)).fetchone():
                raise ValueError('Runtime implementation already cataloged')
            for prefix,key in [('atom','atom_id'),('artifact','artifact_id')]:
                if db.execute(f'''SELECT 1 FROM {prefix}s a LEFT JOIN {prefix}_versions v USING({key})
                    WHERE a.fqdn=%s OR v.version_id=%s OR v.content_hash=%s''',(r['catalog_fqdn'],r['version_id'],r['content_hash'])).fetchone():
                    raise ValueError('Implementation or version identity already cataloged')
    paths=['scripts/review_decomposed_webtraffic_interfaces.py','scripts/decomposed_webtraffic_provider_contracts.py',
        'scripts/review_decomposed_future_sales_interfaces.py','scripts/promote_connectomics_execution.py']
    return dict(format='decomposed-webtraffic-interfaces.v1',passed=True,approved=False,catalog_mutations=0,
        graph_digest=execution['graph_sha256'],new_providers=records,existing_providers=[],
        root_inputs=['payload'],root_outputs=['result'],callable_only_nodes=sorted(callable_only),
        execution_evidence_sha256=sha(ROOT/EVIDENCE),code_sha256={p:sha(ROOT/p) for p in paths},
        limits=['Registration/interface qualification only; semantic provider publication and original CDG approval remain pending.',
                'Duplicate checks cover runtime, version and content identity. Differently named semantic alternatives require separate bounded review.',
                'Object ports include mutable invocation state and caller provenance assumptions; symbolic witnesses reject execution.',
                'Page/calendar and three-model s32 operations retain explicit recipe restrictions; numeric correlation, normalization and integer output alignment accept domain-independent inputs.'])


if __name__=='__main__':
    result=review()
    (ROOT/'docs/reviews/decomposed_webtraffic_interfaces.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,new_providers=len(result['new_providers']),callable_nodes=result['callable_only_nodes'],approved=False)))
