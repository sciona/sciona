"""Read-only exact provider and catalog-state inventory for the decomposed graph."""
import hashlib
import importlib
import inspect
import dis
import json
from types import CodeType
from pathlib import Path
import psycopg
from psycopg.rows import dict_row
from dotenv import dotenv_values
from sciona.flavours_decomposed_evaluation import build_evaluated_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.ghost.registry import REGISTRY
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms

ROOT=Path(__file__).resolve().parents[1]


def registered_call_dependencies(function,registered,visited=None):
    """Follow global function references, not coincidentally named attributes."""
    visited=set() if visited is None else visited
    function=inspect.unwrap(function)
    if function in visited:return set()
    visited.add(function)
    def globals_used(code):
        names={op.argval for op in dis.get_instructions(code) if op.opname in ('LOAD_GLOBAL','LOAD_NAME')}
        for constant in code.co_consts:
            if isinstance(constant,CodeType):names.update(globals_used(constant))
        return names
    found=set()
    for name in globals_used(function.__code__):
        target=function.__globals__.get(name)
        if not inspect.isfunction(target) or not target.__module__.startswith('sciona.atoms.'):continue
        if target in registered:found.add(registered[target])
        else:found.update(registered_call_dependencies(target,registered,visited))
    return found


def inventory():
    graph=build_evaluated_graph();digest,_,_=encode_execution_graph(graph)
    roots={n.matched_primitive for n in graph.nodes}
    for runtime in roots:importlib.import_module(runtime.rsplit('.',1)[0])
    registered={entry['impl']:name for name,entry in REGISTRY.items()}
    dependencies={};todo=list(roots)
    while todo:
        runtime=todo.pop()
        if runtime in dependencies:continue
        deps=registered_call_dependencies(REGISTRY[runtime]['impl'],registered)-{runtime}
        dependencies[runtime]=sorted(deps);todo.extend(deps-dependencies.keys())
    directory=ROOT.parent/'sciona-atoms-ml';records={}
    for spec in _parse_registered_atoms(repo=ProviderRepo(directory.name,directory),artifact_root=directory/'src/sciona/atoms'):
        runtime=spec.import_module+'.'+spec.source_symbol
        if runtime not in dependencies:continue
        if runtime in records or Path(inspect.getsourcefile(inspect.unwrap(REGISTRY[runtime]['impl']))).resolve()!=spec.file_path.resolve():
            raise ValueError('Ambiguous provider source: '+runtime)
        records[runtime]=dict(runtime_fqdn=runtime,catalog_fqdn=spec.fqdn,version_id=str(spec.version_id),
            content_hash=spec.content_hash,provider_sha256=hashlib.sha256(spec.file_path.read_bytes()).hexdigest(),
            directly_bound=runtime in roots,dependencies=dependencies[runtime])
    if set(records)!=set(dependencies):raise ValueError('Incomplete exact provider closure')
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        for record in records.values():
            for label,table,versions,key in [('canonical','artifacts','artifact_versions','artifact_id'),('legacy','atoms','atom_versions','atom_id')]:
                rows=db.execute(f'SELECT a.fqdn,a.status,a.is_publishable,v.version_id::text,v.content_hash,v.trust_tier FROM {table} a JOIN {versions} v USING({key}) WHERE a.fqdn=%s AND v.is_latest',
                    (record['catalog_fqdn'],)).fetchall()
                record[label+'_latest']=rows
                record[label+'_exact_approved']=len(rows)==1 and rows[0]['status']=='approved' and rows[0]['is_publishable'] and rows[0]['content_hash']==record['content_hash'] and rows[0]['trust_tier'] in (1,2,3)
    ordered=[records[name] for name in sorted(records)]
    return dict(passed=True,read_only=True,catalog_mutations=0,approved=False,graph_digest=digest,
        directly_bound_providers=len(roots),closure_providers=len(records),registered_dependency_links=sum(map(len,dependencies.values())),
        both_catalogs_exact_approved=sum(r['canonical_exact_approved'] and r['legacy_exact_approved'] for r in ordered),
        providers=ordered,limitations=['Static registered-call closure and current latest-version status only; no semantic review, intake or promotion.','Direct source helper and non-provider dependency qualification remains required alongside this inventory.'])


if __name__=='__main__':
    result=inventory()
    (ROOT/'docs/reviews/flavours_decomposed_provider_inventory.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('providers','limitations')}))
