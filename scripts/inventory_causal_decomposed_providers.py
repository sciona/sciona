"""Read-only exact provider inventory, including module calls and callable tables."""
import ast
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import textwrap
from types import ModuleType

import psycopg
from psycopg.rows import dict_row
from dotenv import dotenv_values
from sciona.causal_decomposed_features import build_explicit_causal_feature_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.ghost.registry import REGISTRY
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms

ROOT=Path(__file__).resolve().parents[1]


def registered_dependencies(function,registered,visited=None):
    """Resolve project calls through globals, module attributes and static tables.

    Dynamic keys conservatively include every callable in the referenced table.
    Local data attributes and arbitrary properties are never evaluated.
    """
    visited=set() if visited is None else visited
    function=inspect.unwrap(function)
    if function in visited:return set()
    visited.add(function)
    def resolve(expr):
        if isinstance(expr,ast.Name):return [function.__globals__.get(expr.id)]
        if isinstance(expr,ast.Attribute):
            return [vars(value).get(expr.attr) for value in resolve(expr.value) if isinstance(value,ModuleType)]
        if isinstance(expr,ast.Subscript):
            result=[]
            for value in resolve(expr.value):
                if isinstance(value,dict):result.extend(value.values())
                elif isinstance(value,(list,tuple)):result.extend(value)
            return result
        return []
    found=set()
    tree=ast.parse(textwrap.dedent(inspect.getsource(function)))
    for call in (node for node in ast.walk(tree) if isinstance(node,ast.Call)):
        for target in resolve(call.func):
            if not inspect.isfunction(target) or not target.__module__.startswith('sciona.atoms.'):continue
            unwrapped=inspect.unwrap(target)
            if unwrapped in registered:found.add(registered[unwrapped])
            else:found.update(registered_dependencies(target,registered,visited))
    return found


def provider_closure():
    graph=build_explicit_causal_feature_graph()
    roots={n.matched_primitive for n in graph.nodes}
    for runtime in roots:importlib.import_module(runtime.rsplit('.',1)[0])
    registered={inspect.unwrap(entry['impl']):name for name,entry in REGISTRY.items()}
    dependencies={};todo=list(roots)
    while todo:
        runtime=todo.pop()
        if runtime in dependencies:continue
        found=registered_dependencies(REGISTRY[runtime]['impl'],registered)-{runtime}
        dependencies[runtime]=sorted(found)
        todo.extend(found-dependencies.keys())
    return graph,roots,dependencies


def inventory():
    graph,roots,dependencies=provider_closure()
    digest,_,_=encode_execution_graph(graph)
    records={}
    for repo_name in ['sciona-atoms','sciona-atoms-ml']:
        directory=ROOT.parent/repo_name
        for spec in _parse_registered_atoms(repo=ProviderRepo(repo_name,directory),artifact_root=directory/'src/sciona/atoms'):
            runtime=spec.import_module+'.'+spec.source_symbol
            if runtime not in dependencies:continue
            if runtime in records or Path(inspect.getsourcefile(inspect.unwrap(REGISTRY[runtime]['impl']))).resolve()!=spec.file_path.resolve():
                raise ValueError('Ambiguous exact source: '+runtime)
            records[runtime]=dict(runtime_fqdn=runtime,catalog_fqdn=spec.fqdn,version_id=str(spec.version_id),
                content_hash=spec.content_hash,provider_sha256=hashlib.sha256(spec.file_path.read_bytes()).hexdigest(),
                directly_bound=runtime in roots,dependencies=dependencies[runtime])
    if set(records)!=set(dependencies):raise ValueError('Incomplete source closure: '+str(sorted(set(dependencies)-set(records))))
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        for record in records.values():
            for label,table,versions,key in [('canonical','artifacts','artifact_versions','artifact_id'),('legacy','atoms','atom_versions','atom_id')]:
                rows=db.execute(f'SELECT a.fqdn,a.status,a.is_publishable,v.version_id::text,v.content_hash,v.trust_tier FROM {table} a JOIN {versions} v USING({key}) WHERE a.fqdn=%s AND v.is_latest',(record['catalog_fqdn'],)).fetchall()
                record[label+'_latest']=rows
                record[label+'_exact_approved']=len(rows)==1 and rows[0]['status']=='approved' and rows[0]['is_publishable'] and rows[0]['content_hash']==record['content_hash'] and rows[0]['trust_tier'] in (1,2,3)
    ordered=[records[name] for name in sorted(records)]
    return dict(read_only=True,catalog_mutations=0,graph_digest=digest,directly_bound_providers=len(roots),
        closure_providers=len(records),registered_dependency_links=sum(map(len,dependencies.values())),
        both_catalogs_exact_approved=sum(r['canonical_exact_approved'] and r['legacy_exact_approved'] for r in ordered),
        providers=ordered,limitations=['Inventory is not semantic approval or publication evidence',
            'Static callable tables are conservatively included; source helpers and external-library semantics require review',
            'Latest-version hash and publication flags do not alone certify catalog interfaces or dependency rows'])


if __name__=='__main__':
    report=inventory()
    (ROOT/'docs/reviews/causal_decomposed_provider_inventory.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'providers','limitations'}},indent=2))
