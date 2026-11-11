"""Conservative source-reference audit beside the DSB runtime dependency trace.

Traverse referenced local functions/classes, including branches and deferred
methods. Registered providers are boundaries. This does not resolve arbitrary
object dispatch or prove that a referenced branch executes.
"""
import ast
from functools import lru_cache
import importlib.util
import json
from pathlib import Path

from scripts.inventory_dsb_decomposed_providers import inventory, ROOT, sha
from scripts.plan_dsb_provider_dependencies import plan


@lru_cache(None)
def module_source(module):
    spec=importlib.util.find_spec(module)
    if spec is None or spec.origin is None:
        raise ValueError('Unresolved local module: '+module)
    path=Path(spec.origin)
    tree=ast.parse(path.read_text())
    definitions={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
    for node in tree.body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name):definitions[target.id]=node
        elif isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name):
            definitions[node.target.id]=node
    imports={}
    for node in tree.body:
        if isinstance(node,ast.ImportFrom):
            imports.update(import_bindings(node,module))
        elif isinstance(node,ast.Import):
            for alias in node.names:
                imports[alias.asname or alias.name.split('.')[0]]=alias.name
    return path,definitions,imports


def import_bindings(node,module):
    if node.level:
        parent=module.rsplit('.',node.level)[0]
        base=parent+('.'+node.module if node.module else '')
    else:base=node.module or ''
    if any(a.name=='*' for a in node.names):
        raise ValueError('Wildcard import requires explicit review')
    return {a.asname or a.name:base+'.'+a.name for a in node.names}


def conditional_exclusion():
    _,validation,_=module_source('sciona.dsb_decomposed_validation')
    calls=[n for n in ast.walk(validation['validate_detector']) if isinstance(n,ast.Call)
           and isinstance(n.func,ast.Name) and n.func.id=='detector_loss']
    if len(calls)!=1 or not any(k.arg=='training' and isinstance(k.value,ast.Constant)
                               and k.value.value is False for k in calls[0].keywords):
        raise ValueError('Validation training=False exclusion changed')
    _,losses,_=module_source('sciona.dsb_decomposed_losses')
    loss=losses['detector_loss']
    selectors=[n for n in ast.walk(loss) if isinstance(n,ast.Call)
               and isinstance(n.func,ast.Name) and n.func.id=='highest_scoring_indices']
    guards=[n for n in ast.walk(loss) if isinstance(n,ast.If) and isinstance(n.test,ast.BoolOp)
            and isinstance(n.test.op,ast.And) and all(isinstance(v,ast.Name) for v in n.test.values)
            and [v.id for v in n.test.values]==['num_hard','training']]
    if len(selectors)!=1 or len(guards)!=1 or not any(selectors[0] in list(ast.walk(n)) for n in guards[0].body):
        raise ValueError('Training-only ranker guard changed')
    return dict(caller='sciona.atoms.dl.detection.dsb_lifecycle_steps.validate_detector',
                callee='sciona.atoms.dl.loss.ranked_selection.highest_scoring_indices',
                disposition='excluded_for_this_caller',
                reason='Validation passes training=False; the sole ranking call is guarded by num_hard and training.',
                regression_test='tests/test_dsb_ranked_loss.py::test_registered_detector_validation_never_runs_training_ranker')


def audit():
    current=inventory(offline=True)
    registered={r['runtime_fqdn'] for r in current['providers']}
    prior=plan()
    planned={(r['caller'],r['callee']) for r in prior['dependencies']}
    sources={}; edges=set(); reviews=[]
    for root in sorted(registered):
        visited=set(); pending=[root]; external=set(); unresolved=set()
        while pending:
            symbol=pending.pop()
            if symbol in visited:continue
            visited.add(symbol)
            if symbol in registered and symbol!=root:
                edges.add((root,symbol));continue
            module,name=symbol.rsplit('.',1)
            if not (module.startswith('sciona.dsb_') or module.startswith('sciona.atoms.')):
                external.add(module);continue
            path,definitions,imports=module_source(module)
            node=definitions.get(name)
            if node is None:
                if name in imports:pending.append(imports[name])
                else:unresolved.add(symbol)
                continue
            sources[str(path.relative_to(ROOT.parent))]=sha(path)
            bindings=dict(imports)
            # Local imports are scoped to this function/class, so sibling
            # provider implementation imports do not create false dependencies.
            for part in ast.walk(node):
                if isinstance(part,ast.ImportFrom):bindings.update(import_bindings(part,module))
                elif isinstance(part,ast.Import):
                    for alias in part.names:
                        bindings[alias.asname or alias.name.split('.')[0]]=alias.name
            for part in ast.walk(node):
                if not isinstance(part,ast.Name) or not isinstance(part.ctx,ast.Load):continue
                if part.id in bindings:
                    target=bindings[part.id]
                    # Registration/witness machinery is not a runtime atom edge.
                    if target.startswith('sciona.ghost.'):
                        continue
                    if target.startswith(('sciona.dsb_','sciona.atoms.')):pending.append(target)
                    else:external.add(target)
                elif part.id in definitions:pending.append(module+'.'+part.id)
        reviews.append(dict(provider=root,local_symbols=sorted(visited),
                            external_references=sorted(external),unresolved_local_symbols=sorted(unresolved)))
    exclusion=conditional_exclusion()
    excluded={(exclusion['caller'],exclusion['callee'])}
    if edges-planned != excluded:
        raise ValueError('Additional static candidates need review')
    if any(r['unresolved_local_symbols'] for r in reviews):
        raise ValueError('Unresolved local symbols need review')
    return dict(approved=False,catalog_mutations=0,graph_digest=current['graph_digest'],
        providers=len(registered),source_files=len(sources),source_sha256=sources,
        source_reference_edges=[dict(caller=a,callee=b) for a,b in sorted(edges)],
        additional_candidates=[dict(caller=a,callee=b) for a,b in sorted(edges-planned)],
        candidate_resolutions=[exclusion],unresolved_additional_candidates=0,
        runtime_or_deferred_only=[dict(caller=a,callee=b) for a,b in sorted(planned-edges)],
        reviews=reviews,auditor_sha256=sha(Path(__file__)),complete_static_closure=False,
        limitations=['Conservative symbol-reference traversal; conditional references may be unreachable for a specific caller.',
            'Caller-supplied model/factory/callback dispatch remains governed by qualified graph bindings and runtime trace.',
            'External references are names only, not library version qualification.',
            'Additional candidates and unresolved symbols require review before dependency publication.'])


if __name__=='__main__':
    report=audit()
    (ROOT/'docs/reviews/dsb_static_dependency_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('providers','source_files','additional_candidates','runtime_or_deferred_only')},indent=2))
