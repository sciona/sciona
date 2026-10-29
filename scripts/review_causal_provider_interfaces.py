"""Freeze native provider ports and graph aliases for the exact causal closure."""
import hashlib
import importlib
import inspect
import json
from pathlib import Path
from typing import get_args,get_origin,get_type_hints,Literal,Union
from types import UnionType
import numpy as np
from sciona.causal_decomposed_features import build_explicit_causal_feature_graph
from scripts.inventory_causal_decomposed_providers import inventory,ROOT


def kind(annotation):
    origin=get_origin(annotation) or annotation
    if origin is np.ndarray:return 'numpy.ndarray'
    if origin in (list,dict,str,int,float,bool,object):return origin.__name__
    if origin in (Union,UnionType):
        remaining=[a for a in get_args(annotation) if a is not type(None)]
        if len(remaining)==1:return kind(remaining[0])
    if origin is Literal and all(isinstance(v,str) for v in get_args(annotation)):return 'str'
    raise ValueError('Explicit interface review needed: '+str(annotation))


def interfaces(function,nodes):
    signature=inspect.signature(function);hints=get_type_hints(function);ports=[]
    for ordinal,(name,parameter) in enumerate(signature.parameters.items()):
        if parameter.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD,inspect.Parameter.KEYWORD_ONLY):
            raise ValueError('Unsupported parameter kind')
        description=kind(hints[name])
        for node in nodes:
            matching=[p for p in node.inputs if p.name==name]
            if not matching and parameter.default is not inspect.Parameter.empty:
                continue
            if len(matching)!=1 or matching[0].type_desc!=description:
                raise ValueError('Graph/native input mismatch: '+name)
            if parameter.default is inspect.Parameter.empty and not matching[0].required:
                raise ValueError('Graph/native required flag mismatch: '+name)
        ports.append(dict(direction='input',name=name,ordinal=ordinal,type_desc=description,
            required=parameter.default is inspect.Parameter.empty,
            nullable_annotation=type(None) in get_args(hints[name]),
            default_value_repr='' if parameter.default is inspect.Parameter.empty else repr(parameter.default)))
    if any(not {p.name for p in n.inputs}<=set(signature.parameters) for n in nodes):raise ValueError('Unexpected graph port')
    annotation=hints['return']
    if nodes:
        outputs=[(p.name,p.type_desc) for p in nodes[0].outputs]
        if any([(p.name,p.type_desc) for p in n.outputs]!=outputs for n in nodes):raise ValueError('Inconsistent output aliases')
        if annotation is not tuple:
            annotations=get_args(annotation) if get_origin(annotation) is tuple else (annotation,)
            if [kind(a) for a in annotations]!=[t for _,t in outputs]:raise ValueError('Output annotation mismatch')
    else:
        outputs=[('result',kind(annotation))]
    for ordinal,(name,description) in enumerate(outputs):
        ports.append(dict(direction='output',name=name,ordinal=ordinal,type_desc=description,required=True,default_value_repr=''))
    return ports


def review():
    current=inventory();graph=build_explicit_causal_feature_graph()
    evidence_path=ROOT/'docs/reviews/causal_full_decomposition_source_parity.json'
    evidence=json.loads(evidence_path.read_text())
    if not evidence['passed'] or evidence['graph_hash']!=current['graph_digest'] or any(c['uncached_nodes']!=len(graph.nodes) for c in evidence['cases']):
        raise ValueError('Exact complete graph qualification required')
    for path,digest in evidence['implementation_hashes'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:raise ValueError('Implementation drift')
    for module,digest in evidence['provider_hashes'].items():
        if hashlib.sha256(Path(inspect.getfile(importlib.import_module(module))).read_bytes()).hexdigest()!=digest:
            raise ValueError('Provider source drift')
    grouped={}
    for node in graph.nodes:grouped.setdefault(node.matched_primitive,[]).append(node)
    records=[]
    for provider in current['providers']:
        module,name=provider['runtime_fqdn'].rsplit('.',1)
        function=getattr(importlib.import_module(module),name)
        if provider['provider_sha256'] not in evidence['provider_hashes'].values():raise ValueError('Unqualified source')
        calls=grouped.get(provider['runtime_fqdn'],[])
        records.append(dict(provider,interfaces=interfaces(function,calls),
            contract=inspect.getdoc(function),graph_nodes=[n.node_id for n in calls],
            omitted_native_defaults={n.node_id:{name:repr(p.default) for name,p in inspect.signature(function).parameters.items()
                if name not in {port.name for port in n.inputs} and p.default is not inspect.Parameter.empty} for n in calls}))
    return dict(passed=True,read_only=True,catalog_mutations=0,approved=False,graph_digest=current['graph_digest'],
        execution_evidence_sha256=hashlib.sha256(evidence_path.read_bytes()).hexdigest(),providers=records,
        provider_count=len(records),new_provider_count=sum(not r['canonical_latest'] and not r['legacy_latest'] for r in records),
        legacy_only_count=sum(not r['canonical_latest'] and bool(r['legacy_latest']) for r in records),
        limitations=['Native signatures and graph aliases reviewed; stored catalog ports still need comparison',
            'Indirect scalar output name result is a proposed alias, not a claim about existing catalog names',
            'Native docstrings describe contracts; semantic, bounds and transaction reviews remain required'])


if __name__=='__main__':
    report=review()
    (ROOT/'docs/reviews/causal_provider_interface_review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'providers','limitations'}},indent=2))
