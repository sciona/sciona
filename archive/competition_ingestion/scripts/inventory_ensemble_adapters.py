#!/usr/bin/env python3
"""Inventory exact new ensemble adapter versions and their graph interfaces."""
import argparse
import ast
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import textwrap
from typing import get_args, get_origin, get_type_hints
import numpy as np
from sciona.ensemble_execution import build_ensemble_execution_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms


def interfaces(function, node):
    signature = inspect.signature(function)
    hints = get_type_hints(function)
    def kind(annotation):
        origin = get_origin(annotation) or annotation
        if origin is np.ndarray:
            return 'numpy.ndarray'
        if origin in (list, int):
            return origin.__name__
        raise ValueError('Unsupported adapter annotation')
    if [p.name for p in node.inputs] != list(signature.parameters):
        raise ValueError('Adapter parameter names/order differ from graph')
    result = []
    for ordinal, port in enumerate(node.inputs):
        parameter = signature.parameters[port.name]
        if parameter.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD or kind(hints.get(port.name)) != port.type_desc:
            raise ValueError('Adapter input contract differs')
        result.append(dict(direction='input', name=port.name, ordinal=ordinal, type_desc=port.type_desc,
            required=parameter.default is inspect.Parameter.empty,
            default_value_repr='' if parameter.default is inspect.Parameter.empty else repr(parameter.default),
            constraints='See exact provider contract for population, identity and numerical requirements.'))
    annotation = hints.get('return')
    if annotation is tuple:
        # The family boundary deliberately has a heterogeneous tuple annotation.
        # Its sole direct literal return must preserve its ten input references.
        definition = next(n for n in ast.parse(textwrap.dedent(inspect.getsource(function))).body if isinstance(n, ast.FunctionDef))
        returns = [n for n in ast.walk(definition) if isinstance(n, ast.Return)]
        if len(returns) != 1 or not isinstance(returns[0].value, ast.Tuple):
            raise ValueError('Unannotated tuple requires a single explicit literal return')
        values = returns[0].value.elts
        if any(not isinstance(v, ast.Name) for v in values) or [v.id for v in values] != list(signature.parameters):
            raise ValueError('Family boundary must return unchanged input references in order')
        types = [kind(hints[v.id]) for v in values]
    else:
        types = [kind(value) for value in get_args(annotation)] if get_origin(annotation) is tuple else [kind(annotation)]
    if types != [port.type_desc for port in node.outputs]:
        raise ValueError('Adapter output arity/types differ from graph')
    for ordinal, port in enumerate(node.outputs):
        result.append(dict(direction='output', name=port.name, ordinal=ordinal, type_desc=port.type_desc,
            required=True, default_value_repr='', constraints='Exact tuple ordinal and identity order follow provider contract.'))
    return result


def inventory(root):
    graph = build_ensemble_execution_graph()
    grouped = {}
    for node in graph.nodes:
        if '.ensemble_' in node.matched_primitive:
            grouped.setdefault(node.matched_primitive, []).append(node)
    if len(grouped) != 7:
        raise ValueError('Expected seven new ensemble adapters')
    directory = root.parent/'sciona-atoms-signal'
    parsed = {}
    for spec in _parse_registered_atoms(repo=ProviderRepo(directory.name, directory), artifact_root=directory/'src/sciona/atoms'):
        runtime = spec.import_module+'.'+spec.source_symbol
        if runtime in grouped:
            if runtime in parsed:
                raise ValueError('Duplicate parsed runtime identity')
            parsed[runtime] = spec
    if set(parsed) != set(grouped):
        raise ValueError('Missing parsed adapter identities')
    providers = []
    for runtime, spec in sorted(parsed.items()):
        function = getattr(importlib.import_module(spec.import_module), spec.source_symbol)
        if Path(inspect.getsourcefile(function)).resolve() != spec.file_path.resolve():
            raise ValueError('Parsed/imported source differs')
        ports = interfaces(function, grouped[runtime][0])
        if any(interfaces(function, node) != ports for node in grouped[runtime]):
            raise ValueError('Repeated adapter interfaces differ across nodes')
        providers.append(dict(runtime_fqdn=runtime, catalog_fqdn=spec.fqdn,
            version_id=str(spec.version_id), content_hash=spec.content_hash,
            provider_sha256=hashlib.sha256(spec.file_path.read_bytes()).hexdigest(), interfaces=ports,
            graph_nodes=[node.node_id for node in grouped[runtime]]))
    digest, _, _ = encode_execution_graph(graph)
    return dict(read_only=True, approval_applied=False, graph_digest=digest, providers=providers,
        provider_count=len(providers), graph_occurrences=sum(len(p['graph_nodes']) for p in providers))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = inventory(Path(__file__).resolve().parents[1])
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in ['provider_count', 'graph_occurrences', 'approval_applied']}))
