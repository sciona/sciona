#!/usr/bin/env python3
"""Inventory exact parsed provider identities and static Andriy dependencies."""
import argparse
import hashlib
import importlib
import inspect
import json
from pathlib import Path
from types import CodeType
from typing import get_args, get_origin, get_type_hints
import numpy as np
from sciona.andriy_execution import build_andriy_execution_graph
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from sciona.ghost.registry import REGISTRY


def provider_interfaces(function, graph_outputs=None):
    """Build ordered catalog ports; reject unsupported or ambiguous annotations."""
    output_names = {
        'andriy_documented_clip_features': ['features', 'valid_rows'],
        'andriy_assemble_clip_features': ['features', 'valid_rows'],
        'andriy_joint_normalization': ['normalized_training', 'normalized_prediction'],
        'andriy_documented_population': ['training_features', 'training_labels', 'prediction_features', 'valid_prediction_rows'],
        'andriy_population_inputs': ['training_features', 'training_labels', 'prediction_features', 'valid_prediction_rows'],
        'andriy_csp_training_classes': ['sequence6_samples', 'sequence1_samples'],
        'andriy_documented_preprocess_clip': ['signal256', 'signal128'],
    }
    def kind(annotation):
        origin = get_origin(annotation) or annotation
        if origin is np.ndarray:
            return 'numpy.ndarray'
        if origin in (list, int):
            return origin.__name__
        raise ValueError('Unsupported provider port annotation: '+str(annotation))
    signature = inspect.signature(function)
    hints = get_type_hints(function)
    ports = []
    for ordinal, (name, parameter) in enumerate(signature.parameters.items()):
        if parameter.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            raise ValueError('Unsupported provider parameter kind')
        ports.append(dict(direction='input', name=name, ordinal=ordinal,
            type_desc=kind(hints.get(name)), required=parameter.default is inspect.Parameter.empty,
            default_value_repr='' if parameter.default is inspect.Parameter.empty else repr(parameter.default),
            constraints='See exact provider description for numerical and shape requirements.'))
    annotation = hints.get('return')
    outputs = get_args(annotation) if get_origin(annotation) is tuple else (annotation,)
    names = ([port.name for port in graph_outputs] if graph_outputs is not None
             else output_names.get(function.__name__, ['result']))
    if len(names) != len(outputs):
        raise ValueError('Explicit tuple output names required')
    for ordinal, (name, annotation) in enumerate(zip(names, outputs)):
        type_desc = kind(annotation)
        if graph_outputs is not None and graph_outputs[ordinal].type_desc != type_desc:
            raise ValueError('Graph output type differs from provider annotation')
        ports.append(dict(direction='output', name=name, ordinal=ordinal, type_desc=type_desc,
            required=True, default_value_repr='', constraints='Tuple ordinal and shape follow exact provider contract.'))
    return ports


def global_names(code):
    names = set(code.co_names)
    for value in code.co_consts:
        if isinstance(value, CodeType):
            names.update(global_names(value))
    return names


def registered_dependencies(function, registered, visited=None):
    """Follow direct global function references, including local helper bodies."""
    visited = set() if visited is None else visited
    if function in visited:
        return set()
    visited.add(function)
    found = set()
    for name in global_names(function.__code__):
        target = function.__globals__.get(name)
        if not inspect.isfunction(target) or not target.__module__.startswith('sciona.atoms.'):
            continue
        if target in registered:
            found.add(registered[target])
        else:
            found.update(registered_dependencies(target, registered, visited))
    return found


def inventory(root):
    graph = build_andriy_execution_graph()
    roots = {node.matched_primitive for node in graph.nodes}
    graph_outputs = {node.matched_primitive: node.outputs for node in graph.nodes}
    for runtime in roots:
        importlib.import_module(runtime.rsplit('.', 1)[0])
    registered = {entry['impl']: name for name, entry in REGISTRY.items()}
    dependencies = {}
    todo = list(roots)
    while todo:
        runtime = todo.pop()
        if runtime in dependencies:
            continue
        function = REGISTRY[runtime]['impl']
        deps = registered_dependencies(function, registered) - {runtime}
        dependencies[runtime] = sorted(deps)
        todo.extend(deps - dependencies.keys())
    parsed = {}
    for name in ['sciona-atoms-signal', 'sciona-atoms-ml']:
        directory = root.parent/name
        for spec in _parse_registered_atoms(repo=ProviderRepo(name, directory), artifact_root=directory/'src/sciona/atoms'):
            runtime = spec.import_module+'.'+spec.source_symbol
            if runtime in dependencies:
                if runtime in parsed:
                    raise ValueError('Ambiguous parsed provider identity: '+runtime)
                parsed[runtime] = spec
    if set(parsed) != set(dependencies):
        raise ValueError('Missing parsed identities in runtime closure')
    records = []
    for runtime, spec in sorted(parsed.items()):
        function = REGISTRY[runtime]['impl']
        if function.__module__+'.'+function.__name__ != runtime:
            raise ValueError('Runtime identity mismatch')
        if Path(inspect.getsourcefile(function)).resolve() != spec.file_path.resolve():
            raise ValueError('Parsed and imported source locations differ')
        records.append(dict(runtime_fqdn=runtime, catalog_fqdn=spec.fqdn,
            version_id=str(spec.version_id), content_hash=spec.content_hash,
            provider_sha256=hashlib.sha256(spec.file_path.read_bytes()).hexdigest(),
            dependencies=dependencies[runtime],
            interfaces=provider_interfaces(function, graph_outputs.get(runtime)),
            input_parameters=[dict(name=name, required=parameter.default is inspect.Parameter.empty)
                              for name, parameter in inspect.signature(function).parameters.items()]))
    return dict(read_only=True, approval_applied=False, graph_roots=sorted(roots),
        providers=records, provider_count=len(records),
        dependency_count=sum(len(value) for value in dependencies.values()),
        limitations=['Static global-function dependency inventory, including nested code objects and local helpers.',
            'Does not infer dynamic attribute calls, external libraries or embedded R dependencies; those require separate review.',
            'Exact identities are intake candidates, not approved versions or verified catalog bindings.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = inventory(Path(__file__).resolve().parents[1])
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in ['read_only', 'provider_count', 'dependency_count']}))
