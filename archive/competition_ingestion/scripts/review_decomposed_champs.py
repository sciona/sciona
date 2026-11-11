"""Automated review of the complete CHAMPS graph and explicit reusable mean."""
import ast
import hashlib
import importlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import subprocess
from typing import get_args, get_origin, get_type_hints

import numpy as np
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from sciona.decomposed_champs_graph import build_decomposed_champs_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from scripts.promote_connectomics_execution import boundary_ports, provider_ports

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = 'docs/reviews/decomposed_champs_graph_execution.json'
EVIDENCE_COMMIT = '36ab2080'
GENERIC = 'sciona.atoms.numerical.statistics.central_mean.central_rank_mean'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_training_body():
    from sciona import champs_runtime as reference
    from sciona import decomposed_champs_runtime as candidate
    original = ast.parse(inspect.getsource(reference.execute)).body[0]
    actual = ast.parse(inspect.getsource(candidate.train_models)).body[0]
    original.name = actual.name
    returned = original.body[-1].value
    position = next(i for i, key in enumerate(returned.keys) if key.value == 'predictions')
    require(ast.unparse(returned.values[position]) == 'blend_predictions(predictions, prepared.inference_types)',
            'Reference ensemble boundary changed')
    returned.keys[position:position + 1] = [ast.Constant('model_predictions'), ast.Constant('coupling_types')]
    returned.values[position:position + 1] = [ast.Name(id='predictions', ctx=ast.Load()),
        ast.Attribute(value=ast.Name(id='prepared', ctx=ast.Load()), attr='inference_types', ctx=ast.Load())]
    require(ast.dump(original, include_attributes=False) == ast.dump(actual, include_attributes=False),
            'Complete training body changed')
    for name in ('Prepared', 'MODEL_ORDER', 'fit_variant', 'load_for_prediction', 'population_batches',
                 'split_population', 'predict_batches', 'ChampsPreprocessor', 'ChampsSourceRuntime', 'EXECUTION_VERSION'):
        require(getattr(reference, name) is getattr(candidate, name), 'Training dependency changed: ' + name)


def interfaces(function, nodes):
    kinds = {np.ndarray: 'numpy.ndarray', object: 'object', int: 'int', dict: 'dict', tuple: 'tuple'}
    signature, hints = inspect.signature(function), get_type_hints(function)
    result = []
    for ordinal, (name, parameter) in enumerate(signature.parameters.items()):
        require(parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and
                parameter.default is inspect.Parameter.empty, 'Required keyword parameters expected')
        require(hints[name] in kinds, 'Unsupported input annotation')
        for node in nodes:
            require(list(signature.parameters) == [p.name for p in node.inputs], 'Input order differs')
            port = node.inputs[ordinal]
            require(port.type_desc == kinds[hints[name]] and port.required and
                    port.default_value_repr in (None, ''), 'Input contract differs')
        result.append(dict(direction='input', name=name, ordinal=ordinal,
            type_desc=kinds[hints[name]], required=True, default_value_repr='', constraints=nodes[0].inputs[ordinal].constraints))
    returns = get_args(hints['return']) if get_origin(hints['return']) is tuple else (hints['return'],)
    outputs = [(p.name, p.type_desc) for p in nodes[0].outputs]
    require(all([(p.name, p.type_desc) for p in node.outputs] == outputs for node in nodes), 'Output aliases differ')
    require([kinds[t] for t in returns] == [t for _, t in outputs], 'Return annotation differs')
    result.extend(dict(direction='output', name=n, ordinal=i, type_desc=t, required=True,
                       default_value_repr='', constraints=nodes[0].outputs[i].constraints)
                  for i, (n, t) in enumerate(outputs))
    return result


def validate_execution(full):
    require(full.get('format') == 'decomposed-champs-graph-execution.v1' and
            full.get('status') == 'passed' and full.get('approved') is False,
            'Passing unapproved execution evidence required')
    require(full['checks'] == dict(actual_runner_nodes=5, actual_runner_edges=7,
        source_ensemble_exact=True, full_model_lifecycles=13, synthetic_predictions=12,
        complete_checkpoint_reload=True, strict_json_output=True, graph_codec_roundtrip=True),
        'Complete graph lifecycle evidence required')
    require(full['resource_policy'] == dict(native_threads=1, torch_interop_threads=1,
        data_loader_workers=0, serial_models=True), 'Qualified resource policy differs')
    expected_paths = {str(p.relative_to(ROOT)) for p in (ROOT/'sciona').glob('champs_*.py')}
    expected_paths |= {str(p.relative_to(ROOT)) for p in (ROOT/'sciona').glob('decomposed_champs_*.py')}
    expected_paths |= {'scripts/validate_decomposed_champs_graph_execution.py', 'scripts/champs_synthetic.py',
                       'requirements/champs-chemistry.txt', 'sciona/visualizer/runner.py'}
    require(set(full['code_sha256']) == expected_paths, 'Complete implementation hash coverage required')
    for name, digest in full['code_sha256'].items():
        require(sha(ROOT/name) == digest, 'Implementation changed: ' + name)
    expected_packages = {'torch', 'numpy', 'pandas', 'rdkit', 'openbabel-wheel', 'networkx', 'scipy'}
    require(set(full['runtime']) == expected_packages, 'Complete runtime version coverage required')
    require(all(importlib.metadata.version(name) == version for name, version in full['runtime'].items()),
            'Executed runtime versions changed')


def review():
    raw = (ROOT/EVIDENCE).read_bytes()
    committed = subprocess.run(['git', 'show', EVIDENCE_COMMIT + ':' + EVIDENCE], cwd=ROOT,
                               capture_output=True, check=True).stdout
    require(raw == committed, 'Execution evidence differs from immutable commit')
    full = json.loads(raw)
    validate_execution(full)
    check_training_body()
    from sciona.champs_source_runtime import ChampsSourceRuntime
    source = ChampsSourceRuntime(Path('/private/tmp/sciona_champs_source'))
    require(source.commit == '4a42e18b5b88043fb40ec15289216a1d88789698', 'Pinned source differs')
    graph = build_decomposed_champs_graph()
    digest, nodes, edges = encode_execution_graph(graph)
    require(digest == full['serialized_graph_sha256'] and (len(nodes), len(edges)) == (5, 7),
            'Executed graph identity differs')
    wanted = {n.matched_primitive for n in graph.nodes}
    require(set(full['bound_provider_sha256']) == wanted, 'Exact bound provider coverage required')
    specs = {}
    for repo in ('sciona-atoms', 'sciona-atoms-ml'):
        directory = ROOT.parent/repo
        for spec in _parse_registered_atoms(repo=ProviderRepo(repo, directory), artifact_root=directory/'src/sciona/atoms'):
            runtime = spec.import_module + '.' + spec.source_symbol
            if runtime in wanted:
                require(runtime not in specs, 'Ambiguous registered provider')
                specs[runtime] = spec
    require(set(specs) == wanted and len(specs) == 5, 'Five exact provider registrations required')
    records, ports = [], {}
    for runtime, spec in sorted(specs.items()):
        function = getattr(importlib.import_module(spec.import_module), spec.source_symbol)
        require(Path(inspect.getfile(function)).resolve() == spec.file_path.resolve(), 'Provider origin differs')
        require(sha(spec.file_path) == full['bound_provider_sha256'][runtime], 'Provider implementation changed')
        bound = [n for n in graph.nodes if n.matched_primitive == runtime]
        contract = interfaces(function, bound)
        for node in bound:
            ports[node.node_id] = provider_ports(node, function)
        records.append(dict(runtime_fqdn=runtime, catalog_fqdn=spec.fqdn, version_id=str(spec.version_id),
            content_hash=spec.content_hash, provider_sha256=sha(spec.file_path), interfaces=contract,
            signature=str(inspect.signature(function)), bound_nodes=[n.node_id for n in bound],
            classification='generic_operation' if runtime == GENERIC else 'source_recipe_adapter'))
    boundary = boundary_ports(graph, ports)
    require([p.name for p in boundary['input']] == ['payload'] and
            [p.name for p in boundary['output']] == ['result'], 'Complete root contract differs')
    historic = json.loads((ROOT/'docs/reviews/competition_champs_semantic_review.json').read_text())
    return dict(format='champs-decomposed-review.v1', review_source='automated', proposed_tier=3,
        approved=False, catalog_mutations=0, graph_digest=digest, graph_nodes=5, graph_edges=7,
        providers=records, root_inputs=['payload'], root_outputs=['result'], complete_training_ast_equivalent=True,
        source_stage_mapping=historic['intake_stage_mapping'], package_versions=full['runtime'],
        execution_evidence_commit=EVIDENCE_COMMIT,
        evidence_sha256={EVIDENCE: sha(ROOT/EVIDENCE),
                        'docs/reviews/competition_champs_source_pins.json': sha(ROOT/'docs/reviews/competition_champs_source_pins.json'),
                        'docs/reviews/competition_champs_semantic_review.json': sha(ROOT/'docs/reviews/competition_champs_semantic_review.json')},
        reviewer_sha256=sha(Path(__file__)),
        limitations=historic['limitations'] + [
            'The new graph executes all thirteen full configured models for one explicitly requested synthetic SGD epoch each; historical training-duration and accuracy parity remain unproven.',
            'Current execution uses NumPy 2.2.6, recorded in the new immutable report; it does not overwrite the earlier NumPy 2.4.2 qualification.',
            'The complete training body is unchanged apart from moving final ensemble selection/reduction into explicit graph boundaries.',
            'Central rank mean is domain independent. Molecular identity, model masks and retained count stay in recipe policy.',
            'Dictionary shape and identity checks do not independently authenticate training provenance.',
            'Provider eligibility and corrected-original publication require separate transactional qualification.'])


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/champs_decomposed_semantic_review.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('approved', 'catalog_mutations', 'graph_nodes', 'graph_edges')}))
