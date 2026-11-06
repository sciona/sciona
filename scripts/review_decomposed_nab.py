"""Review complete NAB decomposition, pinned scoring and reusable threshold."""
import ast
import hashlib
import importlib
import importlib.metadata as metadata
import inspect
import json
from pathlib import Path
import subprocess
from typing import get_args, get_origin, get_type_hints
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from sciona.decomposed_nab_graph import build_decomposed_nab_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from scripts.promote_connectomics_execution import boundary_ports, provider_ports

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = 'docs/reviews/decomposed_nab_graph_execution.json'
EVIDENCE_COMMIT = 'f5eace31'
GENERIC = 'sciona.atoms.numerical.statistics.prior_threshold.prior_location_scale_threshold'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_forest_loop():
    from sciona import nab_streaming as reference
    from sciona import decomposed_nab_runtime as candidate
    original = ast.parse(inspect.getsource(reference.detect)).body[0]
    actual = ast.parse(inspect.getsource(candidate.forest_scores)).body[0]
    original_loop = next(n for n in original.body if isinstance(n, ast.For) and ast.unparse(n.target) == 'i')
    actual_loop = next(n for n in actual.body if isinstance(n, ast.For) and ast.unparse(n.target) == 'i')
    # Remove only the former threshold reduction and its score-history append.
    removed = []
    retained = []
    for statement in original_loop.body:
        text = ast.unparse(statement)
        if (text.startswith('history = ') or text.startswith('if len(history)')
                or text == 'prior_scores.append(score)'):
            removed.append(text)
        else:
            retained.append(statement)
    require(len(removed) == 3, 'Reference threshold boundary changed')
    original_loop.body = retained
    require(ast.dump(original_loop, include_attributes=False) == ast.dump(actual_loop, include_attributes=False),
            'Complete causal forest loop changed')
    require(reference.IsolationForest is candidate.IsolationForest and reference.np is candidate.np,
            'Forest dependency changed')


def validate_execution(full):
    require(full.get('format') == 'decomposed-nab-graph-execution.v1'
            and full.get('status') == 'passed' and full.get('approved') is False
            and full.get('catalog_mutations') == 0 and full.get('synthetic_only') is True,
            'Passing synthetic execution evidence required')
    require(full['checks'] == dict(actual_runner_nodes=6, actual_runner_edges=13,
        models_fitted=90, stream_points=100, exact_reconstruction_result=True,
        strict_json_output=True, graph_codec_roundtrip=True), 'Complete execution coverage required')
    require(full['resource_policy'] == dict(native_threads=1, forest_jobs=1, serial_models=True),
            'Qualified resource policy differs')
    expected = {str(p.relative_to(ROOT)) for pattern in ('nab_*.py','decomposed_nab_*.py')
                for p in (ROOT/'sciona').glob(pattern)}
    expected |= {'scripts/nab_synthetic.py','scripts/validate_decomposed_nab_graph_execution.py',
                 'sciona/visualizer/runner.py','sciona/services/execution_graph_codec.py'}
    require(set(full['code_sha256']) == expected, 'Complete implementation hash coverage required')
    for name, digest in full['code_sha256'].items():
        require(sha(ROOT/name) == digest, 'Implementation changed: '+name)
    require(set(full['runtime']) == {'numpy','scipy','scikit-learn','joblib','threadpoolctl'},
            'Complete runtime versions required')
    require(all(metadata.version(k) == v for k,v in full['runtime'].items()), 'Runtime version changed')


def review():
    raw = (ROOT/EVIDENCE).read_bytes()
    committed = subprocess.check_output(['git','show',EVIDENCE_COMMIT+':'+EVIDENCE],cwd=ROOT)
    require(raw == committed, 'Immutable execution evidence differs')
    full = json.loads(raw)
    validate_execution(full)
    check_forest_loop()
    pins_path = ROOT/'docs/reviews/competition_nab_source_pins.json'
    pins = json.loads(pins_path.read_text())
    require(pins['repository'] == 'numenta/NAB' and pins['commit'] == 'ea702d75cc2258d9d7dd35ca8e5e2539d71f3140',
            'Pinned benchmark identity differs')
    for item in pins['pins']:
        relative = Path(item['software_path'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe software path')
        require(sha(Path('/private/tmp/sciona_nab_source')/relative) == item['sha256'], 'Pinned software differs')
    scoring_path = ROOT/'docs/reviews/decomposed_nab_scoring.json'
    scoring = json.loads(scoring_path.read_text())
    require(scoring['status'] == 'passed' and scoring['synthetic_only'] is True
        and scoring['source_commit'] == pins['commit'] and scoring['source_comparison_cases'] == 432
        and scoring['max_absolute_score_error'] < 1e-12 and scoring['point_confusion_counts_exact'] is True
        and scoring['corrected_threshold_selection_cases'] > 0, 'Complete pinned scoring evidence required')
    require(scoring['source_sha256'] == next(p['sha256'] for p in pins['pins'] if p['software_path']=='nab/sweeper.py'),
            'Scoring source hash differs')
    require(set(scoring['hashes']) == {'sciona/nab_scoring.py','scripts/validate_decomposed_nab_scoring.py','tests/test_nab_runtime.py'},
            'Scoring implementation coverage differs')
    require(all(sha(ROOT/n) == h for n,h in scoring['hashes'].items()), 'Scoring implementation changed')
    require(set(scoring['runtime']) == {'numpy','pandas'} and all(metadata.version(k)==v for k,v in scoring['runtime'].items()),
            'Scoring environment changed')
    from scripts.audit_decomposed_nab_packages import audit
    package_path = ROOT/'docs/reviews/decomposed_nab_package_files.json'
    require(audit() == json.loads(package_path.read_text()), 'Package evidence changed')
    graph = build_decomposed_nab_graph()
    digest, nodes, edges = encode_execution_graph(graph)
    require(digest == full['serialized_graph_sha256'] and (len(nodes),len(edges)) == (6,13), 'Executed graph differs')
    wanted = {n.matched_primitive for n in graph.nodes}
    require(set(full['bound_provider_sha256']) == wanted, 'Exact provider hash coverage required')
    specs = {}
    for repo in ('sciona-atoms','sciona-atoms-ml'):
        directory = ROOT.parent/repo
        for spec in _parse_registered_atoms(repo=ProviderRepo(repo,directory),artifact_root=directory/'src/sciona/atoms'):
            runtime = spec.import_module+'.'+spec.source_symbol
            if runtime in wanted:
                require(runtime not in specs, 'Ambiguous registration')
                specs[runtime] = spec
    require(set(specs) == wanted and len(specs) == 6, 'Six exact registered providers required')
    records, ports = [], {}
    for runtime,spec in sorted(specs.items()):
        function = getattr(importlib.import_module(spec.import_module),spec.source_symbol)
        require(Path(inspect.getfile(function)).resolve() == spec.file_path.resolve(), 'Provider origin differs')
        require(sha(spec.file_path) == full['bound_provider_sha256'][runtime], 'Provider source changed')
        bound = [n for n in graph.nodes if n.matched_primitive == runtime]
        contract = interfaces(function,bound)
        for node in bound:
            ports[node.node_id] = provider_ports(node,function)
        records.append(dict(runtime_fqdn=runtime,catalog_fqdn=spec.fqdn,version_id=str(spec.version_id),
            content_hash=spec.content_hash,provider_sha256=sha(spec.file_path),interfaces=contract,
            signature=str(inspect.signature(function)),bound_nodes=[n.node_id for n in bound],
            classification='generic_operation' if runtime==GENERIC else 'source_recipe_adapter'))
    boundary = boundary_ports(graph,ports)
    require([p.name for p in boundary['input']]==['payload'] and [p.name for p in boundary['output']]==['result'],
            'Root invocation contract differs')
    historic_path = ROOT/'docs/reviews/competition_nab_semantic_review.json'
    historic = json.loads(historic_path.read_text())
    return dict(format='nab-decomposed-review.v1',review_source='automated',proposed_tier=3,
        approved=False,catalog_mutations=0,graph_digest=digest,graph_nodes=6,graph_edges=13,
        providers=records,root_inputs=['payload'],root_outputs=['result'],complete_forest_loop_ast_equivalent=True,
        source_stage_mapping=historic['intake_stage_mapping'],package_versions=full['runtime'],
        execution_evidence_commit=EVIDENCE_COMMIT,
        evidence_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/EVIDENCE,pins_path,scoring_path,package_path,historic_path)},
        reviewer_sha256=sha(Path(__file__)),
        limitations=historic['limitations']+[
            'Current synthetic graph uses NumPy 2.2.6 and matches the full existing generic reconstruction exactly; historical reports remain unchanged.',
            'Forest fitting loop is AST-identical after removing only threshold reduction and its history append into explicit graph stages.',
            'Rolling prior mean-plus-scale threshold accepts arbitrary finite multipliers; NAB three-sigma and strict comparison remain recipe policy.',
            'Shape checks do not authenticate the provenance of externally supplied intermediate values.',
            'Exact provider eligibility and corrected-original publication remain pending separate transaction gates.'])


# Interface validation is defined below before the CLI entry point.

def interfaces(function, nodes):
    kinds = {list: 'list', float: 'float', object: 'object', int: 'int', dict: 'dict', tuple: 'tuple'}
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


if __name__ == '__main__':
    report = review()
    (ROOT/'docs/reviews/nab_decomposed_semantic_review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('approved','graph_nodes','graph_edges')}))
