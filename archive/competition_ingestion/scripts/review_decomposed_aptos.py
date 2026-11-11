"""Qualify the complete five-node APTOS graph and its reusable cut operation."""
import ast
import importlib
import inspect
import json
from pathlib import Path

from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from sciona.decomposed_aptos_graph import build_decomposed_aptos_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from scripts.review_decomposed_aptos_baseline import review as baseline_review, ROOT, sha, require, check_hashes
from scripts.promote_connectomics_execution import provider_ports, boundary_ports

GENERIC = 'sciona.atoms.numerical.statistics.cut_indices.ordered_cut_indices'


def check_training_body():
    from sciona import aptos_pipeline as reference
    from sciona import decomposed_aptos_pipeline as candidate
    original = ast.parse(inspect.getsource(reference.run_reference)).body[0]
    actual = ast.parse(inspect.getsource(candidate.train_scores)).body[0]
    original.name = actual.name
    original.body.pop(0)
    actual.body.pop(0)
    context = next(n for n in original.body if isinstance(n, ast.With))
    removed = context.body.pop()
    require(ast.unparse(removed) == 'ordinals = classify(scores, tie_policy=tie_policy)',
            'Reference classification boundary changed')
    returned = original.body[-1].value
    position = next(i for i, key in enumerate(returned.keys) if key.value == 'ordinals')
    returned.keys[position] = ast.Constant('tie_policy')
    returned.values[position] = ast.Name(id='tie_policy', ctx=ast.Load())
    require(ast.dump(original, include_attributes=False) == ast.dump(actual, include_attributes=False),
            'Complete source training computation changed')
    for name in ('_predict', '_batches', '_digest', '_restore', 'build_from_checkpoint',
                 'prepare_population', 'first_stage_targets', 'second_stage_targets', 'train_epochs', 'blend'):
        require(getattr(candidate, name) is getattr(reference, name), 'Training dependency changed: '+name)


def review():
    reviews = ROOT/'docs/reviews'
    baseline = baseline_review()
    require(baseline == json.loads((reviews/'decomposed_aptos_baseline_review.json').read_text()),
            'Frozen baseline review changed')
    base = json.loads((reviews/'decomposed_aptos_baseline.json').read_text())
    path = reviews/'decomposed_aptos_graph.json'
    full = json.loads(path.read_text())
    expected = dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
        format='aptos-decomposed-graph.v1', native_threads=1, torch_interop_threads=1,
        model_count=8, training_stage_fits=16, first_stage_epochs_per_model=5,
        second_stage_epochs_per_model=10, optimizer_updates=280, reference_augmentation_used=True,
        checkpoint_replays_exact=True, temporary_checkpoints_removed=True,
        target_role_oracle_passed=True, final_ensemble_and_threshold_oracle_passed=True)
    require(all(full.get(k) == v for k, v in expected.items()), 'Complete decomposed training evidence required')
    expected_paths = set(base['sha256']) - {'scripts/validate_decomposed_aptos_baseline.py', 'scripts/aptos_graph_execution.py'} | {
        'scripts/validate_decomposed_aptos_graph.py', 'scripts/decomposed_aptos_graph_execution.py',
        'sciona/decomposed_aptos_pipeline.py', 'sciona/decomposed_aptos_graph.py'}
    require(set(full['sha256']) == expected_paths, 'Complete implementation evidence required')
    check_hashes(full['sha256'])
    require(full['installed_package_versions'] == base['installed_package_versions'] and
            full['artifact_sha256'] == base['artifact_sha256'], 'Qualified environment or model pins differ')
    check_training_body()
    graph = build_decomposed_aptos_graph()
    digest, nodes, edges = encode_execution_graph(graph)
    require((len(nodes), len(edges)) == (5, 7), 'Complete topology differs')
    execution = full['graph_execution']
    require(execution['serialized_graph_sha256'] == digest and execution['actual_runner_nodes'] == 5
        and execution['edges'] == 7 and execution['all_nodes_uncached'] is True
        and execution['codec_roundtrip'] is True, 'Registered graph execution differs')
    wanted = {n.matched_primitive for n in graph.nodes}
    specs = {}
    for repo in ('sciona-atoms', 'sciona-atoms-ml'):
        directory = ROOT.parent/repo
        for spec in _parse_registered_atoms(repo=ProviderRepo(repo, directory), artifact_root=directory/'src/sciona/atoms'):
            runtime = spec.import_module+'.'+spec.source_symbol
            if runtime in wanted:
                require(runtime not in specs, 'Ambiguous registered provider')
                specs[runtime] = spec
    require(set(specs) == wanted and len(specs) == 5, 'Exact provider registrations missing')
    records, interfaces = [], {}
    for runtime, spec in sorted(specs.items()):
        function = getattr(importlib.import_module(spec.import_module), spec.source_symbol)
        require(Path(inspect.getfile(function)).resolve() == spec.file_path.resolve(), 'Provider implementation origin differs')
        key = ('generic_provider_sha256' if runtime == GENERIC else
            'population_provider_sha256' if runtime.endswith('.aptos_population') else 'provider_sha256')
        require(sha(spec.file_path) == execution[key], 'Provider implementation changed')
        if runtime.endswith('.aptos_population'):
            require(execution[key] == baseline['provider_sha256'], 'Existing population provider changed')
        bound = [n for n in graph.nodes if n.matched_primitive == runtime]
        for node in bound:
            interfaces[node.node_id] = provider_ports(node, function)
        records.append(dict(runtime_fqdn=runtime, catalog_fqdn=spec.fqdn, version_id=str(spec.version_id),
            content_hash=spec.content_hash, provider_sha256=execution[key], signature=str(inspect.signature(function)),
            bound_nodes=[n.node_id for n in bound], classification='generic_operation' if runtime == GENERIC else 'source_recipe_adapter'))
    boundary = boundary_ports(graph, interfaces)
    from scripts.build_aptos_execution_graph import build_graph
    original = build_graph()
    expected_inputs = original.nodes[0].inputs + [p for p in original.nodes[1].inputs if p.name != 'plan']
    require([p.model_dump() for p in boundary['input']] == [p.model_dump() for p in expected_inputs],
            'Source root input contracts differ')
    require([p.name for p in boundary['output']] == ['result'], 'Source root output differs')
    return dict(format='aptos-decomposed-review.v1', review_source='automated', proposed_tier=3,
        approved=False, catalog_mutations=0, graph_digest=digest, graph_nodes=5, graph_edges=7,
        providers=records, root_inputs=[p.name for p in boundary['input']], root_outputs=['result'],
        complete_training_ast_equivalent=True, source_stage_mapping=baseline['source_stage_mapping'],
        evidence_sha256={p.name: sha(p) for p in (path, reviews/'decomposed_aptos_baseline_review.json')},
        auxiliary_sha256={name: sha(ROOT/name) for name in ('scripts/review_decomposed_aptos.py',
            'tests/test_decomposed_aptos_cut_indices.py', 'tests/test_decomposed_aptos_graph.py',
            'tests/test_decomposed_aptos_provider_intake.py', 'tests/test_decomposed_aptos_review.py')},
        limitations=baseline['original_conditions']+[
            'The training body is AST-equivalent to the qualified reference apart from returning the equality policy and moving final cut indexing into the graph.',
            'Full graph validation checks complete fits, optimizer updates, checkpoint replay, target roles and the ensemble/cut oracle; it does not compare all trained tensors across independent runs.',
            'Cut indexing is domain independent; fixed source cuts, image models, target roles and learner settings remain recipe policy.',
            'Intermediate result dictionaries must come from the same invocation; the result encoder alone cannot establish training provenance.',
            'Provider publication, corrected original intake and served selection remain separate requirements.'])


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/aptos_decomposed_semantic_review.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('approved', 'catalog_mutations', 'graph_nodes', 'graph_edges')}))
