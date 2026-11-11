"""Fail-closed semantic evidence review for the reusable correction subgraph."""
import hashlib
import inspect
import json
from pathlib import Path

import sciona.atoms.ml.calibration.conditional_residuals as provider
from sciona.conditional_correction_graph import build_conditional_correction_graph
from sciona.ghost.abstract import AbstractArray, AbstractScalar
from sciona.ghost.dimensions import DimensionalSignature
from sciona.services.execution_graph_codec import encode_execution_graph
from scripts.review_conditional_correction_environment import audit as environment_audit

ROOT = Path(__file__).resolve().parents[1]
SOURCE_VERSION = '7d6f7b78-1dff-5757-9326-4edad55aaa68'
SOURCE_HASH = '5a9856b1f6a4db73e7ce12809248de193b63ad06401a6e73b8322744cb7c44bd'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    reviews = ROOT / 'docs/reviews'
    execution_path = reviews / 'conditional_correction_reuse.json'
    execution = json.loads(execution_path.read_text())
    expected_files = {'sciona/conditional_correction_graph.py', 'scripts/validate_conditional_correction_reuse.py',
                      'tests/test_conditional_correction.py'}
    require(set(execution['implementation_sha256']) == expected_files, 'Incomplete implementation evidence')
    for name, digest in execution['implementation_sha256'].items():
        require(sha(ROOT / name) == digest, 'Implementation evidence drift: ' + name)
    require(sha(Path(provider.__file__)) == execution['provider_sha256'], 'Provider evidence drift')
    require(execution['passed'] is True and execution['synthetic_only'] is True, 'Execution evidence missing')
    require(execution['source_float64_cases'] == 64, 'Source comparison evidence missing')
    require(execution['source_sha256'] == {
        'Train_Models.source.py': '2f8c7b7eb6c6c174128c37bc7d53c1db3236a9320699017f02387b406964fc03',
        'Run_Inference.source.py': '5d78b1dd59c14597deac54ce482ef08150eb6cc4dad99166cc8dfd7a49374c44'}, 'Source identity differs')
    require(execution['scenarios'] == [dict(domain=d, calibration_examples=5, application_examples=3,
            actual_runner_completed=True) for d in ['manufacturing_dimensions', 'energy_forecasts']], 'Cross-domain execution missing')
    graph = build_conditional_correction_graph()
    digest, nodes, edges = encode_execution_graph(graph)
    require(digest == execution['serialized_graph_sha256'] and (len(nodes), len(edges)) == (2, 1), 'Graph evidence drift')
    require(graph.metadata['source_version_ids'] == [SOURCE_VERSION] and
            graph.metadata['source_content_hashes'] == [SOURCE_HASH], 'Intake provenance differs')
    require('Complete original competition pipeline' in graph.metadata['exclusions'], 'Partial scope exclusion missing')
    units = DimensionalSignature(L=1)
    offsets = provider.witness_estimate_conditional_offsets(
        AbstractArray(shape=(5,), dim=units), AbstractArray(shape=(5,), dim=units),
        AbstractArray(shape=(5,)), AbstractScalar(dtype='float64'))
    corrected = provider.witness_apply_conditional_offsets(AbstractArray(shape=(3,), dim=units),
        AbstractArray(shape=(3,)), offsets, AbstractScalar(dtype='float64'))
    require(offsets.shape == (2,) and corrected.shape == (3,) and corrected.dim == units, 'Witness composition differs')
    for node in graph.nodes:
        function = getattr(provider, node.matched_primitive.rsplit('.', 1)[1])
        signature = inspect.signature(function)
        require(list(signature.parameters) == [p.name for p in node.inputs], 'Callable interface differs')
        require(all(p.default is inspect.Parameter.empty for p in signature.parameters.values()), 'Implicit parameter default')
        require(all(p.constraints and p.required for p in node.inputs + node.outputs), 'Port contract missing')
    environment_path = reviews / 'conditional_correction_environment.json'
    environment = json.loads(environment_path.read_text())
    require(environment == environment_audit(), 'Environment evidence drift')
    blockers = []
    if not environment['transitive_dependency_closure']['compatible']:
        blockers.append('Declared transitive dependency closure incomplete')
    if not environment['installed_provider_metadata_matches_source']:
        blockers.append('Installed provider metadata differs from source manifest')
    blockers.append('Negative publication gates and transactional rollback/apply/idempotency/served verification pending')
    return dict(format='conditional-correction-semantic-review.v1', proposed_tier=3,
        semantic_verdict='acceptable_with_limits', publication_ready=False, approved=False,
        source_version_id=SOURCE_VERSION, source_hash=SOURCE_HASH, scope=graph.metadata['scope'],
        applicability=graph.metadata['applicability'], exclusions=graph.metadata['exclusions'],
        limitations=execution['limits'], publication_blockers=blockers,
        serialized_graph_sha256=digest, provider_sha256=execution['provider_sha256'],
        witness_composition_verified=True,
        evidence_sha256={p.name: sha(p) for p in [execution_path, environment_path]},
        reviewer_sha256=sha(Path(__file__)), catalog_mutations=0)


if __name__ == '__main__':
    report = audit()
    (ROOT / 'docs/reviews/conditional_correction_semantic_review.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(semantic_verdict=report['semantic_verdict'], publication_ready=report['publication_ready'],
                          blockers=report['publication_blockers'])))
