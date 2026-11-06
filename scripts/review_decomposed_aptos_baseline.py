"""Bridge immutable APTOS source evidence to a fully executed current runner."""
import importlib.metadata as metadata
import inspect
import json
from pathlib import Path

from scripts.review_aptos_execution import ROOT, SOURCE_VERSION, SOURCE_HASH, STAGE_MAPPING, sha, check_hashes
from scripts.build_aptos_execution_graph import build_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.atoms.ml import aptos_execution as provider


def require(condition, message):
    if not condition:
        raise ValueError(message)


def review():
    reviews = ROOT/'docs/reviews'
    original_path = reviews/'competition_aptos_semantic_review.json'
    original = json.loads(original_path.read_text())
    require((original['source_version_id'], original['source_content_hash'], original['stage_mapping']) ==
            (SOURCE_VERSION, SOURCE_HASH, STAGE_MAPPING), 'Original source semantics differ')
    require(original['proposed_tier'] == 3 and original['verdict'] == 'acceptable_with_limits',
            'Qualified original review required')
    check_hashes(original['evidence_sha256'])
    for name in original['evidence_sha256']:
        if name.endswith('.json'):
            evidence = json.loads((ROOT/name).read_text())
            if 'sha256' in evidence:
                check_hashes(evidence['sha256'])
    baseline_path = reviews/'decomposed_aptos_baseline.json'
    baseline = json.loads(baseline_path.read_text())
    expected = dict(passed=True, approved=False, catalog_mutations=0, synthetic_only=True,
        format='aptos-current-runner-baseline.v1', native_threads=1, torch_interop_threads=1,
        model_count=8, training_stage_fits=16, first_stage_epochs_per_model=5,
        second_stage_epochs_per_model=10, optimizer_updates=280, reference_augmentation_used=True,
        checkpoint_replays_exact=True, temporary_checkpoints_removed=True,
        target_role_oracle_passed=True, final_ensemble_and_threshold_oracle_passed=True)
    require(all(baseline.get(k) == v for k, v in expected.items()), 'Complete current baseline required')
    historical = {}
    for suffix, digest in original['trained_execution_evidence'].items():
        path = reviews/('competition_aptos_'+suffix+'.json')
        require(sha(path) == digest, 'Historical execution evidence changed')
        evidence = json.loads(path.read_text())
        historical[suffix] = evidence
        hashes = dict(evidence['sha256'])
        # The current full execution explicitly replaces only this stale runtime anchor.
        if suffix == 'graph_pipeline_validation':
            hashes.pop('sciona/visualizer/runner.py')
        check_hashes(hashes)
    old_graph = historical['graph_pipeline_validation']
    expected_paths = set(old_graph['sha256']) - {'scripts/validate_aptos_graph_pipeline.py'} | {
        'scripts/validate_decomposed_aptos_baseline.py'}
    require(set(baseline['sha256']) == expected_paths, 'Current implementation coverage differs')
    check_hashes(baseline['sha256'])
    digest, nodes, edges = encode_execution_graph(build_graph())
    require((digest, len(nodes), len(edges)) == (original['serialized_graph_sha256'], 2, 1),
            'Reference graph changed')
    require(baseline['graph_execution'] == dict(serialized_graph_sha256=digest,
        provider_sha256=original['provider_sha256'], actual_runner_nodes=2, codec_roundtrip=True),
        'Current registered provider execution differs')
    require(sha(Path(inspect.getfile(provider))) == original['provider_sha256'], 'Provider source differs')
    pins = json.loads((reviews/'competition_aptos_pretrained_mirror_pins.json').read_text())['artifacts']
    require(baseline['artifact_sha256'] == {k: v['sha256'] for k, v in pins.items()}, 'Model pins differ')
    environment = json.loads((reviews/'competition_aptos_environment_audit.json').read_text())
    installed = {name: metadata.version(name) for name in environment['packages']}
    require(installed == baseline['installed_package_versions'] ==
        {name: row['version'] for name, row in environment['packages'].items()}, 'Package version drift')
    package_path = reviews/'decomposed_aptos_package_files.json'
    package = json.loads(package_path.read_text())
    check_hashes(package['sha256'])
    require(package['all_recorded_hashes_match'] and set(package['packages']) == set(installed),
            'Current package-file qualification incomplete')
    require(all(row['version'] == installed[name] and not row['failures'] and row['matched_hashed_files'] > 0
        for name, row in package['packages'].items()), 'Installed package-file qualification failed')
    return dict(format='aptos-current-baseline-review.v1', approved=False, catalog_mutations=0,
        proposed_tier=3, source_version_id=SOURCE_VERSION, source_content_hash=SOURCE_HASH,
        provider_sha256=original['provider_sha256'], serialized_graph_sha256=digest,
        source_stage_mapping=STAGE_MAPPING, original_conditions=original['conditions'],
        current_runner_qualified=True, model_count=8, training_stage_fits=16, optimizer_updates=280,
        evidence_sha256={str(p.relative_to(ROOT)): sha(p) for p in
            (original_path, baseline_path, package_path, Path(__file__).resolve())},
        limitations=['This bridges the unchanged two-node reference only; reusable decomposition, native provider eligibility and original CDG publication remain separate requirements.',
            'Installed files were compared with their own distribution RECORD; this does not independently authenticate the publisher.',
            'All prior reconstruction, external-artifact use and synthetic-only qualification conditions remain in force.'])


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/decomposed_aptos_baseline_review.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('approved', 'catalog_mutations', 'current_runner_qualified',
        'model_count', 'training_stage_fits', 'optimizer_updates')}))
