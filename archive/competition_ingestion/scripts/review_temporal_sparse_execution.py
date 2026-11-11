"""Automated Tier 3 review of the complete generic temporal_sparse topology."""
import hashlib
import importlib.metadata as metadata
import inspect
import json
import platform
from pathlib import Path
from packaging.requirements import Requirement
ROOT=Path(__file__).resolve().parents[1]
SOURCE_VERSION='ad8e992f-1e5d-530c-9475-c9caf4a08bad'
SOURCE_HASH='5ee3659042611115b7a83dbf165f7b44355a0f09ff346514ce4dd2a15e66340b'
SCOPE='Six-stage temporal sparse regression specialization; independent bounded streaming SGD ensemble with fixed one-pass training.'
SUFFIXES=['source_triage','runtime_tests','graph_execution','environment']
LIMITATIONS=['Automated Tier 3 Community only; no Tier 1 human certification or Tier 2 usage qualification. Original intake remains draft with mandatory provenance.', 'Six-stage nonnegative temporal regression specialization. Uses allowed sparse linear models and hashed encoding; fixed one-pass training omits intake early stopping. No historical winning recipe or empirical accuracy guarantee.', 'Caller supplies private globally time-ordered JSONL event streams and positive resource limits. All source snapshots verified before fitting. Bytes, row parsing and output predictions are bounded by explicit caller controls; no measured high-volume throughput qualification.', 'Targets become available immediately after their event timestamp. Same-time labels remain invisible across chunks; latest timestamp ties are averaged. Delayed labels unsupported. Entity keys retained after history expiration; capacity exhaustion fails closed.', 'Fixed-width unsigned hashing has intentional collisions. Features comprise value, seasonal sine/cosine, rolling and latest-time target means, history age/count and presence. Fixed clipping is applied to CSR features without learned population statistics.', 'Strict chronological train/validation/query separation with configured gaps. Validation and query never supply targets to training history or update estimators. Validation RMSLE is descriptive only; no rolling refit, early stopping or model selection.', 'Two seeded online SGD regressors with squared-error and Huber losses on log1p targets. One pass with fixed learning rate and L2 penalty. Training depends on chunk configuration; tested chunk invariance applies to features and prediction only. Equal log-score mean clipped before inverse transform.', 'Synthetic isolation, numerical and serialized lifecycle evidence only. Private in-process prepared descriptors; source paths and digests must not be published. Pinned installed CPU dependencies and notices retained; clean-install, cross-platform and resource-performance behavior unqualified.']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def check_hashes(base,hashes):
    assert hashes
    for name,digest in hashes.items():
        path=(base/name).resolve()
        assert path.is_relative_to(base.resolve()) and path.is_file() and sha(path)==digest


def audit():
    if not __debug__:raise RuntimeError('Assertions required for review')
    reviews=ROOT/'docs/reviews'
    docs={s:json.loads((reviews/f'competition_temporal_sparse_{s}.json').read_text()) for s in SUFFIXES}
    source=docs['source_triage'];assert source['source_version_id']==SOURCE_VERSION and source['source_content_hash']==SOURCE_HASH and source['source_scope']==SCOPE
    for suffix in SUFFIXES[1:]:assert docs[suffix]['status']=='passed'
    tests=docs['runtime_tests'];assert tests['tests_passed']==36 and tests['serialized_boundary_tests']==13
    assert tests['source_version_id']==SOURCE_VERSION and tests['source_content_hash']==SOURCE_HASH
    assert tests['stage_mapping']=={'efficient_table_loading': 'Bounded JSONL reads and private immutable disk snapshots', 'temporal_feature_engineering': 'Strict-past per-entity rolling/latest-time target means, age/count and periodic features', 'categorical_sparse_encoding': 'Fixed-width unsigned entity/category hashing, eight numeric features and clipped CSR chunks', 'leakage_safe_validation': 'Chronological train/validation/query gaps checked before fitting; frozen held-out history', 'boosted_sparse_ensemble': 'One-pass squared-error and Huber sparse SGD on log1p targets with equal log-score mean', 'metric_postprocessing': 'Clipped inverse transform, streamed RMSLE and bounded predictions'}
    assert tests['checks']=={'all_snapshots_verified_before_fit': True, 'source_mutation_rejected': True, 'summary_mutation_rejected': True, 'limits_checked_before_snapshot': True, 'validation_labels_isolated': True, 'bounded_prediction_output': True, 'snapshots_closed_on_failure': True, 'repeatable_execution': True, 'private_runtime_paths_excluded_from_results': True}
    check_hashes(ROOT,tests['sha256'])
    for suffix, count in [('components',8),('training',5),('source_tests',10)]:
        document=json.loads((reviews/f'competition_temporal_sparse_{suffix}.json').read_text())
        assert document['status']=='passed' and document['tests_passed']==count
        assert document['source_version_id']==SOURCE_VERSION and document['source_content_hash']==SOURCE_HASH
        check_hashes(ROOT,document['sha256'])
    execution=docs['graph_execution']
    assert execution['checks']=={'actual_runner_nodes': 2, 'training_rows': 30, 'validation_rows': 10, 'query_rows': 2, 'models': 2, 'updates_per_model': 5, 'feature_count': 24, 'bounded_predictions': True, 'finite_rmsle': True, 'strict_json_output': True, 'graph_codec_roundtrip': True, 'provider_witness_contracts': True}
    check_hashes(ROOT,execution['code_sha256'])
    assert {str(p.relative_to(ROOT)) for p in (ROOT/'sciona').glob('temporal_sparse_*.py')}<=set(execution['code_sha256'])
    from sciona.temporal_sparse_graph import build_temporal_sparse_graph
    from sciona.services.execution_graph_codec import encode_execution_graph
    import sciona.atoms.ml.temporal_sparse_execution as provider
    graph=build_temporal_sparse_graph()
    assert encode_execution_graph(graph)[0]==execution['serialized_graph_sha256']
    assert graph.metadata['source_version_ids']==[SOURCE_VERSION]
    witness={}
    for node in graph.nodes:
        function=getattr(provider,'temporal_sparse_'+node.node_id)
        assert list(inspect.signature(function).parameters)==[p.name for p in node.inputs]
        assert all(p.required for p in node.inputs)
        witness=getattr(provider,'witness_temporal_sparse_'+node.node_id)(witness)
    assert witness=={'kind':'TemporalSparse.Result'}
    provider_path=Path(inspect.getfile(provider));assert sha(provider_path)==execution['provider_sha256']
    env=docs['environment'];requirements=ROOT/'requirements/temporal-sparse-execution.txt'
    dependencies=dict(line.split('==') for line in requirements.read_text().splitlines())
    assert dependencies==env['dependency_versions'] and sha(requirements)==env['requirements_sha256']
    assert set(dependencies)=={'numpy','scipy','scikit-learn','joblib','threadpoolctl'}
    assert env['python_version']==platform.python_version()
    for name,version in dependencies.items():
        assert metadata.version(name)==version
        for text in metadata.requires(name) or []:
            r=Requirement(text)
            if r.marker and not r.marker.evaluate({'extra':''}):continue
            assert r.specifier.contains(metadata.version(r.name),prereleases=True)
    assert set(env['license_sha256'])=={'TemporalSparse-numpy-10-LICENSE.md', 'TemporalSparse-numpy-14-LICENSE.md', 'TemporalSparse-numpy-16-LICENSE.md', 'TemporalSparse-numpy-1-LICENSE.txt', 'TemporalSparse-numpy-4-dragon4_LICENSE.txt', 'TemporalSparse-numpy-6-LICENSE', 'TemporalSparse-numpy-0-LICENSE.txt', 'TemporalSparse-numpy-7-LICENSE.md', 'TemporalSparse-numpy-13-LICENSE.md', 'TemporalSparse-numpy-12-LICENSE.md', 'TemporalSparse-scipy-0-LICENSE.txt', 'TemporalSparse-numpy-11-LICENSE.md', 'TemporalSparse-numpy-3-LICENSE', 'TemporalSparse-threadpoolctl-0-LICENSE', 'TemporalSparse-numpy-8-LICENSE.txt', 'TemporalSparse-numpy-5-LICENSE.md', 'TemporalSparse-scikit-learn-0-COPYING', 'TemporalSparse-numpy-15-LICENSE.md', 'TemporalSparse-numpy-2-COPYING', 'TemporalSparse-numpy-9-LICENSE', 'TemporalSparse-joblib-0-LICENSE.txt'}
    check_hashes(ROOT/'docs/licenses',env['license_sha256'])
    auxiliary=['scripts/review_temporal_sparse_execution.py','requirements/temporal-sparse-execution.txt','docs/reviews/competition_temporal_sparse_references.json']+['docs/licenses/'+name for name in env['license_sha256']]
    return dict(format='temporal_sparse-semantic-review.v1',review_source='automated',proposed_tier=3,verdict='acceptable_with_limits',
        source_version_id=SOURCE_VERSION,source_hash=SOURCE_HASH,source_scope=SCOPE,
        serialized_graph_sha256=execution['serialized_graph_sha256'],provider_sha256=sha(provider_path),
        provider_package_sha256=sha(ROOT.parent/'sciona-atoms-ml/pyproject.toml'),intake_stage_mapping=tests['stage_mapping'],
        dependencies=dependencies,limitations=LIMITATIONS,
        evidence_sha256={f'competition_temporal_sparse_{s}.json':sha(reviews/f'competition_temporal_sparse_{s}.json') for s in SUFFIXES+['components','training','source_tests']},
        auxiliary_sha256={p:sha(ROOT/p) for p in auxiliary},catalog_mutations=0)


if __name__=='__main__':
    result=audit()
    (ROOT/'docs/reviews/competition_temporal_sparse_semantic_review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(verdict=result['verdict'],proposed_tier=3)))
