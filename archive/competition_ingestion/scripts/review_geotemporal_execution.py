"""Automated Tier 3 review of the complete generic geotemporal topology."""
import hashlib
import importlib.metadata as metadata
import inspect
import json
import platform
from pathlib import Path
from packaging.requirements import Requirement
ROOT=Path(__file__).resolve().parents[1]
SOURCE_VERSION='9bb6428c-0529-56c8-b51d-96e1b4d6990c'
SOURCE_HASH='b22878a5b3fa77fcf721b4343515a4284b9dd7777758bd9e86b1ed56e5c6790e'
SCOPE='Generic five-stage geospatial/spatiotemporal fusion topology; independent causal planar regression realization.'
SUFFIXES=['source_triage','runtime_tests','graph_execution','environment']
LIMITATIONS=['Automated Tier 3 Community only; no Tier 1 human certification or Tier 2 usage qualification. Original generic intake remains draft with mandatory provenance.', 'Complete five-stage causal planar regression realization: context join, spatial/temporal features, forward validation, ExtraTrees fusion and geographic smoothing. Independent implementation; synthetic qualification only, no historical winning recipe or empirical accuracy claim.', 'Caller declares shared planar metric coordinates and seconds clock with shared origin. No coordinate transformation, clock alignment or proof of the declaration. Duplicate space/time rows rejected separately per population.', 'Context time denotes availability, not merely event time: latest available value per site within lookback/radius, then nearest-site mean with coordinate tie order. Late-reported or revised context must use correct availability timestamps supplied by caller.', 'Target lags use only strict-past observations within radius/lookback. Training features exclude own-time targets; validation features see only fitting-fold history, with no sequential update from held-out labels. Missing aggregates use zero plus explicit availability indicators.', 'Expanding contiguous time blocks with exclusion gap; block zero is fitting-only warmup with no validation prediction. Later folds legitimately use earlier validation labels as history. Temporal validation permits repeated locations; no unseen-location generalization guarantee.', 'ExtraTrees regressor per forward fold and full-data refit with fixed controls. Held-out MSE covers only post-warmup predictions after geographic smoothing. No hyperparameter optimization or unbiased performance guarantee beyond supplied split assumptions.', 'Nonnegative clipping precedes convex same-time neighbor smoothing. Query outputs intentionally depend on co-timed query population; different times never smooth together. No mass conservation, map matching or query-population independence claim.', 'Naive in-memory spatial joins, pinned provisioned dependencies and installed notices. No large-scale, clean-install or cross-platform qualification. Strict finite JSON boundary; fitted objects trusted in-process and no portable serialization claim.']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def check_hashes(base,hashes):
    assert hashes
    for name,digest in hashes.items():
        path=(base/name).resolve()
        assert path.is_relative_to(base.resolve()) and path.is_file() and sha(path)==digest


def audit():
    if not __debug__:raise RuntimeError('Assertions required for review')
    reviews=ROOT/'docs/reviews'
    docs={s:json.loads((reviews/f'competition_geotemporal_{s}.json').read_text()) for s in SUFFIXES}
    source=docs['source_triage'];assert source['source_version_id']==SOURCE_VERSION and source['source_content_hash']==SOURCE_HASH and source['source_scope']==SCOPE
    for suffix in SUFFIXES[1:]:assert docs[suffix]['status']=='passed'
    tests=docs['runtime_tests'];assert tests['tests_passed']==29 and tests['serialized_boundary_tests']==11
    assert tests['source_version_id']==SOURCE_VERSION and tests['source_content_hash']==SOURCE_HASH
    assert tests['stage_mapping']=={'geo_temporal_join': 'Latest available spatial context per site within radius/lookback', 'spatial_temporal_features': 'Planar coordinates, seasonal phase, distances, context age, strict-past target lags and missingness', 'leakage_safe_geo_validation': 'Expanding time blocks, explicit gap, fitting-only warmup and fold-local target history', 'fusion_model': 'ExtraTrees regressor for each forward fold followed by full-training refit', 'geographic_postprocessing': 'Nonnegative clipping and same-time spatial-neighbor smoothing'}
    assert tests['checks']=={'hand_causal_join': True, 'future_context_isolation': True, 'heldout_label_isolation': True, 'forward_gap_validation': True, 'warmup_explicitly_unvalidated': True, 'full_training_refit': True, 'heldout_mse_check': True, 'geographic_smoothing_hand_check': True, 'coordinate_clock_contract': True, 'strict_json_boundary': True, 'mutated_intermediate_rejected_before_fit': True}
    check_hashes(ROOT,tests['sha256'])
    execution=docs['graph_execution']
    assert execution['checks']=={'actual_runner_nodes': 2, 'training_rows': 9, 'validation_rows': 6, 'warmup_rows': 3, 'query_rows': 2, 'folds': 2, 'trees': 32, 'feature_count': 11, 'nonnegative_predictions': True, 'strict_json_output': True, 'graph_codec_roundtrip': True, 'provider_witness_contracts': True}
    check_hashes(ROOT,execution['code_sha256'])
    assert {str(p.relative_to(ROOT)) for p in (ROOT/'sciona').glob('geotemporal_*.py')}<=set(execution['code_sha256'])
    from sciona.geotemporal_graph import build_geotemporal_graph
    from sciona.services.execution_graph_codec import encode_execution_graph
    import sciona.atoms.ml.geotemporal_execution as provider
    graph=build_geotemporal_graph()
    assert encode_execution_graph(graph)[0]==execution['serialized_graph_sha256']
    assert graph.metadata['source_version_ids']==[SOURCE_VERSION]
    witness={}
    for node in graph.nodes:
        function=getattr(provider,'geotemporal_'+node.node_id)
        assert list(inspect.signature(function).parameters)==[p.name for p in node.inputs]
        assert all(p.required for p in node.inputs)
        witness=getattr(provider,'witness_geotemporal_'+node.node_id)(witness)
    assert witness=={'kind':'Geotemporal.Result'}
    provider_path=Path(inspect.getfile(provider));assert sha(provider_path)==execution['provider_sha256']
    env=docs['environment'];requirements=ROOT/'requirements/geotemporal-execution.txt'
    dependencies=dict(line.split('==') for line in requirements.read_text().splitlines())
    assert dependencies==env['dependency_versions'] and sha(requirements)==env['requirements_sha256']
    assert set(dependencies)=={'scipy', 'numpy', 'threadpoolctl', 'scikit-learn', 'joblib'}
    assert env['python_version']==platform.python_version()
    for name,version in dependencies.items():
        assert metadata.version(name)==version
        for text in metadata.requires(name) or []:
            r=Requirement(text)
            if r.marker and not r.marker.evaluate({'extra':''}):continue
            assert r.specifier.contains(metadata.version(r.name),prereleases=True)
    assert set(env['license_sha256'])=={'Geotemporal-scipy-0-LICENSE.txt', 'Geotemporal-numpy-13-LICENSE.md', 'Geotemporal-numpy-16-LICENSE.md', 'Geotemporal-threadpoolctl-0-LICENSE', 'Geotemporal-numpy-8-LICENSE.txt', 'Geotemporal-numpy-1-LICENSE.txt', 'Geotemporal-numpy-15-LICENSE.md', 'Geotemporal-numpy-5-LICENSE.md', 'Geotemporal-numpy-11-LICENSE.md', 'Geotemporal-joblib-0-LICENSE.txt', 'Geotemporal-numpy-6-LICENSE', 'Geotemporal-numpy-14-LICENSE.md', 'Geotemporal-numpy-9-LICENSE', 'Geotemporal-numpy-0-LICENSE.txt', 'Geotemporal-numpy-4-dragon4_LICENSE.txt', 'Geotemporal-numpy-7-LICENSE.md', 'Geotemporal-numpy-12-LICENSE.md', 'Geotemporal-numpy-2-COPYING', 'Geotemporal-scikit-learn-0-COPYING', 'Geotemporal-numpy-10-LICENSE.md', 'Geotemporal-numpy-3-LICENSE'}
    check_hashes(ROOT/'docs/licenses',env['license_sha256'])
    auxiliary=['scripts/review_geotemporal_execution.py','requirements/geotemporal-execution.txt']+['docs/licenses/'+name for name in env['license_sha256']]
    return dict(format='geotemporal-semantic-review.v1',review_source='automated',proposed_tier=3,verdict='acceptable_with_limits',
        source_version_id=SOURCE_VERSION,source_hash=SOURCE_HASH,source_scope=SCOPE,
        serialized_graph_sha256=execution['serialized_graph_sha256'],provider_sha256=sha(provider_path),
        provider_package_sha256=sha(ROOT.parent/'sciona-atoms-ml/pyproject.toml'),intake_stage_mapping=tests['stage_mapping'],
        dependencies=dependencies,limitations=LIMITATIONS,
        evidence_sha256={f'competition_geotemporal_{s}.json':sha(reviews/f'competition_geotemporal_{s}.json') for s in SUFFIXES},
        auxiliary_sha256={p:sha(ROOT/p) for p in auxiliary},catalog_mutations=0)


if __name__=='__main__':
    result=audit()
    (ROOT/'docs/reviews/competition_geotemporal_semantic_review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(verdict=result['verdict'],proposed_tier=3)))
