"""Automated Tier 3 review of the source-bound OpenVaccine reconstruction."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCES=[
 dict(version_id='9510a335-f191-5642-9ae1-20459e7d4eba',content_hash='5a6c3af650d1ea240d4b211a57c69d1136fe68f1780250cbb47bab790060c121',stages=6),
 dict(version_id='bc2e55f3-60b8-56a3-8519-87f31ed27246',content_hash='a5b83b0ed23f1c38cc1212632e1ac01906c4b2553a29c411ebdcd3d4325ab32c',stages=5)]
COMMITS=dict(reconstruction='b95af835cc38bd7dc55ed6e7397d124787f86996',folding_engine='64f01d52f8fcba2b88c0011d56360554b9a95012',folding_parameters='87b9aac55cee14fd562049d08f7b92d3131f10ce')
COMPONENTS={'runtime':'models','sections':'sections','ensemble':'ensemble','checkpoint_runtime':'checkpoints','training':'training','teacher':'teacher','structure':'structure','feature_runtime':'features','folding_runtime':'folding'}
SUFFIXES=['source_pins','source_probe','features','preprocessing','tensorflow_loss','models',*COMPONENTS,
 'folding_dispatch','folding_engine','eternafold_parameters','eternafold_engine','contract','graph_execution','environment']
LIMITATIONS=[
 'Automated Tier 3 Community corrected reconstruction. Tier 1 human certification and Tier 2 usage qualification are not claimed. Both original inaccurate intakes remain draft and are mandatory provenance dependencies.',
 'The DasLab reconstruction uses twenty TensorFlow models: five states each of GRU, LSTM, forward and wave heads sharing an autoencoder-pretrained graph base. This is not the original winning clustered final blend or the intake PyTorch/PyG parallel-branch description.',
 'Recovered source is inference-only. Training is reconstructed from the primary winner writeup with caller-defined populations, cluster IDs, proximity factors, splits, step counts, uncertainty eligibility/noise draws, reversal flags, rollback tolerance/policy and standard-deviation convention. Historical optimization schedule, noise distribution and accuracy are not reproduced.',
 'Raw supported sequences are folded by the exact reviewed macOS arm64 CONTRAfold-SE executable with separately licensed EternaFold coefficients, default MEA decoding and posterior cutoff 1e-10. No independent solver or historical competition-output parity is claimed.',
 'Features retain 55 node and eight adjacency channels, including capped Manhattan distance propagation rather than general graph shortest paths. Strict sequence/structure/probability guards and per-record feature batching correct source boundary and batch failures.',
 'All-member clipping corrects the source first-member exemption and order dependence. Predictions reverse-average each model and average clipped member outputs; teacher moments use unclipped outputs and caller-selected ddof.',
 'Source weighted per-sample per-target RMSE semantics include masked positions in the denominator and an epsilon floor. Null denotes missing serialized targets; nonfinite observed values are rejected. Population and fold construction are caller responsibilities, not certified independent validation.',
 'Shared autoencoder pretraining uses the full labeled feature population, including member validation features. Pseudo-label teachers share population information; this is transductive and must not be presented as independent cross-validation accuracy.',
 'Initial supervised and pretraining step counts are explicit. Each refinement round performs one full supervised batch and one pseudo-label batch per member. Validation may roll back model weights only or model plus optimizer, as explicitly selected. Exceptions in a refinement section restore both; stochastic RNG and iterators are not rewound.',
 'Checkpoint files are private temporary runtime state and are removed after population execution. Native checkpoints preserve model and Adam state, not complete stochastic replay, external input provenance or interrupted whole-population resumption.',
 'The execution seed changes process-global random state. Use a dedicated worker process; concurrent RNG isolation is not certified. Full graph evidence uses synthetic inputs, one refinement round and random initial weights, with no competition predictor weights or real records.',
 'Pinned dependency closure and actual imports are qualified on the provisioned macOS arm64 Python 3.13 worker with legacy Keras. Clean installation, unrelated installed packages, GPU/mixed precision, other operating systems and resource limits are not qualified.',
 'DasLab MIT and folding software/parameter licenses are retained. The linked notebook has no established reuse license and is comparison evidence only; its code is not embedded in this implementation.'
]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def check_hashes(base,hashes):
    assert hashes
    for name,digest in hashes.items():
        path=(base/name).resolve()
        assert path.is_relative_to(base.resolve()) and path.is_file() and sha(path)==digest


def audit():
    if not __debug__:raise RuntimeError('Assertions required for semantic review')
    reviews=ROOT/'docs/reviews'
    docs={s:json.loads((reviews/f'competition_openvaccine_{s}.json').read_text()) for s in SUFFIXES}
    source=docs['source_pins']
    assert source['commit']==COMMITS['reconstruction'] and source['license']['spdx_id']=='MIT'
    from sciona.openvaccine_models import SOURCE_SHA256,LICENSE_SHA256
    from sciona.openvaccine_folding import BINARY_SHA256,PARAMETERS_SHA256
    pins={p['software_path']:p['sha256'] for p in source['pins']}
    assert pins['scripts/nullrecurrent_inference.py']==SOURCE_SHA256 and pins['LICENSE']==LICENSE_SHA256
    assert sha(ROOT/'docs/licenses/OpenVaccine-DasLab-MIT.txt')==LICENSE_SHA256
    for suffix,d in docs.items():
        if suffix not in ('source_pins','eternafold_parameters'):assert d['status']=='passed'
        if 'validator_sha256' in d:assert sha(ROOT/f'scripts/validate_openvaccine_{suffix}.py')==d['validator_sha256']
    for report,module in COMPONENTS.items():assert docs[report]['runtime_sha256']==sha(ROOT/f'sciona/openvaccine_{module}.py')
    assert docs['source_probe']['training_fit_calls']==0
    assert docs['source_probe']['first_ensemble_member_unclipped'] and docs['source_probe']['ensemble_order_dependence_reproduced']
    assert docs['runtime']['modified_source_rejected'] and docs['runtime']['invalid_training_cases']==20
    assert len(docs['runtime']['families'])==5 and all(r['direct_source_output_exact'] for r in docs['runtime']['families'])
    models=docs['models']['full_topologies']
    assert {r['family'] for r in models}=={'autoencoder','gru','lstm','forward','wave'}
    assert all(r['parameters']>9_000_000 and r['finite_gradients'] and r['weights_changed'] for r in models)
    assert docs['tensorflow_loss']['masked_gradients_zero']
    sections=docs['sections'];assert sections['post_update_exception_restores_model_optimizer']
    assert {r['rollback_policy'] for r in sections['injected_metric_rollback_cases'] if r['rolled_back']}=={'weights_only','model_and_optimizer'}
    ensemble=docs['ensemble'];assert ensemble['full_model_members']==20 and ensemble['stacked_reference_comparison']=='passed'
    assert all(ensemble[k] for k in ['reverse_averaging','symmetric_clipping_order_comparison','incomplete_duplicate_rejected','caller_inputs_unchanged'])
    training=docs['training'];assert training['partially_observed_validation'] and len(training['families'])==4
    assert all(r['checkpoint_prediction_exact'] and r['pseudo_label_sections']==2 for r in training['families'])
    checkpoint=docs['checkpoint_runtime'];assert checkpoint['prediction_and_model_state_exact'] and checkpoint['optimizer_state_exact']
    teacher=docs['teacher'];assert teacher['trained_checkpoint_members']==20 and teacher['restored_teacher_reference_comparison'] and teacher['missing_duplicate_population_rejected']
    assert docs['structure']['exhaustive_structures']==9359
    features=docs['feature_runtime'];assert features['node_channels']==55 and features['adjacency_channels']==8 and features['independent_reference_comparison']
    folding=docs['folding_engine'];assert folding['engine_commit']==COMMITS['folding_engine'] and folding['binary_sha256']==BINARY_SHA256 and not folding['source_modified']
    engine_license=next(p['sha256'] for p in folding['source_pins'] if p['software_path']=='LICENSE')
    assert sha(ROOT/'docs/licenses/OpenVaccine-CONTRAfold-SE-BSD.txt')==engine_license=='7babc2649b1dc94a31e11525be834c9fca93f3808d4eb61e88718a25efd73af9'
    parameters=docs['eternafold_parameters'];assert parameters['commit']==COMMITS['folding_parameters'] and parameters['sha256']==PARAMETERS_SHA256 and parameters['historical_bytes_match']
    license_pin=next(p['sha256'] for p in parameters['parameter_license_pins'] if p['software_path']=='parameters/LICENSE')
    assert sha(ROOT/'docs/licenses/OpenVaccine-EternaFold-Parameters-BSD.txt')==license_pin
    configured=docs['eternafold_engine'];assert configured['binary_sha256']==BINARY_SHA256 and configured['parameters_sha256']==PARAMETERS_SHA256
    assert configured['configured_engine_cases']==6 and configured['complete_feature_cases']==6 and configured['nondefault_posterior_cases']>0
    runtime=docs['folding_runtime'];assert runtime['direct_engine_comparison_cases']==5 and runtime['changed_dependencies_rejected']==2 and runtime['subprocess_failure_cases']==2
    contract=docs['contract'];assert contract['tests_passed']==81 and contract['serialized_boundary_tests']==13
    check_hashes(ROOT,contract['hashes'])
    graph=docs['graph_execution'];assert graph['checks']==dict(actual_runner_nodes=2,models=20,refinement_rounds=1,masked_validation=True,distinct_population_lengths=True,strict_json_output=True,graph_codec_roundtrip=True,provider_witness_contracts=True)
    assert {str(p.relative_to(ROOT)) for p in (ROOT/'sciona').glob('openvaccine_*.py')}<=set(graph['code_sha256'])
    check_hashes(ROOT,graph['code_sha256'])
    provider=ROOT.parent/'sciona-atoms-ml/src/sciona/atoms/ml/openvaccine_execution.py'
    assert sha(provider)==graph['provider_sha256']
    from sciona.openvaccine_graph import build_openvaccine_graph
    from sciona.services.execution_graph_codec import encode_execution_graph
    assert encode_execution_graph(build_openvaccine_graph())[0]==graph['serialized_graph_sha256']
    from scripts.validate_openvaccine_environment import validate
    environment=docs['environment'];assert environment['dependency_versions']==validate()
    assert environment['declared_dependency_closure_compatible'] and environment['actual_import_versions_match'] and environment['legacy_keras']
    check_hashes(ROOT,environment['sha256'])
    auxiliary=['scripts/review_openvaccine_execution.py','scripts/validate_openvaccine_environment.py','requirements/openvaccine-execution.txt']+[str(p.relative_to(ROOT)) for p in sorted((ROOT/'docs/licenses').glob('OpenVaccine-*.txt'))]
    return dict(format='openvaccine-semantic-review.v1',review_source='automated',proposed_tier=3,verdict='acceptable_with_limits',
        source_versions=SOURCES,source_commits=COMMITS,serialized_graph_sha256=graph['serialized_graph_sha256'],provider_sha256=sha(provider),
        provider_package_sha256=sha(ROOT.parent/'sciona-atoms-ml/pyproject.toml'),dependencies=environment['dependency_versions'],limitations=LIMITATIONS,
        superseded_evidence={'competition_openvaccine_lifecycle.json':'Current serialized graph execution supersedes older lifecycle implementation.',
            'competition_openvaccine_round.json':'Older contract identity; not used to certify current two-round execution. Current graph certifies one refinement round.'},
        evidence_sha256={f'competition_openvaccine_{s}.json':sha(reviews/f'competition_openvaccine_{s}.json') for s in SUFFIXES},
        auxiliary_sha256={p:sha(ROOT/p) for p in auxiliary},catalog_mutations=0)


if __name__=='__main__':
    result=audit()
    (ROOT/'docs/reviews/competition_openvaccine_semantic_review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(verdict=result['verdict'],proposed_tier=3,evidence=len(SUFFIXES))))
