"""Automated Tier 3 review of the complete generic video-multilabel topology."""
import hashlib
import importlib.metadata as metadata
import inspect
import json
import platform
from pathlib import Path
from packaging.requirements import Requirement
ROOT=Path(__file__).resolve().parents[1]
SOURCE_VERSION='13b48adf-8fb5-5ea2-a26c-44344bb8907e'
SOURCE_HASH='30d0444ec640598942c6221136e608d7b15449732ceb848939e1424321c496ad'
SCOPE='Generic five-stage precomputed-feature video multilabel topology; independent attention/context/sparse-fusion realization.'
SUFFIXES=['source_triage','runtime_tests','graph_execution','environment']
LIMITATIONS=['Automated Tier 3 Community only; no Tier 1 human certification or Tier 2 usage qualification. Original generic intake remains draft with mandatory provenance.', 'Complete five-stage precomputed-feature video multilabel realization: batch-padded feature loading, attention pooling, context-gated head, sparse linear fusion and label-wise F1 thresholds. Independent architecture; no historical winning recipe or accuracy claim.', 'Caller supplies finite variable-length frame matrices and nonnegative sparse string-to-number mappings. Padding allocated per batch; complete supplied population and sparse matrix remain in memory. No raw-video feature extraction, disk streaming or large-scale resource qualification.', 'Learned shared frame attention is permutation invariant and contains no temporal position information. Masked frames cannot affect pooling; sigmoid context gate modulates pooled features before multilabel logits.', 'CPU float64 Adam/BCE trains a fixed epoch count with local seeded shuffle. No validation checkpoint selection, early stopping, class weighting or calibrated probability guarantee. Training loss is synthetic fitting evidence only.', 'Training-only DictVectorizer vocabulary feeds one regularized logistic model per label. Unknown sparse keys ignored. Neural/sparse probabilities equally averaged; no rare-label improvement or empirical accuracy guarantee.', 'Meaningful caller-supplied groups must be disjoint across training/calibration/query. Each fitted/calibrated label needs both classes. Separate per-label F1 threshold maximization uses the highest threshold for ties; calibration score would be selection evidence, not unbiased accuracy.', 'Synthetic direct, isolation, boundary and serialized lifecycle checks only. Provisioned runtime dependency closure and installed notices retained; no clean-install, cross-platform or GPU qualification. Strict JSON boundary; intermediate fitted objects trusted in-process.']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def check_hashes(base,hashes):
    assert hashes
    for name,digest in hashes.items():
        path=(base/name).resolve()
        assert path.is_relative_to(base.resolve()) and path.is_file() and sha(path)==digest


def audit():
    if not __debug__:raise RuntimeError('Assertions required for review')
    reviews=ROOT/'docs/reviews'
    docs={s:json.loads((reviews/f'competition_video_multilabel_{s}.json').read_text()) for s in SUFFIXES}
    source=docs['source_triage'];assert source['source_version_id']==SOURCE_VERSION and source['source_content_hash']==SOURCE_HASH and source['source_scope']==SCOPE
    for suffix in SUFFIXES[1:]:assert docs[suffix]['status']=='passed'
    tests=docs['runtime_tests'];assert tests['tests_passed']==36 and tests['serialized_boundary_tests']==11
    assert tests['source_version_id']==SOURCE_VERSION and tests['source_content_hash']==SOURCE_HASH
    assert tests['stage_mapping']=={'feature_stream_loading': 'Caller-supplied precomputed frame matrices padded per batch and sparse mappings in CSR', 'temporal_pooling': 'Masked learned frame attention with explicit order invariance', 'context_gated_multilabel_head': 'Sigmoid context gate and multilabel logits trained by Adam/BCE', 'sparse_baseline_fusion': 'Training-only sparse vocabulary and per-label logistic models; equal neural/sparse score mean', 'rank_ensemble_calibration': 'Separate per-label F1 threshold maximization with highest-threshold tie rule'}
    assert tests['checks']=={'hand_attention_and_gate': True, 'padding_gradient_isolation': True, 'bounded_batch_padding': True, 'training_only_sparse_vocabulary': True, 'equal_neural_sparse_fusion': True, 'calibration_labels_do_not_change_scores': True, 'query_batch_independence': True, 'independent_f1_search': True, 'group_disjointness': True, 'strict_json_boundary': True, 'mutated_intermediate_rejected_before_fit': True}
    check_hashes(ROOT,tests['sha256'])
    execution=docs['graph_execution']
    assert execution['checks']=={'actual_runner_nodes': 2, 'training_rows': 8, 'calibration_rows': 4, 'query_rows': 2, 'epochs': 80, 'label_count': 2, 'sparse_models': 2, 'training_improved': True, 'strict_json_output': True, 'graph_codec_roundtrip': True, 'provider_witness_contracts': True}
    check_hashes(ROOT,execution['code_sha256'])
    assert {str(p.relative_to(ROOT)) for p in (ROOT/'sciona').glob('video_multilabel_*.py')}<=set(execution['code_sha256'])
    from sciona.video_multilabel_graph import build_video_multilabel_graph
    from sciona.services.execution_graph_codec import encode_execution_graph
    import sciona.atoms.ml.video_multilabel_execution as provider
    graph=build_video_multilabel_graph()
    assert encode_execution_graph(graph)[0]==execution['serialized_graph_sha256']
    assert graph.metadata['source_version_ids']==[SOURCE_VERSION]
    witness={}
    for node in graph.nodes:
        function=getattr(provider,'video_multilabel_'+node.node_id)
        assert list(inspect.signature(function).parameters)==[p.name for p in node.inputs]
        assert all(p.required for p in node.inputs)
        witness=getattr(provider,'witness_video_multilabel_'+node.node_id)(witness)
    assert witness=={'kind':'VideoMultilabel.Result'}
    provider_path=Path(inspect.getfile(provider));assert sha(provider_path)==execution['provider_sha256']
    env=docs['environment'];requirements=ROOT/'requirements/video-multilabel-execution.txt'
    dependencies=dict(line.split('==') for line in requirements.read_text().splitlines())
    assert dependencies==env['dependency_versions'] and sha(requirements)==env['requirements_sha256']
    assert set(dependencies)=={'threadpoolctl', 'scikit-learn', 'sympy', 'markupsafe', 'setuptools', 'filelock', 'joblib', 'jinja2', 'networkx', 'typing-extensions', 'fsspec', 'mpmath', 'torch', 'scipy', 'numpy'}
    assert env['python_version']==platform.python_version()
    for name,version in dependencies.items():
        assert metadata.version(name)==version
        for text in metadata.requires(name) or []:
            r=Requirement(text)
            if r.marker and not r.marker.evaluate({'extra':''}):continue
            assert r.specifier.contains(metadata.version(r.name),prereleases=True)
    assert set(env['license_sha256'])=={'VideoMultilabel-numpy-16-LICENSE.md', 'VideoMultilabel-typing-extensions-0-LICENSE', 'VideoMultilabel-torch-0-LICENSE', 'VideoMultilabel-fsspec-0-LICENSE', 'VideoMultilabel-setuptools-14-LICENSE', 'VideoMultilabel-mpmath-0-LICENSE', 'VideoMultilabel-numpy-8-LICENSE.txt', 'VideoMultilabel-numpy-10-LICENSE.md', 'VideoMultilabel-setuptools-7-LICENSE', 'VideoMultilabel-setuptools-8-LICENSE', 'VideoMultilabel-numpy-9-LICENSE', 'VideoMultilabel-numpy-12-LICENSE.md', 'VideoMultilabel-numpy-4-dragon4_LICENSE.txt', 'VideoMultilabel-numpy-5-LICENSE.md', 'VideoMultilabel-numpy-13-LICENSE.md', 'VideoMultilabel-threadpoolctl-0-LICENSE', 'VideoMultilabel-numpy-11-LICENSE.md', 'VideoMultilabel-setuptools-10-LICENSE.BSD', 'VideoMultilabel-setuptools-11-LICENSE', 'VideoMultilabel-setuptools-6-LICENSE', 'VideoMultilabel-setuptools-5-LICENSE', 'VideoMultilabel-numpy-0-LICENSE.txt', 'VideoMultilabel-numpy-3-LICENSE', 'VideoMultilabel-setuptools-13-LICENSE.txt', 'VideoMultilabel-setuptools-4-LICENSE', 'VideoMultilabel-scipy-0-LICENSE.txt', 'VideoMultilabel-setuptools-3-LICENSE', 'VideoMultilabel-setuptools-9-LICENSE.APACHE', 'VideoMultilabel-setuptools-12-LICENSE', 'VideoMultilabel-setuptools-1-LICENSE', 'VideoMultilabel-joblib-0-LICENSE.txt', 'VideoMultilabel-numpy-7-LICENSE.md', 'VideoMultilabel-numpy-1-LICENSE.txt', 'VideoMultilabel-numpy-2-COPYING', 'VideoMultilabel-sympy-0-LICENSE', 'VideoMultilabel-setuptools-0-LICENSE', 'VideoMultilabel-numpy-14-LICENSE.md', 'VideoMultilabel-networkx-0-LICENSE.txt', 'VideoMultilabel-markupsafe-0-LICENSE.txt', 'VideoMultilabel-filelock-0-LICENSE', 'VideoMultilabel-numpy-6-LICENSE', 'VideoMultilabel-numpy-15-LICENSE.md', 'VideoMultilabel-scikit-learn-0-COPYING', 'VideoMultilabel-jinja2-0-LICENSE.txt', 'VideoMultilabel-setuptools-2-LICENSE'}
    check_hashes(ROOT/'docs/licenses',env['license_sha256'])
    auxiliary=['scripts/review_video_multilabel_execution.py','requirements/video-multilabel-execution.txt']+['docs/licenses/'+name for name in env['license_sha256']]
    return dict(format='video-multilabel-semantic-review.v1',review_source='automated',proposed_tier=3,verdict='acceptable_with_limits',
        source_version_id=SOURCE_VERSION,source_hash=SOURCE_HASH,source_scope=SCOPE,
        serialized_graph_sha256=execution['serialized_graph_sha256'],provider_sha256=sha(provider_path),
        provider_package_sha256=sha(ROOT.parent/'sciona-atoms-ml/pyproject.toml'),intake_stage_mapping=tests['stage_mapping'],
        dependencies=dependencies,limitations=LIMITATIONS,
        evidence_sha256={f'competition_video_multilabel_{s}.json':sha(reviews/f'competition_video_multilabel_{s}.json') for s in SUFFIXES},
        auxiliary_sha256={p:sha(ROOT/p) for p in auxiliary},catalog_mutations=0)


if __name__=='__main__':
    result=audit()
    (ROOT/'docs/reviews/competition_video_multilabel_semantic_review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(verdict=result['verdict'],proposed_tier=3)))
